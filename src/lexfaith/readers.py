"""Reader inference on a built benchmark.

    uv run --locked --group readers lexfaith-read --data datasets/squad-dev-p100-v1 \
        --systems pisco,mistral,closedbook --out preds.jsonl

Systems (all greedy, same max_new_tokens, same dtype):
  pisco       naver/pisco-mistral; each variant's 128-token passages are compressed (8 memory
              embeddings each) and decoded with PISCO's own prompt
  mistral     PISCO's decoder base (Mistral-7B-Instruct-v0.2) reading the same passages as text,
              with PISCO's exact system/user prompt -> the only difference is compression
  closedbook  the same Mistral without any background (parametric knowledge / leakage)

Each system runs in its own subprocess so that only one 7B model occupies GPU memory (fits 16 GB).
Predictions are appended as JSONL {system, qid, arm, prediction}; reruns skip finished items.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from lexfaith.io import read_jsonl

PISCO = "naver/pisco-mistral"
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
# Verbatim from PISCO's `blend_prompt_and_memory_tokens` (modelling_pisco.py).
SYSTEM = ("You are a helpful assistant. Your task is to extract relevant information from provided documents "
          "and to answer to questions as briefly as possible.")
SYSTEMS = ("pisco", "mistral", "closedbook")


def load_items(data: Path, arms: list[str] | None, limit: int | None, include_invalid: bool) -> list[dict]:
    items = [r for r in read_jsonl(data / "items.jsonl") if "context" in r]
    if not include_invalid:
        items = [r for r in items if r["valid"]]
    if arms:
        items = [r for r in items if r["arm"] in arms]
    if limit:  # keep whole questions (all arms) so paired comparisons survive
        qids = sorted({r["qid"] for r in items})[:limit]
        items = [r for r in items if r["qid"] in set(qids)]
    return items


def chat_prompt(tokenizer, user: str) -> str:
    """Same fallback as PISCO: Mistral-v0.2's template has no system role, so it is merged."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        if "System role not supported" not in str(e) and "roles must alternate" not in str(e):
            raise
        msgs = [{"role": "user", "content": SYSTEM + "\n" + user}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def mistral_user(item: dict | None, question: str) -> str:
    if item is None:
        return f"Question:{question}"
    return "Background:\n" + "\n".join(item["passages"]) + f"\n\nQuestion:{question}"


def _dtype(name: str):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _done(out: Path, system: str) -> set:
    if not out.exists():
        return set()
    return {(r["qid"], r["arm"]) for r in read_jsonl(out) if r["system"] == system}


def _write(out: Path, rows: list[dict]):
    with out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_pisco(items, args):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(PISCO, trust_remote_code=True, attn_implementation=args.attn)
    if args.dtype != "bf16":  # PISCO casts itself to bf16; convert everything consistently
        model = model.to(_dtype(args.dtype))
    model = model.to("cuda").eval()
    # generate_from_text needs the same number of documents for every question in a batch.
    groups = defaultdict(list)
    for it in items:
        groups[len(it["passages"])].append(it)
    for k, group in sorted(groups.items()):
        for i in range(0, len(group), args.batch_size):
            batch = group[i:i + args.batch_size]
            t = time.time()
            with torch.no_grad():
                preds = model.generate_from_text(questions=[b["question"] for b in batch],
                                                 documents=[b["passages"] for b in batch],
                                                 max_new_tokens=args.max_new_tokens)
            dt = (time.time() - t) / len(batch)
            _write(args.out, [{"system": "pisco", "qid": b["qid"], "arm": b["arm"], "prediction": p.strip(),
                               "n_passages": k, "sec_per_item": round(dt, 3)} for b, p in zip(batch, preds)])


def run_mistral(items, args, closedbook: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system = "closedbook" if closedbook else "mistral"
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=_dtype(args.dtype), attn_implementation=args.attn,
        **({"load_in_8bit": True, "device_map": "auto"} if args.quantization == "8bit" else {}))
    if args.quantization != "8bit":
        model = model.to("cuda")
    model.eval()
    # Closed-book answers depend only on the question text: generate once per distinct question.
    prompts = {}
    for it in items:
        prompts.setdefault(chat_prompt(tok, mistral_user(None if closedbook else it, it["question"])), []).append(it)
    order = sorted(prompts, key=len)
    for i in range(0, len(order), args.batch_size):
        batch = order[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        dt = (time.time() - t) / len(batch)
        rows = []
        for prompt, text in zip(batch, texts):
            for it in prompts[prompt]:
                rows.append({"system": system, "qid": it["qid"], "arm": it["arm"], "prediction": text.strip(),
                             "sec_per_item": round(dt, 3)})
        _write(args.out, rows)


def worker(args):
    import torch

    items = load_items(args.data, args.arms, args.limit, args.include_invalid)
    done = _done(args.out, args.worker)
    todo = [it for it in items if (it["qid"], it["arm"]) not in done]
    print(f"[{args.worker}] {len(items)} items, {len(done)} done, {len(todo)} to generate "
          f"| {torch.cuda.get_device_name(0)} | dtype={args.dtype} attn={args.attn} quant={args.quantization}",
          flush=True)
    if not todo:
        return
    t = time.time()
    if args.worker == "pisco":
        run_pisco(todo, args)
    else:
        run_mistral(todo, args, closedbook=args.worker == "closedbook")
    print(f"[{args.worker}] finished {len(todo)} items in {time.time() - t:.0f}s "
          f"| peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems", default="pisco,mistral,closedbook")
    ap.add_argument("--arms", default=None, help="Comma-separated subset of arms")
    ap.add_argument("--limit", type=int, default=None, help="Number of questions (all their arms)")
    ap.add_argument("--include-invalid", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--attn", default="sdpa", help="FlashAttention-2 is unavailable on V100")
    ap.add_argument("--quantization", choices=["none", "8bit"], default="none",
                    help="Fallback for small GPUs; apply to every system or none")
    ap.add_argument("--worker", choices=SYSTEMS, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.arms = args.arms.split(",") if args.arms else None
    if args.quantization == "8bit" and "pisco" in args.systems:
        raise SystemExit("8-bit is wired for the Mistral readers only; PISCO 8-bit must be added before use")
    if args.worker:
        return worker(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for system in args.systems.split(","):
        if system not in SYSTEMS:
            raise SystemExit(f"unknown system {system!r}")
        cmd = [sys.executable, "-m", "lexfaith.readers", *sys.argv[1:], "--worker", system]
        subprocess.run(cmd, check=True)  # fresh process per model: GPU memory fully released


if __name__ == "__main__":
    main()
