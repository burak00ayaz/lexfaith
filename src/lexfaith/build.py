"""Build the entity-substituted SQuAD benchmark.

    uv run --locked lexfaith-build --n-paragraphs 100 --out data/p100 \
        --llm-provider openrouter --llm-model anthropic/claude-sonnet-5.5

The API key is read from the environment variable named by --llm-api-key-env, after loading
--env-file; it is never passed on the command line.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from lexfaith.annotate import build_inventory, load_nlp, span_pos
from lexfaith.chunk import passages_for
from lexfaith.features import answer_type, item_features, load_tokenizer
from lexfaith.identifiers import add_near_misses, edit_distance
from lexfaith.mapping import REAL_CHECK_TYPES, Mapper, mapping_flags
from lexfaith.numeric import numeric_mapping
from lexfaith.prompts import NAMED_TYPES, PROMPT_VERSION
from lexfaith.substitute import ARMS, POLICY, answer_mention_ids, make_variant
from lexfaith.wiki import WikiChecker


PROVIDERS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   # ask OpenRouter to report per-call cost in `usage`
                   "extra_body": {"usage": {"include": True}}},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "extra_body": {}},
    "local": {"base_url": "http://localhost:8765/v1", "key_env": None, "extra_body": {}},
}


def load_env_file(path: Path) -> list[str]:
    """Set KEY=VALUE lines from a dotenv-style file without overriding the existing environment."""
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


def load_paragraphs(split: str) -> list[dict]:
    # Fetch only the requested split file.
    ds = load_dataset("rajpurkar/squad", data_files={split: f"plain_text/{split}-00000-of-00001.parquet"},
                      split=split)
    paras: dict[str, dict] = {}
    for ex in ds:
        pid = hashlib.sha1(ex["context"].encode()).hexdigest()[:12]
        p = paras.setdefault(pid, {"para_id": pid, "title": ex["title"], "context": ex["context"], "qas": []})
        answers = [{"text": t, "answer_start": s}
                   for t, s in zip(ex["answers"]["text"], ex["answers"]["answer_start"])]
        # Deduplicate annotator answers (SQuAD dev has up to 3, often identical).
        uniq = list({(a["text"], a["answer_start"]): a for a in answers}.values())
        p["qas"].append({"qid": ex["id"], "question": ex["question"], "answers": uniq})
    return list(paras.values())


def sample_paragraphs(paras: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over article titles so the sample spans topics."""
    rng = random.Random(seed)
    by_title = defaultdict(list)
    for p in paras:
        by_title[p["title"]].append(p)
    titles = sorted(by_title)
    rng.shuffle(titles)
    for t in titles:
        rng.shuffle(by_title[t])
    out = []
    while len(out) < n and any(by_title.values()):
        for t in titles:
            if by_title[t] and len(out) < n:
                out.append(by_title[t].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-paragraphs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2",
                    help="Reader tokenizer for subword-count features (empty to skip)")
    ap.add_argument("--doc-max-tokens", type=int, default=128,
                    help="Compressor document budget in reader tokens (PISCO: 128)")
    ap.add_argument("--llm-provider", choices=sorted(PROVIDERS), default="openrouter")
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--llm-base-url", default=None, help="Overrides the provider preset")
    ap.add_argument("--llm-api-key-env", default=None, help="Env var holding the key (provider default)")
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    ap.add_argument("--llm-max-tokens", type=int, default=4096)
    ap.add_argument("--llm-extra-body", default="{}", help="JSON merged into the chat request body")
    ap.add_argument("--llm-workers", type=int, default=8, help="Paragraphs mapped concurrently")
    ap.add_argument("--no-wiki-check", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    load_env_file(args.env_file)
    preset = PROVIDERS[args.llm_provider]
    key_env = args.llm_api_key_env or preset["key_env"]
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise SystemExit(f"{key_env} is not set (looked in the environment and {args.env_file})")
    extra_body = {**preset["extra_body"], **json.loads(args.llm_extra_body)}

    paras = sample_paragraphs(load_paragraphs(args.split), args.n_paragraphs, args.seed)
    print(f"[build] {len(paras)} paragraphs, {sum(len(p['qas']) for p in paras)} questions")

    nlp = load_nlp(args.spacy_model)
    ctx_docs = list(nlp.pipe([p["context"] for p in paras], batch_size=16))
    for p, doc in zip(paras, ctx_docs):
        q_docs = list(nlp.pipe([qa["question"] for qa in p["qas"]]))
        p["inventory"] = build_inventory(doc, q_docs)
        p["_doc"] = doc

    wiki = WikiChecker(args.out / "wiki_cache.json", enabled=not args.no_wiki_check)
    mapper = Mapper(args.llm_model, args.llm_base_url or preset["base_url"], api_key,
                    args.out / "llm_cache.jsonl", max_tokens=args.llm_max_tokens, extra_body=extra_body,
                    wiki=wiki if not args.no_wiki_check else None)

    def map_one(p):
        named = [e for e in p["inventory"] if e["type"] in NAMED_TYPES]
        p["map"] = mapper.map_paragraph(p["context"], named)
        num_map, p["map"]["year_offset"] = numeric_mapping(p["inventory"], p["para_id"], args.seed)
        if p["map"]["mapping"] is not None:
            p["map"]["mapping"].update(num_map)
            add_near_misses(p["map"]["mapping"], random.Random(f"{args.seed}:nm:{p['para_id']}"))

    # Paragraphs are independent; results are attached in place, so output order stays fixed.
    with ThreadPoolExecutor(max_workers=args.llm_workers) as pool:
        list(tqdm(pool.map(map_one, paras), total=len(paras), desc="mapping"))
    print(f"[build] LLM usage: {json.dumps(mapper.stats)}")

    llm_entries = [m for p in paras if p["map"]["mapping"] for m in p["map"]["mapping"].values()
                   if not m["keep"] and m["source"] == "llm"]
    on_wiki = wiki.real([m["fictional"] for m in llm_entries])
    real_ok = wiki.exists([m["real_swap"] for m in llm_entries])

    tok = load_tokenizer(args.tokenizer)
    items, para_rows = [], []
    for p in paras:
        mapping = p["map"]["mapping"]
        ok = mapping is not None
        flags = mapping_flags(mapping) if ok else {}
        if ok:
            flags["fictional_on_wikipedia"] = sorted(
                m["fictional"] for m in mapping.values() if m["fictional"] in on_wiki and not m["keep"])
            flags["real_swap_not_on_wikipedia"] = sorted(
                m["real_swap"] for m in mapping.values()
                if not m["keep"] and m["type"] in REAL_CHECK_TYPES and m["real_swap"] not in real_ok)
        para_rows.append({
            "para_id": p["para_id"], "title": p["title"], "context": p["context"],
            "inventory": p["inventory"], "mapping": mapping, "year_offset": p["map"]["year_offset"],
            "llm_attempts": p["map"]["attempts"], "llm_problems": p["map"]["problems"],
            "unresolved": p["map"].get("unresolved", []), "flags": flags,
            "llm_model": args.llm_model, "prompt_version": PROMPT_VERSION, "spacy_model": args.spacy_model,
        })
        all_ids = {i for i, m in (mapping or {}).items() if not m["keep"]}
        for qa in p["qas"]:
            a0 = qa["answers"][0]
            pos_seq, head_pos = span_pos(p["_doc"], a0["answer_start"], a0["answer_start"] + len(a0["text"]))
            atype = answer_type(qa["answers"], p["inventory"], mapping)
            ans_ids = answer_mention_ids(qa["answers"], p["inventory"], mapping) if ok else set()
            shared = {
                "qid": qa["qid"], "para_id": p["para_id"], "title": p["title"],
                **atype, "answer_pos": " ".join(pos_seq), "answer_head_pos": head_pos,
                "orig_question": qa["question"], "orig_answers": [a["text"] for a in qa["answers"]],
                "n_substitutable_mentions": len(all_ids), "answer_mention_ids": sorted(ans_ids),
                "n_unresolved_mentions": len(p["map"].get("unresolved", [])),
            }
            for arm in ARMS:
                if arm == "original":
                    v = {"context": p["context"], "question": qa["question"],
                         "answers": [dict(a, text_by_string=a["text"]) for a in qa["answers"]],
                         "table": {}, "n_context_edits": 0, "n_question_edits": 0,
                         "frac_context_chars_changed": 0.0, "residual_originals": []}
                elif not ok:
                    v = None
                else:
                    policy = POLICY[arm.split("_")[0]]
                    ids = all_ids if arm.endswith("_all") else ans_ids
                    v = make_variant(p["context"], qa["question"], qa["answers"], mapping, ids, policy)
                checks = _checks(arm, v, qa, ans_ids, ok, mapping, on_wiki)
                row = {**shared, "arm": arm, "valid": all(checks.values()), "checks": checks}
                if v is not None:
                    na = v["answers"][0]
                    row.update({
                        "context": v["context"], "question": v["question"],
                        "answers": [a["text"] for a in v["answers"]],
                        "answer_starts": [a["answer_start"] for a in v["answers"]],
                        "substitutions": v["table"], "n_context_edits": v["n_context_edits"],
                        "n_question_edits": v["n_question_edits"],
                        "frac_context_chars_changed": round(v["frac_context_chars_changed"], 4),
                        "answer_changed": [a["text"] for a in v["answers"]] != shared["orig_answers"],
                        **item_features(v["context"], v["question"], na["text"], na["answer_start"], tok),
                    })
                    if tok is not None:
                        row.update(passages_for(v["context"], na["answer_start"], len(na["text"]), tok,
                                                args.doc_max_tokens))
                items.append(row)

    _write_jsonl(args.out / "items.jsonl", items)
    _write_jsonl(args.out / "paragraphs.jsonl", para_rows)
    summary = _summary(items, para_rows, args)
    summary["llm_provider"] = args.llm_provider
    summary["llm_usage"] = mapper.stats
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def _checks(arm, v, qa, ans_ids, ok, mapping, on_wiki) -> dict:
    if arm == "original":
        return {"mapping_ok": True}
    c = {"mapping_ok": ok}
    if v is None:
        return c
    c["answer_consistent"] = all(a["text"] == a["text_by_string"] and a["text"] in v["context"]
                                 for a in v["answers"])
    c["no_residual_original"] = not v["residual_originals"]
    if arm.endswith("_answer"):
        c["answer_is_substitutable"] = bool(ans_ids)
        # Every gold answer must change; otherwise an unchanged annotator span still answers it.
        c["answer_changed"] = all(a["text"] != o["text"] for a, o in zip(v["answers"], qa["answers"]))
    # An answer overlapping a mention the LLM could not map would keep a real name inside it.
    c["answer_mentions_resolved"] = not any(
        m.get("unresolved") and any(m["surface"] in a["text"] for a in qa["answers"]) for m in mapping.values())
    if arm == "nearmiss_answer":
        # one edit per substituted mention ("Alberta and British Columbia" has two)
        c["one_edit_per_mention"] = all(edit_distance(k, s) == 1 for k, s in v["table"].items())
    if arm.startswith("fictional"):
        # An invented answer that is a documented real entity defeats the fictional arm.
        c["answer_not_on_wikipedia"] = not any(mapping[i]["fictional"] in on_wiki for i in ans_ids)
    return c


def _summary(items, para_rows, args) -> dict:
    by_arm = defaultdict(Counter)
    fail = defaultdict(Counter)
    for r in items:
        by_arm[r["arm"]]["total"] += 1
        by_arm[r["arm"]]["valid"] += r["valid"]
        for k, v in r["checks"].items():
            if not v:
                fail[r["arm"]][k] += 1
    orig = [r for r in items if r["arm"] == "original"]
    return {
        "llm_model": args.llm_model, "prompt_version": PROMPT_VERSION, "spacy_model": args.spacy_model,
        "split": args.split, "seed": args.seed,
        "n_paragraphs": len(para_rows),
        "n_paragraphs_mapping_failed": sum(p["mapping"] is None for p in para_rows),
        "n_named_mentions": sum(e["type"] in NAMED_TYPES for p in para_rows for e in p["inventory"]),
        "n_unresolved_mentions": sum(len(p["unresolved"]) for p in para_rows),
        "llm_attempts": Counter(p["llm_attempts"] for p in para_rows),
        "n_questions": len(orig),
        "n_mentions": sum(len(p["inventory"]) for p in para_rows),
        "n_mentions_kept": sum(m["keep"] for p in para_rows if p["mapping"] for m in p["mapping"].values()),
        "mention_types": Counter(e["type"] for p in para_rows for e in p["inventory"]),
        "answer_type_group": Counter(r["answer_type_group"] for r in orig),
        "arms": {a: dict(c) for a, c in by_arm.items()},
        "answer_straddles_chunk": {a: sum(r.get("answer_passage", 0) is None for r in items if r["arm"] == a)
                                   for a in by_arm},
        "answer_beyond_doc_budget": {a: sum(r.get("answer_token_start", 0) >= args.doc_max_tokens
                                            for r in items if r["arm"] == a) for a in by_arm},
        "check_failures": {a: dict(c) for a, c in fail.items()},
        "soft_flags": {k: sum(len(p["flags"].get(k, [])) for p in para_rows) for k in
                       ("length_ratio_out_of_range", "alias_inconsistent",
                        "fictional_on_wikipedia", "real_swap_not_on_wikipedia")},
    }


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
