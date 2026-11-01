"""Write a Markdown audit sheet: per paragraph, the mapping table and a sample of questions per arm.

    uv run --locked lexfaith-audit --data data/pilot --out audit.md --n-paragraphs 10

For an identifier build (lexfaith-build-ids, no paragraphs.jsonl) the sheet instead lists, per
(source, kind), sampled questions with their substitutes in every arm.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from lexfaith.io import read_jsonl


def _bold_subs(text: str, subs: dict[str, str]) -> str:
    new = sorted({v for k, v in subs.items() if k != v}, key=len, reverse=True)
    if not new:
        return text
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(s) for s in new) + r")(?![\w-])")
    return pat.sub(r"**\1**", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-paragraphs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.data / "paragraphs.jsonl").exists() and not (args.data / "paragraphs.jsonl.gz").exists():
        args.out.write_text(_audit_ids(read_jsonl(args.data / "items.jsonl"), args))
        print(f"wrote {args.out}")
        return
    paras = read_jsonl(args.data / "paragraphs.jsonl")
    items = defaultdict(list)
    for r in read_jsonl(args.data / "items.jsonl"):
        items[r["para_id"]].append(r)
    random.Random(args.seed).shuffle(paras)

    out = [f"# Audit sheet: {args.data}\n"]
    for p in paras[: args.n_paragraphs]:
        out.append(f"## {p['title']} (`{p['para_id']}`)\n")
        out.append(f"LLM attempts: {p['llm_attempts']}; unresolved: {p['unresolved'] or 'none'}; "
                   f"year offset: {p['year_offset']}\n")
        if p["mapping"] is None:
            out.append(f"**Mapping failed:** {'; '.join(p['llm_problems'])[:300]}\n")
            continue
        out.append("| id | type | fine type | original | fictional | real swap | near-miss | keep | source |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for e in p["inventory"]:
            m = p["mapping"][e["id"]]
            out.append(f"| {e['id']} | {e['type']} | {m.get('fine_type') or ''} | {e['surface']} | "
                       f"{m['fictional']} | {m['real_swap']} | {m.get('near_miss', '')} | {'✓' if m['keep'] else ''}"
                       f"{' (unresolved)' if m.get('unresolved') else ''} | {m.get('source', '')} |")
        flags = {k: v for k, v in p["flags"].items() if v}
        if flags:
            out.append(f"\nSoft flags: `{json.dumps(flags, ensure_ascii=False)}`")
        fic = next((r for r in items[p["para_id"]] if r["arm"] == "fictional_all" and "context" in r), None)
        if fic:
            out.append("\n**fictional_all passage** (substitutes in bold):\n")
            out.append("> " + _bold_subs(fic["context"], fic["substitutions"]) + "\n")
        for qid in sorted({r["qid"] for r in items[p["para_id"]]})[:3]:
            rows = {r["arm"]: r for r in items[p["para_id"]] if r["qid"] == qid}
            o = rows["original"]
            out.append(f"- **Q** {o['question']} — **A** {o['answers'][0]} "
                       f"(type {o['answer_ner_type']}, POS `{o['answer_pos']}`)")
            for arm in ("fictional_all", "realswap_all", "fictional_answer", "realswap_answer", "nearmiss_answer"):
                r = rows.get(arm)
                if not r or "question" not in r:
                    continue
                bad = [k for k, v in r["checks"].items() if not v]
                status = "valid" if r["valid"] else f"invalid: {', '.join(bad)}"
                out.append(f"  - `{arm}`: {r['question']} — **A** {r['answers'][0]} *({status})*")
        out.append("")
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")


def _audit_ids(rows: list[dict], args) -> str:
    by_q = defaultdict(dict)
    for r in rows:
        by_q[r["qid"]][r["arm"]] = r
    groups = defaultdict(list)
    for qid, arms in by_q.items():
        o = arms["original"]
        groups[(o["source"], o["id_kind"])].append(qid)
    rng = random.Random(args.seed)
    out = [f"# Identifier audit sheet: {args.data}\n",
           "Per question: substitutes per arm (✗ = invalid, with the failed checks); near-miss marked "
           "*(real)* when the edited string is itself a registered identifier.\n"]
    for (source, kind), qids in sorted(groups.items()):
        qids = sorted(qids)
        rng.shuffle(qids)
        out.append(f"## {source} / {kind} ({len(qids)} questions)\n")
        out.append("| question | original | real swap | fictional | near-miss |")
        out.append("|---|---|---|---|---|")
        for qid in qids[: args.n_paragraphs]:
            a = by_q[qid]
            o = a["original"]

            def cell(arm):
                r = a.get(arm, {})
                txt = (r.get("answers") or ["—"])[0]
                bad = [k for k, v in r.get("checks", {}).items() if not v]
                return f"{txt}{' ✗ ' + ', '.join(bad) if bad else ''}"
            nm = cell("nearmiss_answer") + (" *(real)*" if o.get("near_miss_is_real") else "")
            q = o["question"].replace("|", "/")
            out.append(f"| {q} | **{o['answers'][0]}** | {cell('realswap_answer')} | "
                       f"{cell('fictional_answer')} | {nm} |")
        ex = by_q[qids[0]]
        d = ex.get("distractor_answer", {})
        out.append(f"\nExample passage (distractor arm, `{qids[0]}`):\n")
        out.append("> " + _bold_subs(d.get("context", ex["original"]["context"]),
                                    {"_": d.get("distractor", "")}) + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    main()
