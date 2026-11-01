"""Score predictions against a built benchmark.

    uv run --locked lexfaith-eval --data data/pilot --predictions preds.jsonl \
        --compressed pisco --reference mistral
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from lexfaith.io import read_jsonl
from lexfaith.metrics import accuracy_table, diff_in_diff, memorization_ratio, score_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--compressed", default=None, help="system name of the compressed reader")
    ap.add_argument("--reference", default=None, help="system name of the uncompressed reader")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--include-invalid", action="store_true")
    args = ap.parse_args()

    items = {(r["qid"], r["arm"]): r for r in read_jsonl(args.data / "items.jsonl")}
    scored = []
    for p in map(json.loads, args.predictions.open()):
        item = items.get((p["qid"], p["arm"]))
        if item is None or (not item["valid"] and not args.include_invalid):
            continue
        scored.append({**{k: item.get(k) for k in (
            "qid", "para_id", "arm", "answer_type_group", "answer_ner_type", "answer_is_entity",
            "answer_changed", "answer_n_subwords", "answer_min_zipf", "source", "id_kind")},
            "system": p["system"], "prediction": p["prediction"], **score_item(item, p["prediction"])})

    report = {
        "accuracy": accuracy_table(scored),
        "accuracy_by_type": accuracy_table(scored, by="answer_type_group"),
        **({"accuracy_by_source": accuracy_table(scored, by="source")} if any(r["source"] for r in scored) else {}),
        "memorization_ratio": memorization_ratio(scored),
        "error_categories": _categories(scored),
    }
    if args.compressed and args.reference:
        arms = sorted({r["arm"] for r in scored} - {"original"})
        report["diff_in_diff"] = [diff_in_diff(scored, args.compressed, args.reference, a) for a in arms]

    for name, rows in report.items():
        print(f"\n== {name}")
        for row in rows:
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    out = args.out or args.predictions.with_suffix(".scores")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    with (out / "scored.jsonl").open("w") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _categories(scored):
    agg = defaultdict(Counter)
    for r in scored:
        agg[(r["system"], r["arm"])][r["category"]] += 1
    rows = []
    for (s, a), c in sorted(agg.items()):
        n = sum(c.values())
        rows.append({"system": s, "arm": a, "n": n, **{k: c[k] / n for k in
                     ("correct", "distractor", "parametric", "near_miss", "context", "other")}})
    return rows


if __name__ == "__main__":
    main()
