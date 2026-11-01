"""Scoring of reader predictions on the benchmark.

Predictions: JSONL rows {"system": str, "qid": str, "arm": str, "prediction": str}.

Per item: EM / token F1 (SQuAD normalization), Match (gold contained in prediction; the metric used
by PISCO/BERGEN), lexical similarity to gold, whether the original (pre-substitution) answer was
produced, and an error category:
  correct     Match
  distractor  prediction contains the inserted near-miss sibling (distractor arm only)
  parametric  prediction contains the original answer although the context says otherwise
  near_miss   high character similarity to gold but no Match: the string was read but corrupted
  context     prediction is another span of the given context (wrong span, intact copy)
  other       anything else

Aggregates: accuracy per system x arm (x feature), memorization ratio (Longpre et al., 2021), and
the difference-in-differences of the arm-induced drop between a compressed system and its
uncompressed reference, with a paragraph-clustered bootstrap CI.
"""

import re
import string
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz import fuzz

NEAR_MISS = 0.75


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def contains(pred: str, gold: str) -> bool:
    g = normalize(gold)
    return bool(g) and re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", normalize(pred)) is not None


def score_item(item: dict, pred: str) -> dict:
    golds, origs = item["answers"], item["orig_answers"]
    match = any(contains(pred, g) for g in golds)
    sim = max((fuzz.partial_ratio(normalize(g), normalize(pred)) / 100 for g in golds if normalize(g)),
              default=0.0)
    gold_norm = {normalize(g) for g in golds}
    produced_original = item.get("answer_changed", False) and any(
        contains(pred, o) for o in origs if normalize(o) not in gold_norm)
    produced_distractor = bool(item.get("distractor")) and contains(pred, item["distractor"])
    pn = normalize(pred)
    if match:
        cat = "correct"
    elif produced_distractor:
        cat = "distractor"
    elif produced_original:
        cat = "parametric"
    elif sim >= NEAR_MISS:
        cat = "near_miss"
    elif pn and pn in normalize(item["context"]):
        cat = "context"
    else:
        cat = "other"
    return {
        "em": max(float(normalize(pred) == normalize(g)) for g in golds),
        "f1": max(f1(pred, g) for g in golds),
        "match": float(match),
        "char_sim": sim,
        "produced_original": float(produced_original),
        "produced_distractor": float(produced_distractor),
        "category": cat,
    }


def cluster_bootstrap(values_by_cluster: dict[str, list[float]], n: int = 2000, seed: int = 0):
    """Mean and 95% CI, resampling paragraphs (questions about one paragraph are not independent)."""
    keys = list(values_by_cluster)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sums = np.array([sum(values_by_cluster[k]) for k in keys])
    cnts = np.array([len(values_by_cluster[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return sums.sum() / cnts.sum(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def accuracy_table(scored: list[dict], metric: str = "match", by: str | None = None) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for r in scored:
        key = (r["system"], r["arm"], r.get(by) if by else None)
        groups[key][r["para_id"]].append(r[metric])
    rows = []
    for (system, arm, g), clusters in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        mean, lo, hi = cluster_bootstrap(clusters)
        rows.append({"system": system, "arm": arm, **({by: g} if by else {}),
                     "n": sum(map(len, clusters.values())), metric: mean, "ci_lo": lo, "ci_hi": hi})
    return rows


def memorization_ratio(scored: list[dict]) -> list[dict]:
    """MR = p_o / (p_o + p_s) over items whose answer changed (Longpre et al., 2021)."""
    agg = defaultdict(Counter)
    for r in scored:
        if r.get("answer_changed"):
            agg[(r["system"], r["arm"])]["po"] += r["produced_original"]
            agg[(r["system"], r["arm"])]["ps"] += r["match"]
    return [{"system": s, "arm": a, "p_o": c["po"], "p_s": c["ps"],
             "MR": c["po"] / (c["po"] + c["ps"]) if c["po"] + c["ps"] else float("nan")}
            for (s, a), c in sorted(agg.items())]


def diff_in_diff(scored: list[dict], compressed: str, reference: str, arm: str,
                 metric: str = "match") -> dict:
    """[comp(orig) - comp(arm)] - [ref(orig) - ref(arm)]: positive = the arm hurts the compressed
    reader more than the uncompressed one. Only questions scored for both systems in both arms count."""
    val = {(r["system"], r["arm"], r["qid"]): r for r in scored}
    clusters = defaultdict(list)
    for (s, a, q), r in val.items():
        if s != compressed or a != arm:
            continue
        keys = [(compressed, "original", q), (reference, "original", q), (reference, arm, q)]
        if not all(k in val for k in keys):
            continue
        comp_drop = val[keys[0]][metric] - r[metric]
        ref_drop = val[keys[1]][metric] - val[keys[2]][metric]
        clusters[r["para_id"]].append(comp_drop - ref_drop)
    mean, lo, hi = cluster_bootstrap(clusters)
    return {"compressed": compressed, "reference": reference, "arm": arm, "metric": metric,
            "n": sum(map(len, clusters.values())), "did": mean, "ci_lo": lo, "ci_hi": hi}
