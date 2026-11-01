"""Build the identifier-preservation benchmark (domain corpora, edit-distance ladder).

    uv run --locked lexfaith-build-ids --corpus advisories --n 400 --out data/ids-advisories
    uv run --locked lexfaith-build-ids --corpus bioasq,openfda --n 400 --out data/ids-medical

Arms per question (the answer identifier is replaced everywhere in passage and question):
  original           unchanged
  realswap_answer    another real identifier of the same kind
  fictional_answer   well-formed, unregistered identifier (distance >= 2)
  nearmiss_answer    one character changed
  distractor_answer  original kept; "<answer> should not be confused with <near-miss>." inserted

Arm names match the answer-scope arms of the SQuAD build so results can be pooled by arm.
Every passage is first cropped to the sentence-aligned window of <= --doc-max-tokens reader tokens
that holds the answer, so each item is exactly one compressor document in the original arm.
No LLM is involved: substitutes come from registries (RxNorm, HGNC, PyPI, CVE, CWE) and rules.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from lexfaith.chunk import chunk, passages_for
from lexfaith.corpora import load_advisories, load_bioasq, load_openfda, sample_balanced
from lexfaith.features import item_features, load_tokenizer, n_subwords
from lexfaith.identifiers import (edit_distance, fictional_candidates, insert_distractor, match_case,
                                  near_miss_candidates, occurrences, rewrite_id, version_swaps)
from lexfaith.registries import Registries
from lexfaith.substitute import map_offset
from lexfaith.wiki import WikiChecker

ARMS = ["original", "realswap_answer", "fictional_answer", "nearmiss_answer", "distractor_answer"]
POLICY = {"realswap_answer": "real_swap", "fictional_answer": "fictional", "nearmiss_answer": "near_miss"}
IGNORE_CASE = {"drug", "package"}  # drug labels write XARELTO and Xarelto; package names are case-free


def crop(seed: dict, tok, max_tokens: int) -> dict | None:
    a = seed["answers"][0]
    for s, e in chunk(seed["context"], tok, max_tokens):
        if s <= a["answer_start"] and a["answer_start"] + len(a["text"]) <= e:
            return {**seed, "context": seed["context"][s:e],
                    "answers": [{"text": a["text"], "answer_start": a["answer_start"] - s}]}
    return None


class Chooser:
    """Picks the three substitutes of one item; registry lookups decide real vs. invented."""

    def __init__(self, reg: Registries, wiki: WikiChecker, seeds: list[dict]):
        self.reg, self.wiki = reg, wiki
        self.pool = defaultdict(list)  # (source, kind) -> answers of other items (real by construction)
        self.by_class = defaultdict(list)
        ids = [int(c[4:]) for c in reg.cwes()]
        # free ids below the largest assigned one (CWE-2000, a view). Ids up to ~1450 are all taken,
        # so fictional CWEs are effectively CWE-1451..1999: well-formed, plausible, unassigned.
        self.free_cwes = sorted(set(range(1, max(ids))) - set(ids)) if any(s["kind"] == "cwe" for s in seeds) else []
        for s in seeds:
            self.pool[(s["source"], s["kind"])].append(s["answers"][0]["text"])
            for c in s["meta"].get("pharm_class_epc", []):
                self.by_class[c].append(s["answers"][0]["text"])

    def exists(self, kind: str, s: str, releases=None) -> bool:
        if kind == "drug":
            return self.reg.drug_exists(s) or bool(self.wiki.real([s]))
        if kind == "gene":
            return self.reg.gene_exists(s)
        if kind == "version":
            return s in set(releases or [])
        if kind == "cve":
            return self.reg.cve_exists(s)
        if kind == "cwe":
            return self.reg.cwe_exists(s)
        if kind == "package":
            return self.reg.package_exists(s)
        raise ValueError(kind)

    def choose(self, seed: dict, rng_seed: int) -> dict:
        rng = random.Random(f"{rng_seed}:{seed['qid']}")
        kind, s = seed["kind"], seed["answers"][0]["text"]
        text = seed["context"] + "\n" + seed["question"]

        def free(c):  # a substitute that already occurs in the item would make it ambiguous
            return c and c.lower() != s.lower() and not occurrences(text, c, ignore_case=True)

        releases = self.reg.pypi_releases(seed["meta"]["package"]) if kind == "version" else None
        # real swap
        if kind == "version":
            real_cands = version_swaps(s, releases or [], rng)
        elif kind == "drug" and seed["source"] == "openfda":
            same = [b for c in seed["meta"].get("pharm_class_epc", []) for b in self.by_class[c]]
            rest = list(self.pool[(seed["source"], kind)])
            rng.shuffle(same), rng.shuffle(rest)
            real_cands = same + rest
        elif kind == "drug":
            same = [m for c in self.reg.atc_class(s) for m in self.reg.atc_members(c)]
            rest = list(self.pool[(seed["source"], kind)])
            rng.shuffle(same), rng.shuffle(rest)
            real_cands = same + rest
        else:
            real_cands = list(self.pool[(seed["source"], kind)])
            rng.shuffle(real_cands)
        real = next((c for c in real_cands if free(c) and edit_distance(c.lower(), s.lower()) >= 2), None)
        if real and kind != "version":
            real = match_case(real, s) if kind in IGNORE_CASE else real
        # fictional
        fict = next((c for c in fictional_candidates(s, kind, rng, releases=releases,
                                                         unassigned=self.free_cwes)
                     if free(c) and not self.exists(kind, c, releases)), None)
        # near miss (allowed to be real; recorded)
        nm = next((c for c in near_miss_candidates(s, kind, rng) if free(c)), None)
        return {
            "real_swap": real, "fictional": fict, "near_miss": nm,
            "near_miss_is_real": self.exists(kind, nm, releases) if nm else None,
            # informational: swaps are drawn from real sources, but e.g. new FDA brands are not in RxNorm
            "real_swap_in_registry": self.exists(kind, real, releases) if real else None,
            "n_releases": len(releases) if releases is not None else None,
        }


def make_id_variant(seed: dict, new: str) -> dict:
    kind, a = seed["kind"], seed["answers"][0]
    old, ic = a["text"], seed["kind"] in IGNORE_CASE
    ctx, edits = rewrite_id(seed["context"], old, new, ic)
    q, q_edits = rewrite_id(seed["question"], old, new, ic)
    ns = map_offset(a["answer_start"], edits, "start")
    ne = map_offset(a["answer_start"] + len(old), edits, "end")
    masked = _mask(ctx, edits) + "\n" + _mask(q, q_edits)
    return {
        "context": ctx, "question": q, "answers": [{"text": ctx[ns:ne], "answer_start": ns}],
        "expected": match_case(new, old) if ic else new,
        "n_context_edits": len(edits), "n_question_edits": len(q_edits),
        # plain substring, looser than the replacement pattern: "NF1" left inside "NF1-related" counts
        "residual_original": re.search(re.escape(old), masked, re.I if ic else 0) is not None,
        "kind": kind,
    }


def _mask(text, edits):
    out, pos = [], 0
    for _, _, ns, ne in edits:
        out += [text[pos:ns], " " * (ne - ns)]
        pos = ne
    return "".join(out) + text[pos:]


def build_rows(seed: dict, subs: dict, tok, max_tokens: int) -> list[dict]:
    a0 = seed["answers"][0]
    shared = {
        "qid": seed["qid"], "para_id": seed["para_id"], "title": seed["title"], "source": seed["source"],
        "answer_ner_type": seed["kind"], "answer_type_group": seed["kind"].upper(),
        "answer_fine_type": ", ".join(seed["meta"].get("pharm_class_epc", [])[:1]) or None,
        "answer_is_entity": True, "answer_entity_exact": True, "id_kind": seed["kind"],
        "orig_question": seed["question"], "orig_answers": [a0["text"]],
        "answer_n_subwords_original": n_subwords(tok, a0["text"]),
        "near_miss_is_real": subs["near_miss_is_real"], "real_swap_in_registry": subs["real_swap_in_registry"],
        "meta": seed["meta"],
    }
    rows = []
    for arm in ARMS:
        checks, extra = {}, {}
        if arm == "original":
            v = {"context": seed["context"], "question": seed["question"], "answers": [a0],
                 "n_context_edits": 0, "n_question_edits": 0}
            sub = None
        elif arm == "distractor_answer":
            sub = subs["near_miss"]
            checks["substitute_found"] = sub is not None
            if sub is None:
                v = None
            else:
                sib = match_case(sub, a0["text"]) if seed["kind"] in IGNORE_CASE else sub
                ctx, start = insert_distractor(seed["context"], a0["text"], a0["answer_start"], sib, seed["kind"])
                na = {"text": a0["text"], "answer_start": start}
                v = {"context": ctx, "question": seed["question"], "answers": [na],
                     "n_context_edits": 1, "n_question_edits": 0}
                checks["answer_consistent"] = ctx[start:start + len(a0["text"])] == a0["text"]
                checks["distractor_present"] = bool(occurrences(ctx, sib, ignore_case=seed["kind"] in IGNORE_CASE))
                extra["distractor"] = sib
        else:
            sub = subs[POLICY[arm]]
            checks["substitute_found"] = sub is not None
            if sub is None:
                v = None
            else:
                v = make_id_variant(seed, sub)
                got, want = v["answers"][0]["text"], v["expected"]
                checks["answer_consistent"] = got == want and bool(occurrences(v["context"], want))
                checks["no_residual_original"] = not v["residual_original"]
                checks["answer_changed"] = got.lower() != a0["text"].lower()
                if arm == "nearmiss_answer":
                    checks["edit_distance_1"] = edit_distance(sub, a0["text"]) == 1
                if arm == "fictional_answer":
                    checks["edit_distance_ge_2"] = edit_distance(sub.lower(), a0["text"].lower()) >= 2
        row = {**shared, "arm": arm, "valid": all(checks.values()) if v is not None else False,
               "checks": checks or {"ok": True}, **extra}
        if v is not None:
            na = v["answers"][0]
            row.update({
                "context": v["context"], "question": v["question"], "answers": [na["text"]],
                "answer_starts": [na["answer_start"]], "substitute": sub,
                "substitutions": {a0["text"]: sub} if sub and arm != "distractor_answer" else {},
                "edit_distance_to_original": edit_distance(na["text"].lower(), a0["text"].lower()),
                "n_context_edits": v["n_context_edits"], "n_question_edits": v["n_question_edits"],
                "answer_changed": na["text"] != a0["text"],
                **item_features(v["context"], v["question"], na["text"], na["answer_start"], tok),
            })
            if tok is not None:
                row.update(passages_for(v["context"], na["answer_start"], len(na["text"]), tok, max_tokens))
                if row["answer_passage"] is None:
                    row["checks"]["answer_in_one_passage"] = False
                    row["valid"] = False
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="comma list of: advisories, bioasq, openfda")
    ap.add_argument("--n", type=int, default=400, help="questions, balanced over (source, kind)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", type=Path, default=Path("data/sources"),
                    help="downloaded registries and corpora (OSV dump in <sources>/osv-pypi/json)")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--doc-max-tokens", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reg = Registries(args.sources, args.sources / "registry_cache.json")
    wiki = WikiChecker(args.sources / "wiki_cache.json")
    tok = load_tokenizer(args.tokenizer)
    if tok is None:
        raise SystemExit("the reader tokenizer is required to crop passages")
    seeds = []
    for c in args.corpus.split(","):
        if c == "advisories":
            seeds += load_advisories(args.sources / "osv-pypi" / "json")
        elif c == "bioasq":
            seeds += load_bioasq(reg)
        elif c == "openfda":
            seeds += load_openfda(args.sources / "openfda")
        else:
            raise SystemExit(f"unknown corpus {c}")
    cropped = [x for x in (crop(s, tok, args.doc_max_tokens) for s in seeds) if x is not None]
    print(f"[build-ids] {len(seeds)} seeds, {len(cropped)} after cropping; "
          f"by (source, kind): {dict(Counter((s['source'], s['kind']) for s in cropped))}")
    # Shuffled, balanced order; items whose three substitutes cannot all be found are skipped and
    # the next item of the same (source, kind) takes their place.
    def group(s):
        return s["source"], s["kind"]

    order = sample_balanced(cropped, len(cropped), args.seed, key=group)
    quota = _quotas(Counter(map(group, order)), args.n)
    chooser = Chooser(reg, wiki, cropped)
    taken, chosen = Counter(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=sum(quota.values()), desc="substitutes") as bar:
        for i in range(0, len(order), 64):
            batch = [s for s in order[i:i + 64] if taken[group(s)] < quota[group(s)]]
            for s, subs in zip(batch, pool.map(lambda s: chooser.choose(s, args.seed), batch)):
                if taken[group(s)] < quota[group(s)] and all(subs[k] for k in ("real_swap", "fictional", "near_miss")):
                    taken[group(s)] += 1
                    chosen.append((s, subs))
                    bar.update(1)
            reg.save()
            if all(taken[k] >= q for k, q in quota.items()):
                break
    reg.save()
    items = []
    for s, subs in chosen:
        items += build_rows(s, subs, tok, args.doc_max_tokens)

    _write_jsonl(args.out / "items.jsonl", items)
    summary = _summary(items, args)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def _quotas(available: Counter, n: int) -> dict:
    """Equal share per (source, kind), capped by availability, leftovers redistributed."""
    quota, left, keys = {}, n, sorted(available, key=lambda k: available[k])
    for i, k in enumerate(keys):
        share = left // (len(keys) - i)
        quota[k] = min(available[k], share)
        left -= quota[k]
    return quota


def _summary(items, args) -> dict:
    by_arm, fail = defaultdict(Counter), defaultdict(Counter)
    for r in items:
        by_arm[r["arm"]]["total"] += 1
        by_arm[r["arm"]]["valid"] += r["valid"]
        for k, v in r["checks"].items():
            if not v:
                fail[r["arm"]][k] += 1
    orig = [r for r in items if r["arm"] == "original"]
    sub = defaultdict(list)
    for r in items:
        if r.get("answer_n_subwords") is not None and r["valid"]:
            sub[(r["id_kind"], r["arm"])].append(r["answer_n_subwords"])
    return {
        "corpus": args.corpus, "seed": args.seed, "doc_max_tokens": args.doc_max_tokens,
        "n_questions": len(orig),
        "by_source_kind": {f"{s}/{k}": n for (s, k), n in Counter((r["source"], r["id_kind"]) for r in orig).items()},
        "arms": {a: dict(c) for a, c in by_arm.items()},
        "check_failures": {a: dict(c) for a, c in fail.items()},
        "near_miss_is_real": {k: sum(bool(r["near_miss_is_real"]) for r in orig if r["id_kind"] == k)
                              for k in sorted({r["id_kind"] for r in orig})},
        "mean_answer_subwords": {f"{k}/{a}": round(sum(v) / len(v), 2) for (k, a), v in sorted(sub.items())},
    }


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
