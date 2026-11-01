"""Seed QA items whose answer is a domain identifier, one loader per corpus.

A seed item is {qid, para_id, source, title, context, question, answers: [{text, answer_start}],
kind, meta}. `para_id` is the bootstrap cluster (one source record), `kind` selects the
perturbation rules in `identifiers.py`.

  advisories  PyPI security advisories (OSV dump, PYSEC records). The passage is the NVD-style
              description prefixed with its CVE (and CWE, taken from the aliased GHSA record);
              templated questions ask for the fixed version, the CVE, the CWE or the package.
  bioasq      BioASQ factoid questions (7b with gold spans, 10b with answer strings) whose answer
              is an HGNC gene symbol or an RxNorm drug ingredient/brand. Expert-written questions.
  openfda     FDA drug labels: the first sentence of INDICATIONS AND USAGE ("XARELTO is a factor
              Xa inhibitor indicated for ...") becomes "Which factor Xa inhibitor is indicated for
              ...?"; the answer is the brand name.
"""

import glob
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from wordfreq import zipf_frequency

from lexfaith.identifiers import occurrences

_MARKDOWN = re.compile(r"(^|\n)\s*(#|```|\||[*-] )|\]\(")
_WS = re.compile(r"\s+")


# ---- advisories ----------------------------------------------------------------------------------
def load_advisories(osv_dir: Path) -> list[dict]:
    osv_dir = Path(osv_dir)
    ghsa_cwe = {}
    for f in glob.glob(str(osv_dir / "GHSA-*.json")):
        d = json.loads(Path(f).read_text())
        ghsa_cwe[d["id"]] = (d.get("database_specific") or {}).get("cwe_ids") or []
    seeds = []
    for f in sorted(glob.glob(str(osv_dir / "PYSEC-*.json"))):
        d = json.loads(Path(f).read_text())
        details = _WS.sub(" ", d.get("details") or "").strip()
        cves = [a for a in d.get("aliases", []) if a.startswith("CVE-")]
        if (d.get("withdrawn") or len(cves) != 1 or not d.get("affected") or "withdrawn" in details.lower()
                or not 80 <= len(details) <= 800 or _MARKDOWN.search(d.get("details") or "")):
            continue
        cve = cves[0]
        if cve in details:  # the passage header carries the CVE; a second copy would make it trivial
            continue
        pkg = d["affected"][0]["package"]["name"]
        fixed = sorted({e["fixed"] for a in d["affected"] for r in a.get("ranges", [])
                        for e in r.get("events", []) if "fixed" in e})
        fixed_in_text = [v for v in fixed if occurrences(details, v)]
        cwes = sorted({c for g in d.get("aliases", []) if g.startswith("GHSA-") for c in ghsa_cwe.get(g, [])})
        cwe = cwes[0] if len(cwes) == 1 else None
        context = f"{cve} ({cwe}): {details}" if cwe else f"{cve}: {details}"
        pkg_occ = occurrences(context, pkg, ignore_case=True)
        pkg_surface = context[pkg_occ[0][0]:pkg_occ[0][1]] if pkg_occ else None
        base = {"para_id": d["id"], "source": "advisories", "title": pkg, "context": context,
                "meta": {"package": pkg, "cve": cve, "cwe": cwe, "fixed": fixed}}

        def add(kind, question, answer):
            occ = occurrences(context, answer, ignore_case=kind == "package")
            if not occ:
                return
            seeds.append({**base, "qid": f"{d['id']}:{kind}", "kind": kind, "question": question,
                          "answers": [{"text": context[occ[0][0]:occ[0][1]], "answer_start": occ[0][0]}]})

        name = pkg_surface or pkg
        if len(fixed_in_text) == 1:
            add("version", f"Which version of {name} fixes {cve}?", fixed_in_text[0])
            if pkg_surface:
                add("cve", f"Which CVE affects {name} versions before {fixed_in_text[0]}?", cve)
        if cwe:
            add("cwe", f"Which CWE weakness is {cve} classified under?", cwe)
        if pkg_surface and len(pkg) >= 4:
            add("package", f"Which Python package is affected by {cve}?", pkg_surface)
    return seeds


# ---- BioASQ --------------------------------------------------------------------------------------
def classify_bio(answer: str, reg) -> str | None:
    """'gene' for an HGNC symbol that looks like one, 'drug' for an RxNorm ingredient or brand with
    an ATC class; common words (zinc, iron, liver) are excluded by word frequency."""
    if len(answer) < 3 or zipf_frequency(answer, "en") >= 3.0:
        return None
    genes = reg.genes()
    approved = genes.get(answer) == answer or genes.get(answer.upper()) == answer.upper()
    if (re.fullmatch(r"[A-Za-z0-9-]+", answer) and (approved or (reg.gene_exists(answer) and re.search(r"\d", answer)))
            and (re.search(r"\d", answer) or sum(c.isupper() for c in answer) >= 2)):
        return "gene"
    if re.fullmatch(r"[A-Za-z][A-Za-z-]+", answer) and reg.is_drug(answer) and reg.atc_class(answer):
        return "drug"
    return None


def load_bioasq(reg) -> list[dict]:
    from datasets import load_dataset

    by_q: dict[str, dict] = {}
    for split in ("train", "validation"):
        for r in load_dataset("reginaboateng/Bioasq7b_factoid", split=split):
            qid = r["id"].rsplit("_", 1)[0]
            a, s = r["answers"]["text"][0], r["answers"]["answer_start"][0]
            by_q.setdefault(qid, {"question": r["question"], "context": r["context"], "answer": a, "start": s})
    for split in ("train", "test"):
        for r in load_dataset("legacy107/bioasq10b-factoid", split=split):
            if r["id"] not in by_q and r["answer"].strip() in r["context"]:
                by_q[r["id"]] = {"question": r["question"], "context": r["context"], "answer": r["answer"],
                                 "start": None}
    seeds = []
    for qid, r in sorted(by_q.items()):
        ans = r["answer"].strip().strip("()").strip()
        ctx = _WS.sub(" ", r["context"]).strip()
        kind = classify_bio(ans, reg)
        if kind is None or occurrences(r["question"], ans, ignore_case=True):
            continue
        occ = occurrences(ctx, ans)
        if not occ:
            continue
        seeds.append({"qid": f"bioasq:{qid}", "para_id": f"bioasq:{qid}", "source": "bioasq", "title": ans,
                      "context": ctx, "question": r["question"].strip(), "kind": kind,
                      "answers": [{"text": ans, "answer_start": occ[0][0]}], "meta": {}})
    return seeds


# ---- openFDA labels ------------------------------------------------------------------------------
OPENFDA = "https://api.fda.gov/drug/label.json"
_LABEL_QUERY = ('_exists_:openfda.pharm_class_epc AND _exists_:openfda.brand_name AND '
                '_exists_:indications_and_usage AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"')
_IND = re.compile(r"^(?P<head>.{2,80}?)\s+(?:is|are)\s+(?:an?\s+(?P<cls>[a-zA-Z0-9 ()/-]{3,60}?)\s+)?"
                  r"indicated\s+(?P<rest>(?:for|to|as|in)\s[^.;:•]{10,200}?)\.(?:\s|$)")


def fetch_labels(cache_dir: Path, pages: int = 100) -> list[dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(pages):
        path = cache_dir / f"labels_{i:03d}.json"
        if not path.exists():
            r = requests.get(OPENFDA, params={"search": _LABEL_QUERY, "limit": 100, "skip": 100 * i}, timeout=60)
            r.raise_for_status()
            path.write_text(r.text)
            time.sleep(0.5)
        out += json.loads(path.read_text()).get("results", [])
    return out


def load_openfda(cache_dir: Path, reg=None) -> list[dict]:
    seeds, seen = [], set()
    for lab in fetch_labels(cache_dir):
        o = lab.get("openfda", {})
        brands, generics = o.get("brand_name", []), o.get("generic_name", [])
        if len(brands) != 1 or not generics:
            continue
        brand = brands[0].strip()
        if (brand.lower() in seen or not re.fullmatch(r"[A-Za-z][A-Za-z-]{3,}", brand)
                or any(brand.lower() in g.lower() or g.lower() in brand.lower() for g in generics)):
            continue
        text = _WS.sub(" ", lab["indications_and_usage"][0])
        text = re.sub(r"^\s*(?:\d+(?:\.\d+)?\s+)?INDICATIONS\s*(?:AND|&)\s*USAGE\s*", "", text, flags=re.I).strip()
        m = _IND.match(text)
        if not m or not occurrences(m.group("head"), brand, ignore_case=True):
            continue
        rest = m.group("rest").strip()
        cls = (m.group("cls") or "drug").strip()
        question = f"Which {cls} is indicated {rest}?"
        generic = generics[0].lower()
        # One label per indication and per ingredient: two brands of one generic answer the same question.
        if (len(rest.split()) > 30 or occurrences(rest, brand, ignore_case=True)
                or question.lower() in seen or generic in seen):
            continue
        seen |= {brand.lower(), question.lower(), generic}
        occ = occurrences(text, brand, ignore_case=True)
        set_id = lab.get("set_id") or brand
        seeds.append({
            "qid": f"openfda:{set_id}", "para_id": f"openfda:{set_id}", "source": "openfda", "title": brand,
            "context": text, "question": question, "kind": "drug",
            "answers": [{"text": text[occ[0][0]:occ[0][1]], "answer_start": occ[0][0]}],
            "meta": {"brand": brand, "generic": generics, "pharm_class_epc": o.get("pharm_class_epc", []),
                     "is_brand": True},
        })
    return seeds


LOADERS = {"advisories", "bioasq", "openfda"}


def sample_balanced(seeds: list[dict], n: int, seed: int, key=lambda s: s["kind"]) -> list[dict]:
    """Up to n items, round-robin over `key` groups, at most one question per source record."""
    rng = random.Random(seed)
    by_kind = defaultdict(list)
    for s in seeds:
        by_kind[key(s)].append(s)
    for k in by_kind:
        rng.shuffle(by_kind[k])
    out, used = [], set()
    kinds = sorted(by_kind)
    while len(out) < n and any(by_kind.values()):
        for k in kinds:
            while by_kind[k]:
                s = by_kind[k].pop()
                if s["para_id"] not in used:
                    used.add(s["para_id"])
                    out.append(s)
                    break
            if len(out) >= n:
                break
    return out
