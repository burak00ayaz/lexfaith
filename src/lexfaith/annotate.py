"""NER + POS annotation of paragraphs and questions, and mention-inventory construction."""

import re
from functools import lru_cache

import spacy

from lexfaith.prompts import NAMED_TYPES, NUMERIC_TYPES

TYPE_GROUP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION",
    "NORP": "NORP",
    "PRODUCT": "OTHER_NAME", "EVENT": "OTHER_NAME", "WORK_OF_ART": "OTHER_NAME",
    "LAW": "OTHER_NAME", "LANGUAGE": "OTHER_NAME", "PROPN": "OTHER_NAME",
    "DATE": "TEMPORAL", "TIME": "TEMPORAL",
    "CARDINAL": "NUMERIC", "QUANTITY": "NUMERIC", "MONEY": "NUMERIC", "PERCENT": "NUMERIC",
    "NONE": "NON_ENTITY",
}

_LEADING_DET = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


@lru_cache(maxsize=2)
def load_nlp(model: str = "en_core_web_trf"):
    return spacy.load(model)


def _clean_span(text: str, start: int) -> tuple[str, int]:
    """Strip a leading determiner and trailing possessive/punctuation from an entity span."""
    m = _LEADING_DET.match(text)
    if m:
        text, start = text[m.end():], start + m.end()
    text = re.sub(r"(’s|'s)$", "", text)
    text = text.rstrip(" ,.;:")
    return text, start


def _substitutable(label: str, text: str) -> bool:
    if label in NUMERIC_TYPES:
        # Spelled-out numbers ("two", "first") and vague dates ("today") are left alone:
        # string-level replacement of common words is unsafe.
        return bool(re.search(r"\d", text))
    return label in NAMED_TYPES and len(text) >= 2


def mentions_from_doc(doc) -> list[dict]:
    """Entity mentions (NER) plus untagged proper-noun runs (POS=PROPN)."""
    out = []
    covered = set()
    for ent in doc.ents:
        text, start = _clean_span(ent.text, ent.start_char)
        if not _substitutable(ent.label_, text):
            continue
        out.append({"surface": text, "type": ent.label_, "start": start, "end": start + len(text)})
        covered.update(range(ent.start, ent.end))
    # Proper-noun runs the tagger missed; the LLM decides whether they are real names (keep rule).
    run = []
    for tok in list(doc) + [None]:
        if tok is not None and tok.pos_ == "PROPN" and tok.i not in covered and tok.text[:1].isupper():
            run.append(tok)
            continue
        if run:
            text = doc.text[run[0].idx: run[-1].idx + len(run[-1].text)]
            if len(text) >= 3 and not all(t.is_stop for t in run):
                out.append({"surface": text, "type": "PROPN", "start": run[0].idx, "end": run[0].idx + len(text)})
            run = []
    return out


def _alias_groups(inv: list[dict]) -> None:
    """Union-find over named mentions: token-subsequence aliases and acronyms share a group."""
    parent = list(range(len(inv)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    named = [i for i, e in enumerate(inv) if e["type"] in NAMED_TYPES]
    toks = {i: inv[i]["surface"].split() for i in named}
    for a in named:
        for b in named:
            if a == b or len(toks[a]) >= len(toks[b]):
                continue
            ta, tb = toks[a], toks[b]
            sub = any(tb[k:k + len(ta)] == ta for k in range(len(tb) - len(ta) + 1))
            acronym = len(ta) == 1 and ta[0].isupper() and len(ta[0]) >= 2 and ta[0] == "".join(
                w[0] for w in tb if w[:1].isupper())
            if sub or acronym:
                parent[find(a)] = find(b)
    roots = {}
    for i, e in enumerate(inv):
        if e["type"] in NAMED_TYPES:
            r = find(i)
            members = [j for j in range(len(inv)) if inv[j]["type"] in NAMED_TYPES and find(j) == r]
            if len(members) > 1:
                roots.setdefault(r, f"G{len(roots) + 1}")
                e["group"] = roots[r]


def build_inventory(context_doc, question_docs) -> list[dict]:
    """Unique surface forms over the paragraph and its questions, with mention counts."""
    by_surface: dict[str, dict] = {}
    for src, doc in [("context", context_doc)] + [("question", d) for d in question_docs]:
        for m in mentions_from_doc(doc):
            e = by_surface.get(m["surface"])
            if e is None:
                e = by_surface[m["surface"]] = {
                    "surface": m["surface"], "type": m["type"], "group": None,
                    "n_context": 0, "n_question": 0, "spans": [],
                }
            elif e["type"] == "PROPN" and m["type"] != "PROPN":
                e["type"] = m["type"]
            e[f"n_{src}"] += 1
            if src == "context":
                e["spans"].append([m["start"], m["end"]])
    inv = sorted(by_surface.values(), key=lambda e: (e["spans"][0][0] if e["spans"] else 10**9, e["surface"]))
    _alias_groups(inv)
    for i, e in enumerate(inv):
        e["id"] = f"E{i + 1}"
    return inv


def span_pos(doc, start: int, end: int) -> tuple[list[str], str]:
    """UPOS sequence of tokens overlapping [start, end) and POS of the span's syntactic head."""
    toks = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
    if not toks:
        return [], "NONE"
    ids = {t.i for t in toks}
    heads = [t for t in toks if t.head.i not in ids or t.head.i == t.i]
    return [t.pos_ for t in toks], (heads[0].pos_ if heads else toks[0].pos_)
