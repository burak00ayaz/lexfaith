"""Apply a paragraph mapping to context, question and answers, tracking character offsets.

Arms (one variant per question per arm):
  original          unchanged SQuAD item
  fictional_all     every substitutable mention -> invented name / perturbed value
  realswap_all      every substitutable mention -> other real entity / perturbed value
  fictional_answer  only the answer's mention(s) and their aliases -> invented name
  realswap_answer   only the answer's mention(s) and their aliases -> other real entity
  nearmiss_answer   only the answer's mention(s) and their aliases -> one character changed
                    ("Obama" -> "Obema", "1961" -> "1962"); rule-based, see identifiers.py

`*_all` removes the parametric route (the question itself is rewritten); `*_answer` keeps the
question real, so the context conflicts with memory (Longpre et al., 2021 style).
"""

import re

ARMS = ["original", "fictional_all", "realswap_all", "fictional_answer", "realswap_answer", "nearmiss_answer"]
POLICY = {"fictional": "fictional", "realswap": "real_swap", "nearmiss": "near_miss"}


def _alt(s: str) -> str:
    # Guard a side against partial-word matches only if the surface starts/ends with a word char
    # ("0.3 °" must still match inside "0.3 °C").
    pre = r"(?<![\w-])" if re.match(r"\w", s[0]) else ""
    post = r"(?![\w-])" if re.match(r"\w", s[-1]) else ""
    return pre + re.escape(s) + post


def _pattern(surfaces: list[str]) -> re.Pattern | None:
    if not surfaces:
        return None
    alts = sorted(set(surfaces), key=len, reverse=True)  # longest match wins at each position
    return re.compile("|".join(_alt(s) for s in alts))


def rewrite(text: str, table: dict[str, str]) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Replace every occurrence of table keys. Returns new text and (old_s, old_e, new_s, new_e) edits."""
    pat = _pattern(list(table))
    if pat is None:
        return text, []
    out, edits, pos, shift = [], [], 0, 0
    for m in pat.finditer(text):
        rep = table[m.group(0)]
        out.append(text[pos:m.start()])
        out.append(rep)
        ns = m.start() + shift
        edits.append((m.start(), m.end(), ns, ns + len(rep)))
        shift += len(rep) - (m.end() - m.start())
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), edits


def map_offset(pos: int, edits, side: str) -> int:
    """Map an old character offset to the new text; positions inside an edit snap to its boundary."""
    shift = 0
    for os_, oe, ns, ne in edits:
        if pos <= os_:
            break
        if pos < oe or (pos == oe and side == "end"):
            return ns if side == "start" else ne
        shift = ne - oe
    return pos + shift


def answer_mention_ids(answers: list[dict], inventory: list[dict], mapping: dict) -> set[str]:
    """Mentions overlapping any gold answer span, expanded to their alias groups."""
    spans = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in answers]
    ids = set()
    for inv in inventory:
        if mapping[inv["id"]]["keep"]:
            continue
        overlaps = any(s < ae and e > as_ for s, e in inv["spans"] for as_, ae in spans)
        # Replacement is string-level, so a surface inside the answer text changes it even if the
        # tagger missed that particular occurrence.
        inside = any(_pattern([inv["surface"]]).search(a["text"]) for a in answers)
        if overlaps or inside:
            ids.add(inv["id"])
    groups = {mapping[i]["group"] for i in ids if mapping[i]["group"]}
    ids |= {i for i, m in mapping.items() if m["group"] in groups and not m["keep"]}
    return ids


def _mask(text: str, edits) -> str:
    out, pos = [], 0
    for _, _, ns, ne in edits:
        out += [text[pos:ns], " " * (ne - ns)]
        pos = ne
    return "".join(out) + text[pos:]


def make_variant(context: str, question: str, answers: list[dict], mapping: dict,
                 ids: set[str], policy: str) -> dict:
    """Rewrite one QA item with the substitutes of `ids` under `policy` (a value of POLICY)."""
    table = {mapping[i]["surface"]: mapping[i][policy] for i in ids if not mapping[i]["keep"]}
    new_ctx, edits = rewrite(context, table)
    new_q, q_edits = rewrite(question, table)
    new_answers = []
    for a in answers:
        s, e = a["answer_start"], a["answer_start"] + len(a["text"])
        ns, ne = map_offset(s, edits, "start"), map_offset(e, edits, "end")
        by_string, _ = rewrite(a["text"], table)
        # An answer that is part of a longer mention ("1961" in "early 1961") snaps to the whole
        # rewritten mention; recover the rewritten answer inside it when it is there.
        if new_ctx[ns:ne] != by_string and by_string and by_string in new_ctx[ns:ne]:
            ns += new_ctx[ns:ne].index(by_string)
            ne = ns + len(by_string)
        new_answers.append({"text": new_ctx[ns:ne], "answer_start": ns, "text_by_string": by_string})
    changed_chars = sum(oe - os_ for os_, oe, _, _ in edits)
    # Originals still present outside the inserted substitutes (a substitute may legitimately keep a
    # head word, e.g. "Republic" in "Velorian Republic").
    masked = _mask(new_ctx, edits) + "\n" + _mask(new_q, q_edits)
    residual = sorted(k for k, v in table.items() if k != v and _pattern([k]).search(masked))
    return {
        "context": new_ctx,
        "question": new_q,
        "answers": new_answers,
        "table": table,
        "n_context_edits": len(edits),
        "n_question_edits": len(q_edits),
        "frac_context_chars_changed": changed_chars / max(1, len(context)),
        "residual_originals": residual,
    }
