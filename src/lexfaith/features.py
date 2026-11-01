"""Per-item analysis features (answer type, form, frequency, tokenization, position, overlap)."""

import re
from functools import lru_cache

from wordfreq import zipf_frequency

from lexfaith.annotate import TYPE_GROUP

_WORD = re.compile(r"[A-Za-z]+")
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "and", "or", "is", "was", "what",
         "which", "who", "whom", "when", "where", "why", "how", "did", "does", "do", "that", "with", "as",
         "from", "it", "its", "are", "were", "be", "been", "this", "these", "those", "many", "much"}


@lru_cache(maxsize=2)
def load_tokenizer(name: str | None):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # gated or offline: subword features are skipped
        print(f"[features] tokenizer {name!r} unavailable ({type(e).__name__}); skipping subword counts")
        return None


def n_subwords(tok, text: str) -> int | None:
    # Leading space matches how the span is tokenized mid-sentence.
    return None if tok is None else len(tok.encode(" " + text, add_special_tokens=False))


def min_zipf(text: str) -> float | None:
    words = [w for w in _WORD.findall(text) if w.lower() not in _STOP]
    return min((zipf_frequency(w, "en") for w in words), default=None)


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP}


def answer_type(answers: list[dict], inventory: list[dict], mapping: dict) -> dict:
    """Tagger type of the mention covering most of the first gold answer (>=50% of its chars)."""
    a = answers[0]
    s, e = a["answer_start"], a["answer_start"] + len(a["text"])
    best, best_cov = None, 0
    for inv in inventory:
        for ms, me in inv["spans"]:
            cov = max(0, min(e, me) - max(s, ms))
            if cov > best_cov:
                best, best_cov = inv, cov
    if best is None or best_cov < 0.5 * max(1, e - s):
        return {"answer_ner_type": "NONE", "answer_type_group": "NON_ENTITY", "answer_fine_type": None,
                "answer_is_entity": False, "answer_entity_exact": False}
    return {
        "answer_ner_type": best["type"],
        "answer_type_group": TYPE_GROUP[best["type"]],
        "answer_fine_type": mapping[best["id"]]["fine_type"] if mapping else None,
        "answer_is_entity": True,
        "answer_entity_exact": best["surface"] == a["text"].strip(),
    }


def item_features(context: str, question: str, answer_text: str, answer_start: int, tok) -> dict:
    ctx_words = content_words(context)
    q_words = content_words(question)
    return {
        "answer_n_chars": len(answer_text),
        "answer_n_words": len(answer_text.split()),
        "answer_n_subwords": n_subwords(tok, answer_text),
        "answer_min_zipf": min_zipf(answer_text),
        "answer_n_occurrences": len(re.findall(r"(?<!\w)" + re.escape(answer_text) + r"(?!\w)", context))
        if answer_text else 0,
        "answer_rel_position": answer_start / max(1, len(context)),
        "context_n_chars": len(context),
        "question_context_overlap": len(q_words & ctx_words) / max(1, len(q_words)),
    }
