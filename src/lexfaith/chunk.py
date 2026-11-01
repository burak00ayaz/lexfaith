"""Sentence-aligned passages that fit the compressor's document budget.

PISCO compresses documents of at most 128 reader tokens (config `doc_max_length`); longer inputs are
truncated. SQuAD paragraphs often exceed this, and fictional names tokenize into more subwords than
the names they replace, so a fixed truncation could cut the answer in one arm but not another.
Every variant is therefore chunked independently, and the chunk holding the answer is recorded.
"""

import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e > s]


def chunk(text: str, tok, max_tokens: int = 128) -> list[tuple[int, int]]:
    """Greedy packing of whole sentences into chunks of <= max_tokens; overlong sentences are split
    at word boundaries. Returns character spans into `text`."""
    def ntok(s, e):
        return len(tok.encode(text[s:e], add_special_tokens=False))

    pieces = []
    for s, e in sentence_spans(text):
        if ntok(s, e) <= max_tokens:
            pieces.append((s, e))
            continue
        words = [(m.start() + s, m.end() + s) for m in re.finditer(r"\S+", text[s:e])]
        cur = words[0][0]
        last = cur
        for ws, we in words:
            if ntok(cur, we) > max_tokens and last > cur:
                pieces.append((cur, last))
                cur = ws
            last = we
        pieces.append((cur, last))
    chunks = []
    for s, e in pieces:
        if chunks and ntok(chunks[-1][0], e) <= max_tokens:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return chunks


def passages_for(text: str, answer_start: int, answer_len: int, tok, max_tokens: int = 128) -> dict:
    spans = chunk(text, tok, max_tokens)
    idx = next((i for i, (s, e) in enumerate(spans) if s <= answer_start and answer_start + answer_len <= e),
               None)
    return {
        "passages": [text[s:e] for s, e in spans],
        "answer_passage": idx,  # None if the answer straddles a chunk boundary
        "context_n_tokens": len(tok.encode(text, add_special_tokens=False)),
        "answer_token_start": len(tok.encode(text[:answer_start], add_special_tokens=False)),
    }
