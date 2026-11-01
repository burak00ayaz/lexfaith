# lexfaith — entity-substitution benchmark for soft context compression

Builds controlled variants of SQuAD v1.1 items in which names and values in the passage (and,
depending on the arm, the question and answer) are replaced consistently, then scores readers
(e.g. PISCO vs. its uncompressed Mistral-7B-Instruct-v0.2 decoder) on every variant.

Pipeline (`lexfaith-build`):
1. Sample SQuAD paragraphs (round-robin over articles, fixed seed).
2. spaCy NER + POS over passage and questions -> mention inventory (OntoNotes types, untagged
   PROPN runs, alias groups such as "Barack Obama"/"Obama" and acronyms).
3. One LLM call per paragraph with a fixed system prompt (`src/lexfaith/prompts.py`) returns, per
   mention, a fine type, a keep flag, an invented substitute (`fictional`) and a real same-type
   substitute (`real_swap`); numbers/dates get one perturbed value with a shared year offset.
4. String-level, longest-first, word-bounded replacement with offset tracking builds six arms:
   `original`, `fictional_all`, `realswap_all`, `fictional_answer`, `realswap_answer`,
   `nearmiss_answer` (rule-based one-character edit of the answer mention, e.g. Obama -> Obema).
5. Hard checks (answer recoverable at mapped offset, no residual original surface, answer changed for
   answer arms, invented answer is not a Wikipedia title) set `valid`; soft flags are kept for audit.
6. Features for analysis: answer NER type / type group / LLM fine type, POS pattern, length in chars
   and reader subwords, word frequency, position, occurrences, question-context overlap, and
   sentence-aligned passages that fit the compressor's 128-token document budget.

```sh
uv sync
# any OpenAI-compatible endpoint; e.g. a local mlx_lm.server or an API model
uv run --locked lexfaith-build --n-paragraphs 40 --out data/pilot \
    --llm-model <model> --llm-base-url http://localhost:8765/v1
# predictions: JSONL {"system", "qid", "arm", "prediction"}
uv run --locked lexfaith-eval --data data/pilot --predictions preds.jsonl \
    --compressed pisco --reference mistral
```

`lexfaith-eval` reports Match/EM/F1 per system x arm (paragraph-clustered bootstrap CIs), by answer
type, memorization ratio, an error taxonomy (correct / distractor / parametric / near_miss / context /
other), and the difference-in-differences of the arm-induced drop between compressed and uncompressed
readers.

## Identifier preservation (domain corpora)

`lexfaith-build-ids` builds the same answer-scope arms for questions whose answer is a domain
identifier, without an LLM: substitutes come from registries and rules (`identifiers.py`,
`registries.py`), questions from existing datasets or structured records (`corpora.py`).

| corpus | source | identifier kinds |
|---|---|---|
| `advisories` | PyPI security advisories (OSV dump, PYSEC: NVD descriptions; CWE from the aliased GHSA) | version, CVE, CWE, package |
| `bioasq` | BioASQ 7b/10b factoid questions | gene (HGNC), drug (RxNorm + ATC) |
| `openfda` | FDA label "INDICATIONS AND USAGE" first sentence, templated question | drug brand name |

Arms: `original`, `realswap_answer` (another real identifier of the kind), `fictional_answer`
(well-formed, unregistered, Levenshtein >= 2), `nearmiss_answer` (exactly one character changed; whether
it is itself a registered identifier is recorded in `near_miss_is_real`), `distractor_answer` (original
kept, "<answer> should not be confused with <near-miss>." inserted). Each passage is cropped to the
sentence-aligned <= 128-token window holding the answer, i.e. one PISCO document.

```sh
# one-time downloads into data/sources/ (HGNC and CWE lists are fetched automatically)
mkdir -p data/sources/osv-pypi && curl -so data/sources/osv-pypi/all.zip \
    https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip && \
    unzip -qo data/sources/osv-pypi/all.zip -d data/sources/osv-pypi/json
uv run --locked lexfaith-build-ids --corpus advisories --n 400 --out data/ids-advisories
uv run --locked lexfaith-build-ids --corpus bioasq,openfda --n 400 --out data/ids-medical
uv run --locked lexfaith-audit --data data/ids-medical --out audit-medical.md --n-paragraphs 10
```
