"""Substitutes for domain identifiers: the edit-distance ladder.

For an answer identifier `s` of a given kind, each arm needs one substitute:
  real_swap   another real identifier of the same kind (conflicts with parametric memory)
  fictional   well-formed for the kind but not in its registry (no parametric route); kept at
              Levenshtein distance >= 2 so it cannot be mistaken for a near-miss
  near_miss   exactly one character changed (`metoprolol -> metaprolol`, `3.11.4 -> 3.11.5`);
              tests whether the compressed representation keeps the exact string
  distractor  the original is kept and a near-miss sibling is inserted next to it

Kinds: drug, gene, version, cve, cwe, package, and `name` / `number` for SQuAD mentions.
Everything here is deterministic given the rng; registry checks are done by the caller.
"""

import random
import re

from rapidfuzz.distance import Levenshtein

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnprstvz"  # no q/w/x/y: keeps coinages pronounceable
# Visually / phonetically close consonants, for near-misses that look like real misreadings.
CLOSE = {"b": "dp", "c": "kgs", "d": "bt", "f": "pv", "g": "ck", "h": "n", "j": "g", "k": "cg",
         "l": "rt", "m": "n", "n": "mr", "p": "bf", "r": "ln", "s": "cz", "t": "dl", "v": "fz",
         "x": "z", "z": "s", "q": "k", "w": "v", "y": "i"}

# USAN / INN stems, longest first, so a coined name keeps the pharmacological class signal.
DRUG_STEMS = sorted("""
mab ximab zumab umab tinib ciclib parib rafenib degib lisib olol pril sartan statin xaban gatran
gliptin gliflozin glutide prazole tidine dipine navir previr asvir buvir vir mycin cillin floxacin
cycline conazole azole setron triptan semide thiazide lukast afil dronate lutamide azepam pam
relin cept kinra plase parin mide oxetine ipramine apine peridol profen coxib olone asone sone nide
siran tide stat dil caine fungin bactam penem cef sulfa lukin kin nib zomib ib ine one ol il in
""".split(), key=len, reverse=True)

ID_BOUND_PRE = r"(?<![\w.-])"
ID_BOUND_POST = r"(?![\w-]|\.\w)"  # "1.2" must not match inside "1.2.3"; a sentence-final "." is fine


# ---- matching ------------------------------------------------------------------------------------
def id_pattern(s: str, ignore_case: bool = False) -> re.Pattern:
    return re.compile(ID_BOUND_PRE + re.escape(s) + ID_BOUND_POST, re.I if ignore_case else 0)


def occurrences(text: str, s: str, ignore_case: bool = False) -> list[tuple[int, int]]:
    return [m.span() for m in id_pattern(s, ignore_case).finditer(text)]


def match_case(sub: str, like: str) -> str:
    """Render `sub` in the case style of `like` (XARELTO / Xarelto / xarelto); mixed case is kept."""
    if like.isupper() and any(c.isalpha() for c in like):
        return sub.upper()
    if like.islower():
        return sub.lower()
    if like[:1].isupper() and like[1:].islower():
        return sub[:1].upper() + sub[1:].lower()
    return sub


def rewrite_id(text: str, old: str, new: str, ignore_case: bool = False):
    """Replace every bounded occurrence of `old`; returns (text, edits) in substitute.rewrite format."""
    out, edits, pos, shift = [], [], 0, 0
    for m in id_pattern(old, ignore_case).finditer(text):
        rep = match_case(new, m.group(0)) if ignore_case else new
        out += [text[pos:m.start()], rep]
        ns = m.start() + shift
        edits.append((m.start(), m.end(), ns, ns + len(rep)))
        shift += len(rep) - (m.end() - m.start())
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), edits


def edit_distance(a: str, b: str) -> int:
    return Levenshtein.distance(a, b)


# ---- near-miss: exactly one character -----------------------------------------------------------
def _digit_edits(s: str) -> list[str]:
    """Change the last digit by +-1 (then +-2), keeping the digit count."""
    idx = [i for i, c in enumerate(s) if c.isdigit()]
    if not idx:
        return []
    i = idx[-1]
    d = int(s[i])
    out = []
    for delta in (1, -1, 2, -2):
        nd = d + delta
        if 0 <= nd <= 9:
            out.append(s[:i] + str(nd) + s[i + 1:])
    return out


def _letter_edits(s: str, rng: random.Random) -> list[str]:
    """One interior letter of the longest alphabetic run replaced by a same-class letter."""
    runs = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", s)]
    if not runs:
        return []
    a, b = max(runs, key=lambda r: r[1] - r[0])
    # avoid the first letter (most salient) and, for longer words, the last two (often a stem)
    lo, hi = a + 1, (b - 2 if b - a >= 6 else b)
    positions = list(range(lo, hi)) or list(range(a, b))
    rng.shuffle(positions)
    out = []
    for i in positions:
        c = s[i]
        low = c.lower()
        if low in VOWELS:
            alts = [v for v in VOWELS if v != low]
        else:
            alts = list(CLOSE.get(low, "")) + [x for x in CONSONANTS if x != low and x not in CLOSE.get(low, "")]
        for alt in alts[:4]:
            alt = alt.upper() if c.isupper() else alt
            out.append(s[:i] + alt + s[i + 1:])
    return out


def near_miss_candidates(s: str, kind: str, rng: random.Random) -> list[str]:
    if kind in ("version", "cve", "cwe", "number"):
        cands = _digit_edits(s)
    elif kind == "gene":
        cands = _digit_edits(s) + _letter_edits(s, rng)
    else:  # drug, package, name
        cands = _letter_edits(s, rng)
    seen, out = set(), []
    for c in cands:
        if c != s and c not in seen and edit_distance(c, s) == 1:
            seen.add(c)
            out.append(c)
    return out


def add_near_misses(mapping: dict, rng: random.Random) -> None:
    """Add a one-character `near_miss` to every substitutable mention of a SQuAD paragraph mapping.
    Aliases of one referent share the edited word ("Barack Obama"/"Obama" -> "Barack Obema"/"Obema")."""
    surfaces = {m["surface"] for m in mapping.values()}
    word_edit: dict[str, tuple[str, str]] = {}  # group -> (word, edited word)
    groups: dict[str, list[dict]] = {}
    for m in mapping.values():
        if not m["keep"] and m.get("group"):
            groups.setdefault(m["group"], []).append(m)
    for g, members in groups.items():
        rep = max(members, key=lambda m: len(m["surface"]))
        words = re.findall(r"[A-Za-z]{4,}", rep["surface"])
        if words:
            w = words[-1]
            cand = next((c for c in near_miss_candidates(w, "name", rng) if c not in surfaces), None)
            if cand:
                word_edit[g] = (w, cand)
    for m in mapping.values():
        if m["keep"]:
            continue
        s = m["surface"]
        we = word_edit.get(m.get("group"))
        if we and re.search(r"\b" + re.escape(we[0]) + r"\b", s):
            m["near_miss"] = re.sub(r"\b" + re.escape(we[0]) + r"\b", we[1], s)
            continue
        kind = "number" if m.get("source") == "rule" else "name"
        m["near_miss"] = next((c for c in near_miss_candidates(s, kind, rng) if c not in surfaces), s)


# ---- fictional: well-formed, not registered, distance >= 2 -------------------------------------
def _syllables(rng: random.Random, n_chars: int) -> str:
    out = ""
    while len(out) < n_chars:
        out += rng.choice(CONSONANTS) + rng.choice(VOWELS)
    return out[:max(2, n_chars)]


def coin_drug(s: str, rng: random.Random) -> str:
    """Invented name keeping the INN stem: `tofacitinib` -> `velomatinib`."""
    low = s.lower()
    stem = next((st for st in DRUG_STEMS if low.endswith(st) and len(low) - len(st) >= 3), None)
    if stem is None:  # no known stem: keep the final vowel-initial chunk ("-ocin", "-em")
        m = re.search(r"[aeiou][^aeiou]*$", low)
        stem = m.group(0) if m and len(low) - len(m.group(0)) >= 3 else low[-2:]
    n = len(low) - len(stem)
    prefix = _syllables(rng, n + n % 2)  # CVCV...: ends in a vowel
    if stem[0] in VOWELS:
        prefix = prefix[:-1]  # ...C + vowel-initial stem
    return match_case(prefix + stem, s)


def shape_preserving(s: str, rng: random.Random) -> str:
    """Same character classes: letters -> letters (same case, vowel/consonant), digits -> digits."""
    out = []
    for i, c in enumerate(s):
        if c.isdigit():
            first = i == 0 or not s[i - 1].isdigit()
            out.append(str(rng.randint(1 if first and c != "0" else 0, 9)))
        elif c.isalpha():
            pool = VOWELS if c.lower() in VOWELS else CONSONANTS
            ch = rng.choice(pool)
            out.append(ch.upper() if c.isupper() else ch)
        else:
            out.append(c)
    return "".join(out)


_AFFIXES = {"py", "python", "django", "flask", "api", "client", "core", "sdk", "lib", "utils", "tools", "update",
            "server", "cli", "plugin", "auth", "http", "web", "admin", "extensions", "framework", "async"}


def coin_package(s: str, rng: random.Random) -> str:
    """Pronounceable package name of similar length; keeps separators and a `py` affix."""
    parts = re.split(r"([-_.])", s)
    out = []
    for p in parts:
        if p in "-_." or not p:
            out.append(p)
        elif p.lower() in _AFFIXES:
            out.append(p)  # generic affixes keep the name realistic
        elif p.lower().startswith("py") and len(p) > 4:
            out.append(p[:2] + _syllables(rng, len(p) - 2))
        else:
            out.append(_syllables(rng, len(p)))
    return match_case("".join(out), s)


def fictional_version(v: str, releases: list[str], rng: random.Random) -> str | None:
    """Same major and component count, unreleased, at distance >= 2 from `v`."""
    comps = v.split(".")
    if not all(c.isdigit() for c in comps) or len(comps) < 2:
        return None
    released = set(releases or [])
    for _ in range(200):
        new = [comps[0]] + [str(rng.randint(0, max(19, int(c) + 10))) for c in comps[1:]]
        cand = ".".join(new)
        if cand not in released and edit_distance(cand, v) >= 2:
            return cand
    return None


def fictional_candidates(s: str, kind: str, rng: random.Random, n: int = 12, releases=None,
                         unassigned=None) -> list[str]:
    """`unassigned`: for CWE, the free ids inside the catalogue's range. CWE ids are so densely
    assigned that a random id of the same length almost always exists."""
    out = []
    for _ in range(n * 5):
        if kind == "drug":
            c = coin_drug(s, rng)
        elif kind == "gene":
            c = shape_preserving(s, rng)
        elif kind == "package":
            c = coin_package(s, rng)
        elif kind == "version":
            c = fictional_version(s, releases or [], rng)
        elif kind == "cve":
            m = re.fullmatch(r"(CVE-\d{4}-)(\d+)", s)
            c = m.group(1) + shape_preserving(m.group(2), rng) if m else None
        elif kind == "cwe":
            c = f"CWE-{rng.choice(unassigned)}" if unassigned else None
        else:
            raise ValueError(kind)
        if c and c != s and edit_distance(c.lower(), s.lower()) >= 2 and c not in out:
            out.append(c)
        if len(out) >= n:
            break
    return out


# ---- real swap -----------------------------------------------------------------------------------
def version_swaps(v: str, releases: list[str], rng: random.Random) -> list[str]:
    """Other real releases with the same component count, same major first, then by closeness."""
    comps = v.split(".")
    same = [r for r in releases if r != v and re.fullmatch(r"\d+(\.\d+)*", r) and len(r.split(".")) == len(comps)]
    rng.shuffle(same)
    same.sort(key=lambda r: (r.split(".")[0] != comps[0], edit_distance(r, v) < 2))
    return same


# ---- distractor ----------------------------------------------------------------------------------
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def insert_distractor(context: str, answer: str, answer_start: int, sibling: str, kind: str) -> tuple[str, int]:
    """Insert '<answer> should not be confused with <sibling>.' after the answer's sentence, or before
    it when the (cropped) passage has no sentence end after the answer. Returns the new context and
    the new answer offset."""
    label = {"version": "version "}.get(kind, "")
    sent = f"{label}{answer} should not be confused with {label}{sibling}."
    sent = sent[0].upper() + sent[1:]
    m = _SENT_END.search(context, answer_start + len(answer))
    if m:
        return context[:m.start()] + " " + sent + context[m.start():], answer_start
    before = [b.end() for b in _SENT_END.finditer(context, 0, answer_start)]
    at = before[-1] if before else 0
    ins = sent + " "
    return context[:at] + ins + context[at:], answer_start + len(ins)
