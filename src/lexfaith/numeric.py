"""Deterministic perturbation of numeric and temporal mentions.

Years in a paragraph move by one shared offset (a non-zero multiple of 10, so decades such as
"1980s" stay well-formed and chronology is preserved). Other numbers move to a clearly different
value of the same magnitude and format. The same value is used for both substitution policies.
"""

import random
import re

_YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-2][0-9])(?!\d)")
_NUM = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\d.,])\d+(?:\.\d+)?(?![\d.])")
MAX_YEAR = 2024


def paragraph_rng(para_id: str, seed: int) -> random.Random:
    return random.Random(f"{seed}:{para_id}")


def choose_year_offset(years: list[int], rng: random.Random) -> int:
    """Non-zero multiple of 10 in [-40, 40] that keeps every shifted year in [1000, MAX_YEAR]."""
    options = [o for o in (-40, -30, -20, -10, 10, 20, 30, 40)
               if all(1000 <= y + o <= MAX_YEAR for y in years)]
    return rng.choice(options or [-10])


def _perturb_number(tok: str, rng: random.Random, is_percent: bool) -> str:
    commas = "," in tok
    raw = tok.replace(",", "")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    v = float(raw)
    if v < 10 and decimals == 0:
        new = rng.choice([x for x in range(2, 10) if x != int(v)])
    else:
        for _ in range(20):
            factor = rng.choice([rng.uniform(0.4, 0.75), rng.uniform(1.3, 2.2)])
            new = round(v * factor, decimals)
            if is_percent and new > 100:
                continue
            if new != v:
                break
        if decimals == 0:
            new = int(new)
            # keep a round-looking number round ("5,000" -> "8,000", not "8,137")
            zeros = len(str(int(v))) - len(str(int(v)).rstrip("0"))
            if zeros:
                new = max(10 ** zeros, round(new, -zeros))
    text = f"{new:,.{decimals}f}" if commas else f"{new:.{decimals}f}"
    return text


def perturb(surface: str, label: str, year_offset: int, rng: random.Random, memo: dict) -> str | None:
    """Perturbed surface form, or None when nothing substitutable was found. `memo` keeps a number
    that recurs in several mentions of one paragraph mapped to the same new value."""
    if label in ("DATE",):
        def shift_year(m):
            return str(int(m.group(1)) + year_offset)
        new = _YEAR.sub(shift_year, surface)
        return new if new != surface else None
    if label in ("CARDINAL", "QUANTITY", "MONEY", "PERCENT"):
        # Bare 4-digit years tagged as CARDINAL follow the year offset.
        if label == "CARDINAL" and _YEAR.fullmatch(surface) and 1000 <= int(surface) <= MAX_YEAR:
            return str(int(surface) + year_offset)
        def repl(m):
            key = (m.group(0), label == "PERCENT")
            if key not in memo:
                memo[key] = _perturb_number(m.group(0), rng, label == "PERCENT")
            return memo[key]
        new = _NUM.sub(repl, surface)
        return new if new != surface else None
    return None  # TIME and others are kept


def numeric_mapping(inventory: list[dict], para_id: str, seed: int) -> tuple[dict, int]:
    rng = paragraph_rng(para_id, seed)
    years = [int(y) for e in inventory if e["type"] in ("DATE", "CARDINAL") for y in _YEAR.findall(e["surface"])]
    offset = choose_year_offset(years, rng)
    mapping, memo = {}, {}
    for e in inventory:
        if e["type"] not in ("DATE", "TIME", "CARDINAL", "QUANTITY", "MONEY", "PERCENT"):
            continue
        new = perturb(e["surface"], e["type"], offset, rng, memo)
        mapping[e["id"]] = {
            "surface": e["surface"], "type": e["type"], "group": None, "same_as": None,
            "fine_type": "year/date" if e["type"] == "DATE" else e["type"].lower(),
            "keep": new is None, "fictional": new or e["surface"], "real_swap": new or e["surface"],
            "source": "rule",
        }
    return mapping, offset
