"""LLM entity mapping: OpenAI-compatible client, response cache, and mapping validation."""

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from openai import OpenAI
from rapidfuzz import fuzz

from lexfaith.prompts import PROMPT_VERSION, REPAIR_MESSAGE, SYSTEM_PROMPT, build_user_message

# Head words that rule 2 of the prompt asks to keep; sharing them with the original is fine.
GENERIC = set("""
university college school institute institution academy state states united republic kingdom empire
federal national royal imperial government council congress parliament party church cathedral abbey
saint st sir lord lady king queen prince princess river lake mount mountain mountains island islands bay
sea ocean gulf north south east west northern southern eastern western new old great upper lower
city county province district region territory valley park street avenue road bridge tower castle palace
hall square center centre museum library hospital airport station port company corporation corp inc
ltd group association society foundation league club cup prize award festival act treaty war battle
revolution dynasty era age research science sciences education educational information network
international american the of and for de la le von van al el
""".split())


def _tokens(s: str) -> set[str]:
    return {t.strip(".,'’()").lower() for t in s.split() if len(t.strip(".,'’()")) >= 3}


def lexical_problems(surface: str, fictional: str) -> list[str]:
    """Rule-8 violations of an invented name: reused distinctive words or near-spellings."""
    out = []
    shared = (_tokens(surface) & _tokens(fictional)) - GENERIC
    if shared:
        out.append(f"fictional {fictional!r} reuses the original word(s) {sorted(shared)}")
    a = [t for t in _tokens(surface) if t not in GENERIC]
    b = [t for t in _tokens(fictional) if t not in GENERIC]
    near = [(x, y) for x in a for y in b if x != y and len(x) >= 4 and fuzz.ratio(x, y) >= 70]
    if near:
        out.append(f"fictional {fictional!r} is a near-spelling of the original ({near[0][0]!r} ~ {near[0][1]!r})")
    return out


# Types whose real-swap substitute should itself have a Wikipedia page; demonyms, languages and
# untyped proper nouns rarely match an article title verbatim.
REAL_CHECK_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT", "LAW"}


class Mapper:
    """Calls the mapping LLM once per paragraph; responses are cached on disk by request hash."""

    def __init__(self, model: str, base_url: str | None, api_key: str | None, cache_path: Path,
                 temperature: float = 0.0, max_tokens: int = 4096, extra_body: dict | None = None,
                 wiki=None, max_retries: int = 5):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", max_retries=0, timeout=300)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.wiki = wiki
        self.cache = {}
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["response"]

    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps([self.model, PROMPT_VERSION, messages, self.temperature])
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, messages: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        key = self._key(messages)
        with self.lock:
            if key in self.cache:
                self.stats["cached"] += 1
                return self.cache[key]
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    max_tokens=self.max_tokens, extra_body=self.extra_body,
                )
                text = resp.choices[0].message.content or ""
                break
            except Exception as e:  # rate limits, timeouts, provider errors: back off and retry
                if attempt == self.max_retries:
                    raise
                print(f"[mapper] {type(e).__name__}: {str(e)[:120]}; retry {attempt + 1}")
                time.sleep(min(60, 2 ** attempt * 2))
        usage = resp.usage.model_dump() if resp.usage else {}
        extra = resp.model_extra or {}
        rec = {"key": key, "model": self.model, "served_model": resp.model,
               "provider": extra.get("provider"), "prompt_version": PROMPT_VERSION,
               "usage": usage, "response": text}
        with self.lock:
            self.cache[key] = text
            self.stats["calls"] += 1
            self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
            self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
            self.stats["cost_usd"] += float(usage.get("cost") or 0.0)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return text

    def map_paragraph(self, paragraph: str, inventory: list[dict], max_attempts: int = 3) -> dict:
        """Request, then up to `max_attempts - 1` repair turns that feed the detected problems back.
        Entities still problematic after the last turn are kept unchanged and marked unresolved."""
        if not inventory:
            return {"mapping": {}, "attempts": 0, "problems": [], "unresolved": [], "raw": ""}
        messages = [{"role": "user", "content": build_user_message(paragraph, inventory)}]
        best = None
        for attempt in range(max_attempts):
            raw = self.complete(messages)
            try:
                parsed = parse_json(raw)
                mapping, problems = validate_mapping(parsed, inventory, self.wiki)
            except (ValueError, AttributeError, KeyError, TypeError) as e:
                mapping, problems = None, [(None, f"output is not the requested JSON object ({e})")]
            if mapping is not None and (best is None or len(problems) <= len(best[1])):
                best = (mapping, problems, raw)
            if not problems:
                break
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": REPAIR_MESSAGE.format(
                             problems="\n".join(f"- {m}" for _, m in problems), n=len(inventory))}]
        if best is None:
            return {"mapping": None, "attempts": attempt + 1, "problems": [m for _, m in problems],
                    "unresolved": [], "raw": raw}
        mapping, problems, raw = best
        unresolved = sorted({i for i, _ in problems if i is not None} | (set(e["id"] for e in inventory) - set(mapping)))
        for inv in inventory:
            if inv["id"] in unresolved:
                m = mapping.setdefault(inv["id"], {"surface": inv["surface"], "type": inv["type"],
                                                   "group": inv.get("group"), "fine_type": None, "same_as": None})
                m.update(keep=True, unresolved=True, fictional=inv["surface"], real_swap=inv["surface"])
        return {"mapping": mapping, "attempts": attempt + 1, "problems": [m for _, m in problems],
                "unresolved": unresolved, "raw": raw}


def parse_json(raw: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])  # ignores trailing text
    return obj


_DET = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _match_determiner(surface: str, sub: str) -> str:
    """Surfaces are stored without a leading article (it stays in the text), so drop one the LLM added."""
    return _DET.sub("", sub) if not _DET.match(surface) else sub


def alias_compatible(a: str, b: str) -> bool:
    """Short form, sub-span or acronym of one another ("Obama"/"Barack Obama", "US"/"United States")."""
    ta, tb = sorted([a.replace(".", "").split(), b.replace(".", "").split()], key=len)
    if [t.lower() for t in ta] == [t.lower() for t in tb]:
        return True
    if any(tb[k:k + len(ta)] == ta for k in range(len(tb) - len(ta) + 1)):
        return True
    return len(ta) == 1 and ta[0].isupper() and ta[0] == "".join(w[0] for w in tb if w[:1].isupper())


def validate_mapping(parsed: dict, inventory: list[dict], wiki=None) -> tuple[dict, list[tuple[str | None, str]]]:
    """Return {id: entry} and (id, message) problems; an empty problem list means fully usable."""
    problems = []
    entries = {e.get("id"): e for e in parsed["entities"] if isinstance(e, dict)}
    for inv in inventory:
        if inv["id"] not in entries:
            problems.append((inv["id"], f"{inv['id']} ({inv['surface']!r}) is missing"))
    originals = {e["surface"] for e in inventory}
    mapping = {}
    for inv in inventory:
        ent = entries.get(inv["id"])
        if ent is None:
            continue
        keep = bool(ent.get("keep"))
        fic = _match_determiner(inv["surface"], str(ent.get("fictional") or inv["surface"]).strip())
        real = _match_determiner(inv["surface"], str(ent.get("real_swap") or inv["surface"]).strip())
        same_as = ent.get("same_as") if ent.get("same_as") in entries else None
        if not keep:
            for policy, sub in (("fictional", fic), ("real_swap", real)):
                if sub == inv["surface"]:
                    problems.append((inv["id"], f"{inv['id']} ({inv['surface']!r}): {policy} equals the original "
                                                "but keep is false"))
                elif sub in originals:
                    problems.append((inv["id"], f"{inv['id']} ({inv['surface']!r}): {policy} {sub!r} is a surface form "
                                                "from the inventory; choose a name not in the paragraph"))
        mapping[inv["id"]] = {
            "surface": inv["surface"], "type": inv["type"], "group": inv.get("group"),
            "fine_type": ent.get("fine_type"), "same_as": same_as, "keep": keep,
            "fictional": inv["surface"] if keep else fic,
            "real_swap": inv["surface"] if keep else real, "source": "llm",
        }
    for i, m in mapping.items():
        if not m["keep"]:
            problems += [(i, f"{i} ({m['surface']!r}): {msg}") for msg in lexical_problems(m["surface"], m["fictional"])]
    if wiki is not None:
        fic = {m["fictional"] for m in mapping.values() if not m["keep"]}
        for i, m in mapping.items():
            if not m["keep"] and m["fictional"] in wiki.real(fic):
                problems.append((i, f"{i} ({m['surface']!r}): fictional {m['fictional']!r} is a real, documented "
                                    "entity; invent a name that does not exist"))
        real = {m["real_swap"] for m in mapping.values() if not m["keep"] and m["type"] in REAL_CHECK_TYPES}
        known = wiki.exists(real)
        for i, m in mapping.items():
            if not m["keep"] and m["type"] in REAL_CHECK_TYPES and m["real_swap"] not in known:
                problems.append((i, f"{i} ({m['surface']!r}): real_swap {m['real_swap']!r} is not a known real "
                                    "entity; pick an existing, well-known one"))
    # Distinctness. Identical substitutes are allowed only for true aliases of one referent; alternative
    # names ("Yuan" / "Dai Ön Ulus") must stay distinct or the passage degenerates.
    def root(i, seen=()):
        j = mapping[i]["same_as"]
        return root(j, seen + (i,)) if j and j in mapping and j not in seen else i
    for policy in ("fictional", "real_swap"):
        by_sub = {}
        for i, m in mapping.items():
            if not m["keep"]:
                by_sub.setdefault(m[policy], []).append(i)
        for sub, ids in by_sub.items():
            ids = sorted(ids, key=lambda x: int(x[1:]))
            for i in ids[1:]:
                j = ids[0]
                ok = root(i) == root(j) and alias_compatible(mapping[i]["surface"], mapping[j]["surface"])
                if not ok:
                    problems.append((i, f"{i} ({mapping[i]['surface']!r}): {policy} {sub!r} is also used for "
                                        f"{j} ({mapping[j]['surface']!r}); different names need different "
                                        "substitutes unless one is a short form or abbreviation of the other"))
    return mapping, problems


def mapping_flags(mapping: dict) -> dict:
    """Soft diagnostics (recorded, not filtered on)."""
    flags = {"length_ratio_out_of_range": [], "alias_inconsistent": []}
    for mid, m in mapping.items():
        if m["keep"] or m["source"] != "llm":
            continue
        for policy in ("fictional", "real_swap"):
            r = len(m[policy]) / max(1, len(m["surface"]))
            if not 0.3 <= r <= 3.0:
                flags["length_ratio_out_of_range"].append(f"{mid}:{policy}")
    # A shorter alias should map to a substring of the longer alias's substitute.
    groups = {}
    for m in mapping.values():
        if m["group"] and not m["keep"]:
            groups.setdefault(m["group"], []).append(m)
    for members in groups.values():
        for a in members:
            for b in members:
                if a is not b and a["surface"] in b["surface"] and a["fictional"] not in b["fictional"]:
                    flags["alias_inconsistent"].append(f"{a['surface']}->{b['surface']}")
    return flags
