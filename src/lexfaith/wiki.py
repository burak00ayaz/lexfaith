"""Does a name denote a real, documented entity? (English Wikipedia, cached on disk)

A name counts as real if it resolves (after redirects) to a non-disambiguation article of at least
`min_bytes`. The size threshold keeps invented single-word names from failing on stub or
surname pages while catching established entities such as "Cairo" or "Springfield".
"""

import json
import threading
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"


class WikiChecker:
    def __init__(self, cache_path: Path, min_bytes: int = 3000, enabled: bool = True):
        self.cache_path = cache_path
        self.min_bytes = min_bytes
        self.enabled = enabled
        self.cache: dict[str, dict] = {}
        self.lock = threading.Lock()
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "lexfaith-benchmark/0.1 (academic research)"

    def lookup(self, names) -> dict[str, dict]:
        with self.lock:  # shared across mapping threads
            names = sorted({n for n in names if n and n not in self.cache})
            if self.enabled:
                for i in range(0, len(names), 50):
                    self._fetch(names[i:i + 50])
                if names:
                    self.cache_path.write_text(json.dumps(self.cache))
            return self.cache

    def real(self, names) -> set[str]:
        info = self.lookup(names)
        return {n for n in names if (r := info.get(n)) and r["exists"] and not r["disambiguation"]
                and r["length"] >= self.min_bytes}

    def exists(self, names) -> set[str]:
        """Any page, including disambiguation pages (enough to show a real-swap name is real)."""
        info = self.lookup(names)
        return {n for n in names if (r := info.get(n)) and r["exists"]}

    def _fetch(self, batch: list[str]):
        for attempt in range(3):
            try:
                r = self.session.get(API, params={
                    "action": "query", "titles": "|".join(batch), "redirects": 1, "format": "json",
                    "prop": "info|pageprops", "ppprop": "disambiguation"}, timeout=30)
                q = r.json().get("query", {})
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        else:
            return
        alias = {}
        for kind in ("normalized", "redirects"):
            for m in q.get(kind, []):
                alias[m["to"]] = alias.get(m["from"], m["from"])
        resolved = {}
        for page in q.get("pages", {}).values():
            t = page.get("title")
            src = t
            while src in alias:
                src = alias[src]
            resolved[src] = {
                "exists": "missing" not in page and "invalid" not in page,
                "title": t,
                "disambiguation": "disambiguation" in page.get("pageprops", {}),
                "length": page.get("length", 0),
            }
        for n in batch:
            self.cache[n] = resolved.get(n, {"exists": False, "title": None, "disambiguation": False, "length": 0})
        time.sleep(0.2)
