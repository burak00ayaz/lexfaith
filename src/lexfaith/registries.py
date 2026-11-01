"""Authoritative registries used to decide whether an identifier is real (cached on disk).

  drugs     RxNorm (NLM RxNav REST): ingredient / brand names, ATC class of an ingredient
  genes     HGNC complete set (approved, alias and previous symbols)
  packages  PyPI JSON API: package existence and its release list
  CVEs      CVE Services (cveawg.mitre.org): a published or reserved record exists
  CWEs      MITRE CWE catalogue (weakness ids)

Bulk lists (HGNC, CWE) are downloaded once into `sources/`; per-name lookups are cached in
`cache.json` so rebuilding a dataset does not repeat network calls.
"""

import json
import re
import threading
import time
import zipfile
from pathlib import Path

import requests

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
RXNAV = "https://rxnav.nlm.nih.gov/REST"


class Registries:
    def __init__(self, sources: Path, cache_path: Path):
        self.sources = Path(sources)
        self.sources.mkdir(parents=True, exist_ok=True)
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "lexfaith-benchmark/0.1 (academic research)"
        self._genes = None
        self._cwes = None

    # ---- cached HTTP -------------------------------------------------------------------------
    def _get(self, ns: str, key: str, url: str, params=None, parse=lambda r: r.json(), delay=0.05):
        ck = f"{ns}:{key}"
        with self.lock:
            if ck in self.cache:
                return self.cache[ck]
        val = None
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 404:
                    val = None
                    break
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                val = parse(r)
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"registry lookup failed: {url}")
        time.sleep(delay)
        with self.lock:
            self.cache[ck] = val
        return val

    def save(self):
        with self.lock:
            self.cache_path.write_text(json.dumps(self.cache))

    def _download(self, url: str, name: str) -> Path:
        path = self.sources / name
        if not path.exists():
            r = self.session.get(url, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        return path

    # ---- drugs (RxNorm) ----------------------------------------------------------------------
    def drug_rxcui(self, name: str) -> str | None:
        """RxCUI of an exact (normalized) RxNorm name, or None."""
        ids = self._get("rxcui", name.lower(), f"{RXNAV}/rxcui.json", {"name": name, "search": 2},
                        lambda r: r.json().get("idGroup", {}).get("rxnormId"))
        return ids[0] if ids else None

    def drug_tty(self, rxcui: str) -> str | None:
        return self._get("tty", rxcui, f"{RXNAV}/rxcui/{rxcui}/property.json", {"propName": "TTY"},
                         lambda r: (r.json().get("propConceptGroup") or {}).get("propConcept", [{}])[0]
                         .get("propValue"))

    def is_drug(self, name: str) -> bool:
        """Single-ingredient or brand name (IN, PIN, BN), not a dose form or clinical drug."""
        cui = self.drug_rxcui(name)
        return bool(cui) and self.drug_tty(cui) in ("IN", "PIN", "BN", "MIN")

    def drug_exists(self, name: str) -> bool:
        """Any RxNorm concept, or an approximate match at the top score (catches spelling variants)."""
        if self.drug_rxcui(name):
            return True
        cands = self._get("approx", name.lower(), f"{RXNAV}/approximateTerm.json",
                          {"term": name, "maxEntries": 3, "option": 1},
                          lambda r: [c.get("name") for c in r.json().get("approximateGroup", {})
                                     .get("candidate", []) if c.get("name")])
        return any(c and c.lower() == name.lower() for c in cands or [])

    def atc_class(self, name: str) -> list[str]:
        """ATC level-4 class ids of an ingredient (e.g. C07AB for metoprolol)."""
        return self._get("atc", name.lower(), f"{RXNAV}/rxclass/class/byDrugName.json",
                         {"drugName": name, "relaSource": "ATC"},
                         lambda r: sorted({i["rxclassMinConceptItem"]["classId"] for i in
                                           r.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
                                           if i["minConcept"]["tty"] == "IN"})) or []

    def atc_members(self, class_id: str) -> list[str]:
        return self._get("atcmem", class_id, f"{RXNAV}/rxclass/classMembers.json",
                         {"classId": class_id, "relaSource": "ATC", "ttys": "IN"},
                         lambda r: sorted({m["minConcept"]["name"] for m in
                                           r.json().get("drugMemberGroup", {}).get("drugMember", [])})) or []

    # ---- genes (HGNC) ------------------------------------------------------------------------
    def genes(self) -> dict[str, str]:
        """Every approved / alias / previous HGNC symbol -> its approved symbol."""
        if self._genes is None:
            path = self._download(HGNC_URL, "hgnc_complete_set.txt")
            lines = path.read_text().splitlines()
            head = lines[0].split("\t")
            col = {c: head.index(c) for c in ("symbol", "alias_symbol", "prev_symbol", "status")}
            out = {}
            for line in lines[1:]:
                f = line.split("\t")
                if f[col["status"]] != "Approved":
                    continue
                sym = f[col["symbol"]]
                out.setdefault(sym, sym)
                for c in ("alias_symbol", "prev_symbol"):
                    for a in f[col[c]].strip('"').split("|"):
                        if a:
                            out.setdefault(a, sym)
            self._genes = out
        return self._genes

    def gene_exists(self, symbol: str) -> bool:
        g = self.genes()
        return symbol in g or symbol.upper() in g

    # ---- software ----------------------------------------------------------------------------
    def pypi_releases(self, package: str) -> list[str] | None:
        """Release versions of a PyPI package (None if the package does not exist)."""
        return self._get("pypi", package.lower(), f"https://pypi.org/pypi/{package}/json", None,
                         lambda r: sorted(r.json().get("releases", {})))

    def package_exists(self, package: str) -> bool:
        return self.pypi_releases(package) is not None

    def cve_exists(self, cve_id: str) -> bool:
        """A CVE record (published, rejected or reserved) exists for this id."""
        state = self._get("cve", cve_id, f"https://cveawg.mitre.org/api/cve/{cve_id}", None,
                          lambda r: r.json().get("cveMetadata", {}).get("state", "UNKNOWN"))
        return state is not None

    def cwes(self) -> set[str]:
        if self._cwes is None:
            path = self._download(CWE_URL, "cwec_latest.xml.zip")
            with zipfile.ZipFile(path) as z:
                xml = z.read(z.namelist()[0]).decode()
            # Weaknesses, categories and views share one id space.
            self._cwes = {f"CWE-{i}" for i in re.findall(r'<(?:Weakness|Category|View) ID="(\d+)"', xml)}
        return self._cwes

    def cwe_exists(self, cwe_id: str) -> bool:
        return cwe_id in self.cwes()

