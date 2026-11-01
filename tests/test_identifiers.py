import random
import re

from lexfaith.corpora import sample_balanced
from lexfaith.identifiers import (add_near_misses, coin_drug, edit_distance, fictional_candidates,
                                  insert_distractor, match_case, near_miss_candidates, occurrences,
                                  rewrite_id, version_swaps)
from lexfaith.metrics import score_item
from lexfaith.substitute import make_variant, map_offset


def test_bounded_occurrences():
    text = "Fixed in 1.2.3; 1.2 is affected. Upgrade to 1.2."
    assert occurrences(text, "1.2") == [(16, 19), (44, 47)]  # not inside 1.2.3; sentence-final dot ok
    assert occurrences("CWE-79 and CWE-798", "CWE-79") == [(0, 6)]
    assert occurrences("hRPE65v2 and RPE65", "RPE65") == [(13, 18)]


def test_rewrite_id_case_and_offsets():
    text = "XARELTO is indicated. Take Xarelto daily."
    new, edits = rewrite_id(text, "Xarelto", "Velomarex", ignore_case=True)
    assert new == "VELOMAREX is indicated. Take Velomarex daily."
    assert new[map_offset(27, edits, "start"):map_offset(34, edits, "end")] == "Velomarex"
    assert match_case("velomarex", "Xarelto") == "Velomarex" and match_case("PD-L1x", "PD-L1") == "PD-L1X"


def test_near_miss_is_one_edit():
    rng = random.Random(0)
    for s, kind in [("3.11.4", "version"), ("CVE-2021-44228", "cve"), ("CWE-79", "cwe"), ("metoprolol", "drug"),
                    ("TBC1D7", "gene"), ("BRG1", "gene"), ("suricata-update", "package"), ("Obama", "name")]:
        cands = near_miss_candidates(s, kind, rng)
        assert cands, s
        assert all(edit_distance(c, s) == 1 for c in cands)
    assert near_miss_candidates("3.11.4", "version", rng)[0] == "3.11.5"


def test_fictional_is_well_formed_and_far():
    rng = random.Random(1)
    cases = {
        "cve": ("CVE-2021-44228", r"CVE-2021-\d{5}"),
        "gene": ("TBC1D7", r"[A-Z]{3}\d[A-Z]\d"),
        "drug": ("tofacitinib", r"[a-z]+tinib"),
    }
    for kind, (s, pat) in cases.items():
        cands = fictional_candidates(s, kind, rng)
        assert cands, kind
        for c in cands:
            assert re.fullmatch(pat, c), (kind, c)
            assert edit_distance(c.lower(), s.lower()) >= 2
    cwes = fictional_candidates("CWE-79", "cwe", rng, unassigned=[3, 13, 1171])
    assert cwes and set(cwes) <= {"CWE-3", "CWE-13", "CWE-1171"}
    vs = fictional_candidates("1.7.2", "version", rng, releases=["1.7.2", "1.7.3", "1.8.0"])
    assert vs and all(v.startswith("1.") and v not in ("1.7.3", "1.8.0") for v in vs)
    assert coin_drug("XARELTO", rng).isupper()


def test_version_swaps_prefer_same_major():
    rels = ["0.9.1", "1.6.1", "1.7.2", "1.7.3", "2.0.0", "1.7"]
    out = version_swaps("1.7.2", rels, random.Random(0))
    assert "1.7.2" not in out and "1.7" not in out
    assert out[0] == "1.6.1"  # same major and not a one-character neighbour


def test_distractor_insertion_keeps_answer_offset():
    ctx = "CherryPy before 2.1.1 allows traversal. It was reported in 2006."
    new, start = insert_distractor(ctx, "2.1.1", 16, "2.1.2", "version")
    assert start == 16 and new[16:21] == "2.1.1"
    assert "Version 2.1.1 should not be confused with version 2.1.2." in new
    assert new.endswith("It was reported in 2006.")
    # cropped passage without a sentence end after the answer: insert before the answer's sentence
    ctx = "Adalimumab binds it. The rate of TNF permeation versus the rate of"
    new, start = insert_distractor(ctx, "TNF", 33, "TMF", "gene")
    assert new == "Adalimumab binds it. TNF should not be confused with TMF. The rate of TNF permeation versus the rate of"
    assert new[start:start + 3] == "TNF" and start == 33 + len("TNF should not be confused with TMF. ")


def test_squad_near_miss_mapping_shares_alias_edit():
    mapping = {
        "E1": dict(surface="Barack Obama", group="G1", keep=False, source="llm"),
        "E4": dict(surface="Obama", group="G1", keep=False, source="llm"),
        "E3": dict(surface="1961", group=None, keep=False, source="rule"),
        "E9": dict(surface="Honolulu", group=None, keep=True, source="llm"),
    }
    add_near_misses(mapping, random.Random(0))
    w = mapping["E4"]["near_miss"]
    assert edit_distance(w, "Obama") == 1
    assert mapping["E1"]["near_miss"] == "Barack " + w
    assert mapping["E3"]["near_miss"] == "1962"
    assert "near_miss" not in mapping["E9"]
    ctx = "Barack Obama was born in 1961. Obama moved."
    v = make_variant(ctx, "When was Obama born?", [{"text": "1961", "answer_start": 25}],
                     {k: {**m, "fictional": m["surface"], "real_swap": m["surface"]} for k, m in mapping.items()},
                     {"E3"}, "near_miss")
    assert v["answers"][0]["text"] == "1962" and v["question"] == "When was Obama born?"


def test_distractor_category():
    item = {"answers": ["1.7.2"], "orig_answers": ["1.7.2"], "answer_changed": False,
            "distractor": "1.7.3", "context": "fixed in 1.7.2. 1.7.2 should not be confused with 1.7.3."}
    assert score_item(item, "1.7.3")["category"] == "distractor"
    assert score_item(item, "Version 1.7.2")["category"] == "correct"


def test_sample_balanced_one_question_per_record():
    seeds = [{"para_id": f"r{i // 2}", "kind": k, "source": "a"} for i, k in enumerate(["x", "y"] * 10)]
    out = sample_balanced(seeds, 6, 0, key=lambda s: (s["source"], s["kind"]))
    assert len(out) == 6 and len({s["para_id"] for s in out}) == 6
    assert {s["kind"] for s in out} == {"x", "y"}
