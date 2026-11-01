from lexfaith.substitute import answer_mention_ids, make_variant, map_offset, rewrite

CTX = ("Barack Obama was born in Honolulu in 1961. Obama later moved to Chicago, "
       "where Obama's career began in 1985.")
INV = [
    {"id": "E1", "surface": "Barack Obama", "type": "PERSON", "spans": [[0, 12]]},
    {"id": "E2", "surface": "Honolulu", "type": "GPE", "spans": [[25, 33]]},
    {"id": "E3", "surface": "1961", "type": "DATE", "spans": [[37, 41]]},
    {"id": "E4", "surface": "Obama", "type": "PERSON", "spans": [[43, 48], [79, 84]]},
    {"id": "E5", "surface": "Chicago", "type": "GPE", "spans": [[64, 71]]},
    {"id": "E6", "surface": "1985", "type": "DATE", "spans": [[103, 107]]},
]
MAP = {
    "E1": dict(surface="Barack Obama", type="PERSON", group="G1", keep=False, fine_type="politician",
               fictional="Desmond Varnell", real_swap="Angela Merkel"),
    "E2": dict(surface="Honolulu", type="GPE", group=None, keep=False, fine_type="city",
               fictional="Kestrava", real_swap="Lisbon"),
    "E3": dict(surface="1961", type="DATE", group=None, keep=False, fine_type="year",
               fictional="1979", real_swap="1979"),
    "E4": dict(surface="Obama", type="PERSON", group="G1", keep=False, fine_type="politician",
               fictional="Varnell", real_swap="Merkel"),
    "E5": dict(surface="Chicago", type="GPE", group=None, keep=False, fine_type="city",
               fictional="Ordwick", real_swap="Madrid"),
    "E6": dict(surface="1985", type="DATE", group=None, keep=False, fine_type="year",
               fictional="2003", real_swap="2003"),
}


def test_offsets_present():
    for e in INV:
        for s, t in e["spans"]:
            assert CTX[s:t] == e["surface"]


def test_rewrite_longest_first_and_possessive():
    new, edits = rewrite(CTX, {m["surface"]: m["fictional"] for m in MAP.values()})
    assert new.startswith("Desmond Varnell was born in Kestrava in 1979. Varnell later")
    assert "Varnell's career began in 2003" in new
    assert "Obama" not in new and len(edits) == 7


def test_answer_arm_only_changes_answer_group():
    answers = [{"text": "Obama", "answer_start": 43}]
    ids = answer_mention_ids(answers, INV, MAP)
    assert ids == {"E1", "E4"}  # alias group expansion
    v = make_variant(CTX, "Where was Obama born?", answers, MAP, ids, "fictional")
    assert v["question"] == "Where was Varnell born?"
    assert "Honolulu" in v["context"] and "Desmond Varnell" in v["context"]
    a = v["answers"][0]
    assert a["text"] == "Varnell" == a["text_by_string"]
    assert v["context"][a["answer_start"]:].startswith("Varnell later")
    assert not v["residual_originals"]


def test_numeric_answer_offsets_shift():
    answers = [{"text": "1985", "answer_start": 103}]
    ids = set(MAP)
    v = make_variant(CTX, "When did Obama's career begin?", answers, MAP, ids, "real_swap")
    a = v["answers"][0]
    assert a["text"] == "2003" and v["context"][a["answer_start"]:a["answer_start"] + 4] == "2003"
    assert v["question"] == "When did Merkel's career begin?"


def test_answer_starting_inside_mention_is_recovered():
    # gold span "Obama was born in Honolulu" starts inside "Barack Obama"
    answers = [{"text": "Obama was born in Honolulu", "answer_start": 7}]
    v = make_variant(CTX, "q", answers, MAP, set(MAP), "fictional")
    assert v["answers"][0]["text"] == "Varnell was born in Kestrava" == v["answers"][0]["text_by_string"]


def test_answer_fragment_of_unrelated_rewrite_is_inconsistent():
    ctx = "The Islamic Republic rules."
    mp = {"E1": dict(surface="Islamic Republic", group=None, keep=False, fictional="Velorian Republic", real_swap="x")}
    v = make_variant(ctx, "q", [{"text": "Islamic", "answer_start": 4}], mp, {"E1"}, "fictional")
    a = v["answers"][0]
    assert a["text"] != a["text_by_string"]  # rejected by the answer_consistent check


def test_map_offset_identity_without_edits():
    assert map_offset(10, [], "start") == 10


def test_answer_inside_longer_mention():
    ctx = "In early 1961, NASA met. Later in 1961 it decided."
    mp = {"E1": dict(surface="early 1961", group=None, keep=False, fictional="early 1941", real_swap="early 1941"),
          "E2": dict(surface="1961", group=None, keep=False, fictional="1941", real_swap="1941")}
    v = make_variant(ctx, "q", [{"text": "1961", "answer_start": 9}], mp, {"E1", "E2"}, "fictional")
    assert v["answers"][0]["text"] == "1941" == v["answers"][0]["text_by_string"]
    assert v["context"][v["answers"][0]["answer_start"]:].startswith("1941, NASA")


def test_residual_ignores_head_word_inside_substitute():
    ctx = "The Islamic Republic rules. The Republic is old."
    mp = {"E1": dict(surface="Islamic Republic", group=None, keep=False, fictional="Velorian Republic", real_swap="x"),
          "E2": dict(surface="Republic", group=None, keep=False, fictional="Velorian Republic", real_swap="x")}
    v = make_variant(ctx, "q", [{"text": "Islamic Republic", "answer_start": 4}], mp, {"E1", "E2"}, "fictional")
    assert v["residual_originals"] == []


def test_surface_ending_in_symbol_matches_before_letter():
    ctx = "Temperature will increase by about 0.3 °C per decade."
    mp = {"E1": dict(surface="about 0.3 °", group=None, keep=False, fictional="about 0.2 °", real_swap="about 0.2 °")}
    v = make_variant(ctx, "q", [{"text": "about 0.3 °C", "answer_start": 29}], mp, {"E1"}, "fictional")
    assert "about 0.2 °C per decade" in v["context"]
    assert v["answers"][0]["text"] == "about 0.2 °C" == v["answers"][0]["text_by_string"]
