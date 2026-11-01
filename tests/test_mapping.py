import json
import os

from lexfaith.build import load_env_file
from lexfaith.mapping import Mapper, validate_mapping

INV = [
    {"id": "E1", "surface": "Barack Obama", "type": "PERSON", "group": "G1"},
    {"id": "E2", "surface": "Obama", "type": "PERSON", "group": "G1"},
    {"id": "E3", "surface": "Canada", "type": "GPE", "group": None},
]


def _resp(entities):
    return json.dumps({"entities": entities})


def _ent(i, fic, real, same_as=None, keep=False):
    return {"id": i, "fine_type": "x", "same_as": same_as, "keep": keep, "fictional": fic, "real_swap": real}


def test_env_file_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nLEXFAITH_A='one'\nexport LEXFAITH_B=two\nLEXFAITH_C=three\n")
    monkeypatch.setenv("LEXFAITH_C", "kept")
    monkeypatch.delenv("LEXFAITH_A", raising=False)
    monkeypatch.delenv("LEXFAITH_B", raising=False)
    assert sorted(load_env_file(f)) == ["LEXFAITH_A", "LEXFAITH_B"]
    assert (os.environ["LEXFAITH_A"], os.environ["LEXFAITH_B"], os.environ["LEXFAITH_C"]) == ("one", "two", "kept")
    assert load_env_file(tmp_path / "missing") == []


def test_validation_flags_near_spelling_and_allows_alias_sharing():
    parsed = {"entities": [_ent("E1", "Desmond Varnell", "Angela Merkel"),
                           _ent("E2", "Varnell", "Merkel", same_as="E1"),
                           _ent("E3", "Canadia", "Japan")]}
    _, problems = validate_mapping(parsed, INV)
    ids = {i for i, _ in problems}
    assert ids == {"E3"}  # near-spelling; the alias pair is fine


class StubMapper(Mapper):
    """Replays canned responses instead of calling an API."""

    def __init__(self, responses, tmp_path):
        super().__init__("stub", None, "x", tmp_path / "cache.jsonl")
        self.responses = list(responses)
        self.seen = []

    def complete(self, messages):
        self.seen.append(messages)
        return self.responses.pop(0)


def test_repair_turn_fixes_problem(tmp_path):
    bad = _resp([_ent("E1", "Desmond Varnell", "Angela Merkel"), _ent("E2", "Varnell", "Merkel", "E1"),
                 _ent("E3", "Canadia", "Japan")])
    good = _resp([_ent("E1", "Desmond Varnell", "Angela Merkel"), _ent("E2", "Varnell", "Merkel", "E1"),
                  _ent("E3", "Ostrevia", "Japan")])
    m = StubMapper([bad, good], tmp_path)
    out = m.map_paragraph("Barack Obama ... Obama ... Canada", INV)
    assert out["attempts"] == 2 and out["unresolved"] == []
    assert out["mapping"]["E3"]["fictional"] == "Ostrevia"
    assert "near-spelling" in m.seen[1][-1]["content"]  # problems were fed back


def test_unfixable_mention_is_kept_and_marked(tmp_path):
    bad = _resp([_ent("E1", "Desmond Varnell", "Angela Merkel"), _ent("E2", "Varnell", "Merkel", "E1"),
                 _ent("E3", "Canadia", "Japan")])
    m = StubMapper([bad] * 3, tmp_path)
    out = m.map_paragraph("p", INV)
    assert out["unresolved"] == ["E3"]
    e3 = out["mapping"]["E3"]
    assert e3["keep"] and e3["unresolved"] and e3["fictional"] == "Canada"
