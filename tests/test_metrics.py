from lexfaith.metrics import diff_in_diff, score_item

ITEM = {"answers": ["Desmond Varnell"], "orig_answers": ["Barack Obama"], "answer_changed": True,
        "context": "Desmond Varnell met Angela Ruiz in Kestrava."}


def test_categories():
    assert score_item(ITEM, "It was Desmond Varnell.")["category"] == "correct"
    assert score_item(ITEM, "Barack Obama")["category"] == "parametric"
    assert score_item(ITEM, "Desmond Varnel")["category"] == "near_miss"
    assert score_item(ITEM, "Angela Ruiz")["category"] == "context"
    assert score_item(ITEM, "Paris")["category"] == "other"


def test_did_sign():
    rows = []
    for q, para in [("q1", "p1"), ("q2", "p2")]:
        rows += [
            {"system": "comp", "arm": "original", "qid": q, "para_id": para, "match": 1.0},
            {"system": "comp", "arm": "fictional_all", "qid": q, "para_id": para, "match": 0.0},
            {"system": "ref", "arm": "original", "qid": q, "para_id": para, "match": 1.0},
            {"system": "ref", "arm": "fictional_all", "qid": q, "para_id": para, "match": 1.0},
        ]
    d = diff_in_diff(rows, "comp", "ref", "fictional_all")
    assert d["n"] == 2 and d["did"] == 1.0  # positive = extra drop under compression
