"""Tests for arf.triage — Triage, TriageResult."""

from arf.triage import Triage, TriageResult


def test_classify_basic():
    t = Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70)
    candidates = [
        ({"title": "doc1"}, 0.92),
        ({"title": "doc2"}, 0.78),
        ({"title": "doc3"}, 0.50),
    ]
    result = t.classify(candidates)
    assert len(result.accepted) == 1
    assert result.accepted[0][1] == 0.92
    assert len(result.needs_review) == 1
    assert result.needs_review[0][1] == 0.78
    assert len(result.rejected) == 1


def test_classify_below_min_score():
    t = Triage(min_score=0.65, verify_threshold=0.70, accept_threshold=0.85)
    candidates = [({"title": "low"}, 0.40)]
    result = t.classify(candidates)
    assert len(result.rejected) == 1
    assert len(result.accepted) == 0


def test_classify_between_min_and_verify():
    t = Triage(min_score=0.65, verify_threshold=0.70, accept_threshold=0.85)
    candidates = [({"title": "grey"}, 0.67)]
    result = t.classify(candidates)
    assert len(result.rejected) == 1  # below verify_threshold


def test_by_zones():
    t = Triage()
    items = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    probs = [0.8, 0.5, 0.2]
    result = t.by_zones(items, probs, zones=(0.4, 0.6))
    assert ("a", 0.9) in result.accepted
    assert ("b", 0.5) in result.needs_review
    assert ("c", 0.1) in result.rejected


def test_gap_filter():
    t = Triage(gap=0.10)
    items = [
        ({"title": "top"}, 0.92),
        ({"title": "close"}, 0.85),
        ({"title": "far"}, 0.70),
    ]
    filtered = t.gap_filter(items)
    assert len(filtered) == 2
    assert filtered[0][1] == 0.92
    assert filtered[1][1] == 0.85


def test_gap_filter_empty():
    t = Triage(gap=0.10)
    assert t.gap_filter([]) == []


def test_dedupe():
    list1 = [({"id": "a", "title": "A"}, 0.90)]
    list2 = [({"id": "a", "title": "A"}, 0.95)]
    merged = Triage.dedupe(list1, list2, key_fn=lambda d: d.get("id"))
    assert len(merged) == 1
    assert merged[0][1] == 0.95  # kept the higher score


def test_dedupe_multiple_items():
    list1 = [({"id": "a"}, 0.8), ({"id": "b"}, 0.7)]
    list2 = [({"id": "b"}, 0.9), ({"id": "c"}, 0.6)]
    merged = Triage.dedupe(list1, list2, key_fn=lambda d: d.get("id"))
    assert len(merged) == 3
    # Sorted by score descending
    assert merged[0][0]["id"] == "b"
    assert merged[0][1] == 0.9


def test_apply_full():
    t = Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70, gap=0.20, top_k=10)
    candidates = [
        ({"id": "1"}, 0.92),
        ({"id": "2"}, 0.78),
        ({"id": "3"}, 0.50),
    ]

    def review_fn(items):
        return [(item, score + 0.10) for item, score in items]

    result = t.apply(candidates, review_fn=review_fn, key=lambda c: c[1])
    assert len(result) >= 1
    assert result[0][1] == 0.92  # top accepted


def test_triage_result_default():
    r = TriageResult()
    assert r.accepted == []
    assert r.needs_review == []
    assert r.rejected == []
