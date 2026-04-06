"""Tests for arf.score_parser — extract_score, multiplier, adjust_score."""

import pytest

from arf.score_parser import adjust_score, adjust_scores, extract_score, multiplier


class TestExtractScore:
    def test_json_object(self):
        assert extract_score('{"score": 7}') == 7

    def test_json_scalar(self):
        assert extract_score("3") == 3

    def test_json_with_reason(self):
        assert extract_score('{"score": 8, "reason": "highly relevant"}') == 8

    def test_score_line(self):
        assert extract_score("Some analysis...\nScore: 5\nMore text") == 5

    def test_bare_number(self):
        assert extract_score("2") == 2

    def test_clamped_high(self):
        assert extract_score("15") == 9

    def test_clamped_low(self):
        assert extract_score("-3") == 0

    def test_rating_key(self):
        assert extract_score('{"rating": 6}') == 6

    def test_relevance_key(self):
        assert extract_score('{"relevance": 4}') == 4

    def test_no_score_raises(self):
        with pytest.raises(ValueError):
            extract_score("This text has no score at all.")


class TestMultiplier:
    def test_zero(self):
        assert multiplier(0) == 0.50

    def test_nine(self):
        assert multiplier(9) == 1.50

    def test_midpoint(self):
        m = multiplier(4.5)
        assert 0.99 < m < 1.01

    def test_custom_range(self):
        assert multiplier(0, min_mult=0.8, max_mult=1.2) == 0.8
        assert multiplier(9, min_mult=0.8, max_mult=1.2) == 1.2


class TestAdjustScore:
    def test_basic(self):
        result = adjust_score(0.72, "Score: 8")
        assert 0.0 <= result <= 1.0
        assert result > 0.72  # score 8 should boost

    def test_low_score_penalizes(self):
        result = adjust_score(0.72, "Score: 1")
        assert result < 0.72

    def test_capped_at_one(self):
        result = adjust_score(0.95, "Score: 9")
        assert result == 1.0

    def test_raises_on_bad_output(self):
        with pytest.raises(ValueError):
            adjust_score(0.5, "no score here")


class TestAdjustScores:
    def test_batch(self):
        candidates = [
            ({"title": "doc1"}, 0.72),
            ({"title": "doc2"}, 0.68),
        ]
        outputs = ["Score: 8", '{"score": 3}']
        results = adjust_scores(candidates, outputs)
        assert len(results) == 2
        assert results[0][1] > 0.72  # boosted
        assert results[1][1] < 0.68  # penalized

    def test_fallback_on_bad_output(self):
        candidates = [({"title": "doc1"}, 0.50)]
        outputs = ["unparseable"]
        results = adjust_scores(candidates, outputs)
        assert results[0][1] == 0.50  # kept original
