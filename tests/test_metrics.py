"""Tests for retrieval evaluation metrics."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mrr,
    ndcg_at_k,
    compute_all_metrics,
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["A", "B"], {"A", "B"}, k=2) == 1.0

    def test_half(self):
        assert precision_at_k(["A", "C"], {"A", "B"}, k=2) == 0.5

    def test_zero(self):
        assert precision_at_k(["C", "D"], {"A", "B"}, k=2) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"A"}, k=5) == 0.0


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["A", "B", "C"], {"A", "B"}, k=3) == 1.0

    def test_partial(self):
        assert recall_at_k(["A", "C"], {"A", "B"}, k=2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["A"], set(), k=1) == 1.0  # vacuously true


class TestReciprocalRank:
    def test_first(self):
        assert reciprocal_rank(["A", "B", "C"], {"A"}) == 1.0

    def test_second(self):
        assert reciprocal_rank(["B", "A", "C"], {"A"}) == 0.5

    def test_not_found(self):
        assert reciprocal_rank(["B", "C"], {"A"}) == 0.0


class TestMRR:
    def test_basic(self):
        pairs = [
            (["A", "B"], {"A"}),  # RR = 1.0
            (["B", "A"], {"A"}),  # RR = 0.5
        ]
        assert mrr(pairs) == 0.75

    def test_empty(self):
        assert mrr([]) == 0.0


class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k(["A"], {"A"}, k=1) == 1.0

    def test_second_position(self):
        result = ndcg_at_k(["B", "A"], {"A"}, k=2)
        assert 0 < result < 1.0  # penalized for not being first

    def test_no_relevant(self):
        assert ndcg_at_k(["A", "B"], set(), k=2) == 0.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        m = compute_all_metrics(["A", "B", "C"], {"A"})
        assert "rr" in m
        assert "p@1" in m
        assert "r@1" in m
        assert "ndcg@1" in m
        assert m["rr"] == 1.0
        assert m["p@1"] == 1.0
