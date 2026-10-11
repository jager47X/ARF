"""Tests for arf.evaluation — ranking metrics and evaluate()."""

import math

import pytest

from arf.evaluation import (
    EvalResult,
    evaluate,
    hit_rate_at_k,
    metrics_for,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from arf.pipeline import Pipeline

RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = ["b", "d", "z"]  # z never retrieved


def test_precision_recall_hit_rate():
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == 0.5
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == 0.5
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 10) == pytest.approx(2 / 3)
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 2) == 1.0
    assert recall_at_k(RETRIEVED, [], 5) == 0.0 and precision_at_k(RETRIEVED, RELEVANT, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
    assert reciprocal_rank(RETRIEVED, ["a"]) == 1.0
    assert reciprocal_rank(RETRIEVED, ["nope"]) == 0.0
    assert reciprocal_rank([1, 2], {"2": 1}) == 0.5  # ids are compared as strings


def test_ndcg_binary_and_graded():
    # Binary: hits at ranks 2 and 4 → DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["b", "d", "z"], RELEVANT, 3) == 1.0  # perfect ranking
    # Graded: putting the grade-3 document first beats putting it second.
    graded = {"x": 3, "y": 1}
    assert ndcg_at_k(["x", "y"], graded, 2) == 1.0
    assert ndcg_at_k(["y", "x"], graded, 2) < 1.0
    assert ndcg_at_k(RETRIEVED, {}, 3) == 0.0


def test_metrics_for_keys():
    m = metrics_for(RETRIEVED, RELEVANT, ks=(1, 3))
    assert set(m) == {
        "mrr",
        "precision@1",
        "recall@1",
        "hit_rate@1",
        "ndcg@1",
        "precision@3",
        "recall@3",
        "hit_rate@3",
        "ndcg@3",
    }


def _pipeline():
    index = {
        "q1": [({"_id": "a", "title": "A", "text": "..."}, 0.95), ({"_id": "b", "title": "B", "text": "..."}, 0.90)],
        "q2": [({"_id": "c", "title": "C", "text": "..."}, 0.88), ({"_id": "a", "title": "A", "text": "..."}, 0.86)],
    }
    return Pipeline(search_fn=lambda emb, k: list(index.get(emb, [])), embed_fn=lambda text: text)


def test_evaluate_runs_the_pipeline_and_aggregates():
    seen = []
    dataset = [
        {"query": "q1", "relevant": ["b"]},  # b at rank 2
        {"query": "q2", "relevant": {"c": 2, "zzz": 1}},  # c at rank 1, one relevant never retrieved
        {"query": "q3", "relevant_ids": ["a"]},  # nothing retrieved
    ]
    result = evaluate(_pipeline(), dataset, ks=(1, 2), progress_fn=lambda i, q: seen.append(i))

    assert isinstance(result, EvalResult) and result.n_queries == 3 and seen == [0, 1, 2]
    assert [q.retrieved for q in result.per_query] == [["a", "b"], ["c", "a"], []]
    assert result.metrics["mrr"] == pytest.approx(round((0.5 + 1.0 + 0.0) / 3, 4))
    assert result.metrics["recall@2"] == pytest.approx(round((1.0 + 0.5 + 0.0) / 3, 4))
    assert result.metrics["hit_rate@1"] == pytest.approx(round(1 / 3, 4))
    assert result.stats["runs"] == 3 and result.stats["funnel"]["fetched"] == 4
    assert result.per_query[0].stats["returned"] == 2

    as_dict = result.to_dict()
    assert as_dict["n_queries"] == 3 and as_dict["per_query"][1]["relevant"] == {"c": 2.0, "zzz": 1.0}
    table = result.markdown()
    assert table.startswith("| metric | value |") and "| mrr |" in table and "| queries | 3 |" in table


def test_evaluate_custom_id_fn_and_empty_dataset():
    result = evaluate(_pipeline(), [{"query": "q1", "relevant": ["A"]}], ks=(1,), id_fn=lambda r: r["document"].title)
    assert result.metrics["hit_rate@1"] == 1.0
    empty = evaluate(_pipeline(), [], ks=(1,))
    assert empty.n_queries == 0 and empty.metrics == {} and empty.stats["runs"] == 0
