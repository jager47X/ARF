"""Tests for arf.pipeline — Pipeline end-to-end with mocks."""

import logging

import pytest

from arf.pipeline import Pipeline
from arf.triage import Triage


def _mock_embed(text):
    return [0.1] * 10


def _mock_search(embedding, top_k):
    return [
        ({"_id": "1", "title": "14th Amendment", "text": "All persons born..."}, 0.92),
        ({"_id": "2", "title": "5th Amendment", "text": "No person shall..."}, 0.78),
        ({"_id": "3", "title": "Random doc", "text": "Unrelated content."}, 0.50),
    ]


def test_minimal_pipeline():
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed)
    results = pipeline.run("what is due process")
    assert isinstance(results, list)


def test_pipeline_with_triage():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        triage=Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70),
    )
    results = pipeline.run("what is due process")
    assert len(results) >= 1
    assert results[0]["score"] >= 0.85


def test_pipeline_with_predict_fn():
    def mock_predict(vectors):
        return [0.9, 0.5, 0.1]  # first high, second uncertain, third low

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=mock_predict,
        triage=Triage(min_score=0.50, accept_threshold=0.85, verify_threshold=0.70),
    )
    results = pipeline.run("what is due process")
    assert isinstance(results, list)


def test_pipeline_with_cache_hit():
    cached = [
        ({"title": "Cached Result", "text": "From cache."}, 0.95),
    ]

    def mock_lookup(query):
        return {"results": cached}

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        cache_lookup=mock_lookup,
    )
    results = pipeline.run("cached query")
    assert len(results) == 1
    assert results[0]["document"].title == "Cached Result"


def test_pipeline_moderation_blocks():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        moderate_fn=lambda q: False,
    )
    results = pipeline.run("blocked query")
    assert results == []


def test_pipeline_preprocess():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        preprocess_fn=lambda q: q.upper(),
    )
    results = pipeline.run("what is due process")
    assert isinstance(results, list)


def test_pipeline_with_summarize():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        summarize_fn=lambda q, doc, ctx: f"Summary for {doc.title}",
    )
    results = pipeline.run("what is due process")
    assert any(r.get("summary") for r in results)


def test_pipeline_empty_search():
    pipeline = Pipeline(
        search_fn=lambda emb, k: [],
        embed_fn=_mock_embed,
    )
    results = pipeline.run("nothing here")
    assert results == []


# ---------------------------------------------------------------------------
# Unreviewed "needs review" band (no predict_fn / llm_fn)
# ---------------------------------------------------------------------------


def test_unreviewed_candidates_kept_by_default():
    # 0.92 → accepted, 0.78 → needs_review, 0.50 → rejected.  With no reviewer
    # configured the 0.78 doc must still come back, below the 0.92 one.
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed)
    results = pipeline.run("what is due process")
    assert [r["document"].id for r in results] == ["1", "2"]
    assert results[1]["score"] == 0.78


def test_unreviewed_policy_drop_restores_old_behaviour():
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, unreviewed_policy="drop")
    results = pipeline.run("what is due process")
    assert [r["document"].id for r in results] == ["1"]


def test_unreviewed_policy_applies_to_mlp_uncertain():
    # The single needs_review candidate gets an uncertain probability and
    # there is no llm_fn to resolve it → policy decides.
    kept = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=lambda vectors: [0.5] * len(vectors),
    )
    assert [r["document"].id for r in kept.run("q")] == ["1", "2"]

    dropped = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=lambda vectors: [0.5] * len(vectors),
        unreviewed_policy="drop",
    )
    assert [r["document"].id for r in dropped.run("q")] == ["1"]


def test_mlp_rejected_candidates_are_not_kept():
    # A confident MLP reject is a decision, not an unreviewed candidate.
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=lambda vectors: [0.1] * len(vectors),
    )
    assert [r["document"].id for r in pipeline.run("q")] == ["1"]


def test_invalid_unreviewed_policy_raises():
    with pytest.raises(ValueError, match="unreviewed_policy"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, unreviewed_policy="maybe")


def test_all_candidates_rejected_warns_once(caplog):
    # Scores in the 0.3–0.5 range (typical for some embedding models) with the
    # default 0.65 min_score → everything rejected → one calibration warning.
    def low_score_search(embedding, top_k):
        return [({"_id": "a", "title": "A", "text": "..."}, 0.48), ({"_id": "b", "title": "B", "text": "..."}, 0.31)]

    pipeline = Pipeline(search_fn=low_score_search, embed_fn=_mock_embed)
    with caplog.at_level(logging.WARNING, logger="arf.pipeline"):
        assert pipeline.run("q1") == []
        assert pipeline.run("q2") == []

    warnings = [r for r in caplog.records if "rejected all" in r.getMessage()]
    assert len(warnings) == 1
    assert "0.480" in warnings[0].getMessage()
    assert "min_score=0.65" in warnings[0].getMessage()
