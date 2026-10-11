"""Tests for arf.pipeline — Pipeline end-to-end with mocks."""

import logging

import pytest

from arf.document import DocumentConfig
from arf.features import FEATURE_NAMES
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


# ---------------------------------------------------------------------------
# Keyword hits merged on the semantic scale
# ---------------------------------------------------------------------------


def _scores_by_id(results):
    return {r["document"].id: round(r["score"], 4) for r in results}


def test_keyword_hit_also_in_vector_results_keeps_cosine_plus_boost():
    # BM25-style score (12.5) must NOT replace the 0.78 cosine score.
    def keyword_fn(query):
        return [({"_id": "2", "title": "5th Amendment", "text": "No person shall..."}, 12.5)]

    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn)
    scores = _scores_by_id(pipeline.run("due process"))
    assert scores["2"] == 0.83  # 0.78 + 0.05 boost, still in the review band


def test_keyword_only_hit_is_rescored_by_cosine_against_its_embedding():
    # Identical embedding → cosine 1.0 → capped at 1.0 → accepted and ranked first.
    def keyword_fn(query):
        return [({"_id": "kw", "title": "Keyword doc", "text": "...", "embedding": [0.1] * 10}, 3.0)]

    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn)
    results = pipeline.run("due process")
    assert results[0]["document"].id == "kw"
    assert results[0]["score"] == 1.0


def test_keyword_only_hit_without_embedding_goes_to_review():
    def keyword_fn(query):
        return [({"_id": "kw", "title": "Keyword doc", "text": "..."}, 99.0)]

    # Default floor = verify_threshold (0.70) + 0.05 boost → 0.75 → "needs review".
    kept = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn)
    assert _scores_by_id(kept.run("q"))["kw"] == 0.75

    # ...which is exactly what the unreviewed policy governs, so "drop" removes it.
    dropped = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn, unreviewed_policy="drop")
    assert "kw" not in _scores_by_id(dropped.run("q"))


def test_keyword_floor_and_boost_are_configurable():
    def keyword_fn(query):
        return [({"_id": "kw", "title": "Keyword doc", "text": "..."}, 99.0)]

    pipeline = Pipeline(
        search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn, keyword_floor=0.90, keyword_boost=0.0
    )
    scores = _scores_by_id(pipeline.run("q"))
    assert scores["kw"] == 0.90  # accepted outright, no boost applied
    assert scores["2"] == 0.78


def test_keyword_rescore_honours_custom_embedding_field():
    config = DocumentConfig(embedding_field="vec")

    def keyword_fn(query):
        return [({"_id": "kw", "title": "Keyword doc", "text": "...", "vec": [0.1] * 10}, 1.0)]

    pipeline = Pipeline(doc_config=config, search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn)
    assert _scores_by_id(pipeline.run("q"))["kw"] == 1.0


def test_keyword_mode_max_merges_raw_scores():
    # Legacy behaviour: an arbitrary 0.99 "keyword score" is treated as if it were cosine.
    def keyword_fn(query):
        return [({"_id": "kw", "title": "Keyword doc", "text": "..."}, 0.99)]

    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, keyword_fn=keyword_fn, keyword_mode="max")
    results = pipeline.run("q")
    assert results[0]["document"].id == "kw"
    assert results[0]["score"] == 0.99


def test_invalid_keyword_mode_raises():
    with pytest.raises(ValueError, match="keyword_mode"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, keyword_mode="rrf")


# ---------------------------------------------------------------------------
# MLP sees the same features it was trained on
# ---------------------------------------------------------------------------


def _feature(vector, name):
    return vector[FEATURE_NAMES.index(name)]


def test_mlp_receives_keyword_alias_and_global_gap_features():
    seen = []

    def predict_fn(vectors):
        seen.extend(vectors)
        return [0.9] * len(vectors)

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=predict_fn,
        keyword_fn=lambda q: [({"_id": "2", "title": "5th Amendment", "text": "No person shall..."}, 7.0)],
        alias_fn=lambda q: [("fifth", "5th Amendment")],
    )
    pipeline.run("fifth amendment")

    assert len(seen) == 1  # only the 0.78 (+0.05) doc is in the review band
    vec = seen[0]
    assert _feature(vec, "keyword_match") == 1
    assert _feature(vec, "alias_match") == 1
    assert _feature(vec, "match_type") == 2
    # Gap is measured against the query's overall top (0.92), not the review band's own top.
    assert round(_feature(vec, "score_gap_from_top"), 4) == round(0.92 - 0.83, 4)


def test_mlp_gap_feature_without_keyword_fn_uses_global_top():
    seen = []

    def predict_fn(vectors):
        seen.extend(vectors)
        return [0.9] * len(vectors)

    Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, predict_fn=predict_fn).run("q")
    assert round(_feature(seen[0], "score_gap_from_top"), 4) == 0.14
    assert _feature(seen[0], "keyword_match") == 0
    assert _feature(seen[0], "alias_match") == 0


def test_alias_fn_failure_is_logged_and_ignored(caplog):
    def broken_alias_fn(query):
        raise RuntimeError("alias index down")

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=lambda vectors: [0.9] * len(vectors),
        alias_fn=broken_alias_fn,
    )
    with caplog.at_level(logging.WARNING, logger="arf.pipeline"):
        results = pipeline.run("q")
    assert [r["document"].id for r in results] == ["1", "2"]
    assert any("alias_fn failed" in r.getMessage() for r in caplog.records)
