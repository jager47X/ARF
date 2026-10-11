"""Tests for arf.pipeline — Pipeline end-to-end with mocks."""

import logging
import threading

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


def test_dedupe_uses_configured_id_field():
    # Two distinct documents sharing a title must not collapse into one, and a
    # keyword hit with the same configured id must merge with its vector twin.
    config = DocumentConfig(id_field="pk")

    def search_fn(embedding, top_k):
        return [
            ({"pk": "a", "title": "Same Title", "text": "first"}, 0.90),
            ({"pk": "b", "title": "Same Title", "text": "second"}, 0.88),
        ]

    pipeline = Pipeline(
        doc_config=config,
        search_fn=search_fn,
        embed_fn=_mock_embed,
        keyword_fn=lambda q: [({"pk": "a", "title": "Same Title", "text": "first"}, 5.0)],
    )
    results = pipeline.run("same title")
    assert [r["document"].id for r in results] == ["a", "b"]
    assert round(results[0]["score"], 4) == 0.95  # 0.90 + keyword boost


# ---------------------------------------------------------------------------
# LLM verification: gate (default) vs blend
# ---------------------------------------------------------------------------


def _llm_saying(score_by_id):
    """Fake llm_fn: answers 'Score: N' per document id."""

    def llm_fn(query, item):
        return f"Reasoning: ...\nScore: {score_by_id[item['_id']]}"

    return llm_fn


def test_llm_gate_keeps_retrieval_score_and_reports_llm_score():
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_llm_saying({"2": 8}))
    results = pipeline.run("q")
    by_id = {r["document"].id: r for r in results}
    assert set(by_id) == {"1", "2"}
    assert by_id["2"]["score"] == 0.78  # retrieval score, not 0.78 * 1.39
    assert by_id["2"]["llm_score"] == 8
    assert by_id["1"]["llm_score"] is None  # accepted directly, never saw the LLM


def test_llm_gate_pass_score_is_configurable():
    dropped = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_llm_saying({"2": 5}))
    assert [r["document"].id for r in dropped.run("q")] == ["1"]

    kept = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_llm_saying({"2": 5}), llm_pass_score=5)
    assert [r["document"].id for r in kept.run("q")] == ["1", "2"]


def test_llm_gate_after_uncertain_mlp():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=lambda vectors: [0.5] * len(vectors),
        llm_fn=_llm_saying({"2": 7}),
    )
    results = pipeline.run("q")
    assert [(r["document"].id, r["score"]) for r in results] == [("1", 0.92), ("2", 0.78)]


def test_llm_blend_mode_is_legacy_multiplicative():
    def run_with(score):
        pipeline = Pipeline(
            search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_llm_saying({"2": score}), llm_mode="blend"
        )
        return {r["document"].id: round(r["score"], 4) for r in pipeline.run("q")}

    assert run_with(9)["2"] == 1.0  # 0.78 * 1.5 capped
    assert run_with(6)["2"] == round(0.78 * (0.5 + 6 / 9), 4)  # 0.91 ≥ accept
    assert "2" not in run_with(4)  # 0.78 * 0.94 < 0.85


def test_gap_filter_uses_retrieval_scores_even_when_blending():
    # A: 0.86 accepted outright; B: 0.72 in the review band, LLM says 9/9.
    def search_fn(embedding, top_k):
        return [({"_id": "A", "title": "A", "text": "..."}, 0.86), ({"_id": "B", "title": "B", "text": "..."}, 0.72)]

    def make(gap):
        return Pipeline(
            search_fn=search_fn,
            embed_fn=_mock_embed,
            llm_fn=_llm_saying({"B": 9}),
            llm_mode="blend",
            triage=Triage(gap=gap),
        )

    # Blended B becomes 1.0.  Before 0.3.0 the gap filter used that 1.0 as the
    # top and evicted A (1.0 - 0.86 > 0.10).  Now the gap is measured on
    # retrieval scores: top is A at 0.86 and B (0.72) is the one too far away.
    assert [r["document"].id for r in make(0.10).run("q")] == ["A"]
    # With the default-sized gap both survive and B still reports its blended score.
    results = make(0.20).run("q")
    assert {r["document"].id: r["score"] for r in results} == {"A": 0.86, "B": 1.0}


def _four_uncertain_docs(embedding, top_k):
    # All four land in the 0.70–0.85 review band, so all four reach the LLM.
    return [({"_id": str(i), "title": f"Doc {i}", "text": "..."}, 0.80 - i * 0.01) for i in range(4)]


def test_llm_verification_runs_in_parallel_and_preserves_order():
    # Every llm_fn call waits at a barrier that only opens once all four calls
    # are in flight — impossible with sequential calls, so passing proves
    # concurrency.  Scores are derived from the id to check ordering.
    barrier = threading.Barrier(4, timeout=2)

    def llm_fn(query, item):
        barrier.wait()
        return f"Score: {9 - int(item['_id'])}"

    pipeline = Pipeline(
        search_fn=_four_uncertain_docs, embed_fn=_mock_embed, llm_fn=llm_fn, llm_workers=4, llm_error_policy="drop"
    )
    results = pipeline.run("q")
    assert [r["document"].id for r in results] == ["0", "1", "2", "3"]
    assert [r["llm_score"] for r in results] == [9, 8, 7, 6]


def test_llm_verification_is_sequential_by_default():
    barrier = threading.Barrier(2, timeout=0.2)
    errors = []

    def llm_fn(query, item):
        barrier.wait()  # never satisfied sequentially → BrokenBarrierError per call
        return "Score: 9"

    pipeline = Pipeline(
        search_fn=_four_uncertain_docs,
        embed_fn=_mock_embed,
        llm_fn=llm_fn,
        on_error=lambda stage, exc: errors.append((stage, type(exc).__name__)),
    )
    results = pipeline.run("q")
    assert len(results) == 4  # default error policy keeps them
    assert all(r["llm_score"] is None for r in results)
    assert errors == [("llm_fn", "BrokenBarrierError")] * 4  # reported from the calling thread, in order


def test_invalid_llm_workers_raises():
    with pytest.raises(ValueError, match="llm_workers"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_workers=0)


def test_invalid_llm_mode_and_pass_score_raise():
    with pytest.raises(ValueError, match="llm_mode"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_mode="vote")
    with pytest.raises(ValueError, match="llm_pass_score"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_pass_score=10)


# ---------------------------------------------------------------------------
# Candidate pool size
# ---------------------------------------------------------------------------


def _recording_search(requested):
    """search_fn that records the k it was asked for and returns k good docs."""

    def search_fn(embedding, k):
        requested.append(k)
        return [({"_id": str(i), "title": f"Doc {i}", "text": "..."}, 0.95 - i * 0.001) for i in range(k)]

    return search_fn


def test_search_fetches_three_times_top_k_by_default():
    requested = []
    pipeline = Pipeline(search_fn=_recording_search(requested), embed_fn=_mock_embed)
    results = pipeline.run("q")  # top_k defaults to 10
    assert requested == [30]
    assert len(results) == 10  # still sliced back to top_k


def test_candidate_multiplier_and_cap_are_configurable():
    requested = []
    search_fn = _recording_search(requested)

    Pipeline(search_fn=search_fn, embed_fn=_mock_embed, candidate_multiplier=4).run("q", top_k=5)
    Pipeline(search_fn=search_fn, embed_fn=_mock_embed, max_candidates=12).run("q", top_k=10)
    Pipeline(search_fn=search_fn, embed_fn=_mock_embed, candidate_multiplier=1.0).run("q", top_k=7)
    Pipeline(search_fn=search_fn, embed_fn=_mock_embed, candidate_multiplier=2.5).run("q", top_k=3)
    assert requested == [20, 12, 7, 8]  # 3 * 2.5 = 7.5 → rounds to 8


def test_fetch_k_helper_and_validation():
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, candidate_multiplier=3.0, max_candidates=50)
    assert pipeline.fetch_k(10) == 30
    assert pipeline.fetch_k(20) == 50
    with pytest.raises(ValueError, match="candidate_multiplier"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, candidate_multiplier=0.5)
    with pytest.raises(ValueError, match="max_candidates"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, max_candidates=0)


# ---------------------------------------------------------------------------
# Error policy: failures stay visible and are configurable
# ---------------------------------------------------------------------------


def _raising_llm(query, item):
    raise RuntimeError("LLM endpoint down")


def test_llm_failure_keeps_candidate_by_default_and_warns(caplog):
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_raising_llm)
    with caplog.at_level(logging.WARNING, logger="arf.pipeline"):
        results = pipeline.run("q")
    by_id = {r["document"].id: r for r in results}
    assert set(by_id) == {"1", "2"}
    assert by_id["2"]["score"] == 0.78
    assert by_id["2"]["llm_score"] is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("llm_fn failed" in m and "LLM endpoint down" in m for m in messages)


def test_llm_error_policy_drop():
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_raising_llm, llm_error_policy="drop")
    assert [r["document"].id for r in pipeline.run("q")] == ["1"]


def test_unparseable_llm_output_follows_error_policy(caplog):
    def chatty_llm(query, item):
        return "I think this one is quite relevant, honestly."

    kept = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=chatty_llm)
    with caplog.at_level(logging.WARNING, logger="arf.pipeline"):
        assert [r["document"].id for r in kept.run("q")] == ["1", "2"]
    assert any("output parsing failed" in r.getMessage() for r in caplog.records)

    dropped = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=chatty_llm, llm_error_policy="drop")
    assert [r["document"].id for r in dropped.run("q")] == ["1"]


def test_on_error_hook_receives_every_swallowed_error():
    seen = []

    def failing_cache_store(query, results):
        raise OSError("redis down")

    def failing_summarize(query, doc, context):
        raise RuntimeError("summary failed")

    def failing_store_query(query, data):
        raise OSError("mongo down")

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        llm_fn=_raising_llm,
        cache_store=failing_cache_store,
        store_query_fn=failing_store_query,
        summarize_fn=failing_summarize,
        on_error=lambda stage, exc: seen.append((stage, type(exc).__name__)),
    )
    results = pipeline.run("q")
    assert len(results) == 2
    assert all(r["summary"] == "" for r in results)
    assert ("llm_fn", "RuntimeError") in seen
    assert ("cache_store", "OSError") in seen
    assert ("store_query_fn", "OSError") in seen
    assert seen.count(("summarize_fn", "RuntimeError")) == 2


def test_on_error_hook_exceptions_are_ignored():
    def bad_hook(stage, exc):
        raise ValueError("hook bug")

    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_fn=_raising_llm, on_error=bad_hook)
    assert len(pipeline.run("q")) == 2


def test_predict_fn_failure_is_reported_and_falls_through(caplog):
    def broken_predict(vectors):
        raise ValueError("feature count mismatch")

    seen = []
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=broken_predict,
        llm_fn=_llm_saying({"2": 8}),
        on_error=lambda stage, exc: seen.append(stage),
    )
    with caplog.at_level(logging.WARNING, logger="arf.pipeline"):
        results = pipeline.run("q")
    assert [r["document"].id for r in results] == ["1", "2"]  # LLM still verified doc 2
    assert seen == ["predict_fn"]
    assert any("predict_fn failed" in r.getMessage() for r in caplog.records)


def test_invalid_llm_error_policy_raises():
    with pytest.raises(ValueError, match="llm_error_policy"):
        Pipeline(search_fn=_mock_search, embed_fn=_mock_embed, llm_error_policy="retry")


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
