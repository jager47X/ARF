"""Tests for arf.stats — per-run RunStats produced by the pipeline and the PipelineStats aggregate."""

from arf.pipeline import Pipeline
from arf.stats import PipelineStats, RunStats


def _embed(text):
    return [0.1] * 4


def _search(embedding, k):
    return [
        ({"_id": "1", "title": "A", "text": "..."}, 0.92),  # accepted outright
        ({"_id": "2", "title": "B", "text": "..."}, 0.78),  # review band
        ({"_id": "3", "title": "C", "text": "..."}, 0.50),  # rejected
    ]


def test_run_stats_describe_the_funnel():
    pipeline = Pipeline(search_fn=_search, embed_fn=_embed, llm_fn=lambda q, item: "Score: 8")
    results = pipeline.run("what is due process", top_k=5)
    s = pipeline.last_stats

    assert isinstance(s, RunStats)
    assert s.query == "what is due process" and s.top_k == 5
    assert (s.fetched, s.accepted_direct, s.needs_review, s.rejected) == (3, 1, 1, 1)
    assert (s.llm_evaluated, s.llm_accepted, s.llm_errors) == (1, 1, 0)
    assert s.mlp_evaluated == 0 and s.rerank_evaluated == 0
    assert s.returned == len(results) == 2
    assert s.cache_hit is False and s.errors == []
    assert s.llm_share == 1 / 3
    for stage in ("embed", "search", "llm", "output", "total"):
        assert stage in s.timings_ms and s.timings_ms[stage] >= 0
    assert s.to_dict()["llm_share"] == round(1 / 3, 4)


def test_run_stats_count_mlp_rerank_and_unreviewed_decisions():
    def search(embedding, k):
        return [({"_id": str(i), "title": f"D{i}", "text": "..."}, 0.80 - i * 0.01) for i in range(4)]

    pipeline = Pipeline(
        search_fn=search,
        embed_fn=_embed,
        predict_fn=lambda vectors: [0.9, 0.5, 0.5, 0.1],  # accept / uncertain / uncertain / reject
        rerank_fn=lambda q, items: [0.9, 0.5],  # accept / uncertain
    )
    pipeline.run("q")
    s = pipeline.last_stats
    assert (s.mlp_evaluated, s.mlp_accepted, s.mlp_rejected) == (4, 1, 1)
    assert (s.rerank_evaluated, s.rerank_accepted, s.rerank_rejected) == (2, 1, 0)
    assert s.llm_evaluated == 0 and s.unreviewed_kept == 1
    assert s.returned == 3


def test_run_stats_record_cache_hits_errors_and_rephrases():
    cache = {}
    errors_seen = []

    def failing_summarize(query, doc, context):
        raise RuntimeError("boom")

    pipeline = Pipeline(
        search_fn=_search,
        embed_fn=_embed,
        cache_lookup=cache.get,
        cache_store=lambda key, payload: cache.setdefault(key, {}).update(results=payload),
        summarize_fn=failing_summarize,
        on_error=lambda stage, exc: errors_seen.append(stage),
    )
    pipeline.run("q")
    first = pipeline.last_stats
    assert first.cache_hit is False
    assert first.errors == ["summarize_fn", "summarize_fn"] == errors_seen

    pipeline.run("Q ")
    second = pipeline.last_stats
    assert second.cache_hit is True and second.cache_hops == 0
    assert second.fetched == 0 and second.returned == 2
    assert "llm" not in second.timings_ms and "cache" in second.timings_ms

    attempts = []

    def flaky_search(embedding, k):
        attempts.append(k)
        return [] if len(attempts) == 1 else _search(embedding, k)

    rephrasing = Pipeline(search_fn=flaky_search, embed_fn=_embed, rephrase_fn=lambda q, prev: q + " (rephrased)")
    rephrasing.run("q")
    assert rephrasing.last_stats.rephrase_attempts == 1
    assert rephrasing.last_stats.fetched == 3  # counted once the retry found something


def test_blocked_queries_are_recorded():
    pipeline = Pipeline(search_fn=_search, embed_fn=_embed, moderate_fn=lambda q: False)
    assert pipeline.run("bad") == []
    assert pipeline.last_stats.blocked is True
    assert pipeline.stats.summary()["blocked"] == 1


def test_stats_fn_is_called_and_may_not_break_the_query():
    seen = []
    pipeline = Pipeline(search_fn=_search, embed_fn=_embed, stats_fn=lambda s: seen.append(s.returned))
    pipeline.run("q")
    assert seen == [2]

    def bad_hook(stats):
        raise ValueError("exporter down")

    assert len(Pipeline(search_fn=_search, embed_fn=_embed, stats_fn=bad_hook).run("q")) == 2


def test_stats_are_recorded_even_when_the_pipeline_raises():
    def broken_search(embedding, k):
        raise ConnectionError("vector db down")

    pipeline = Pipeline(search_fn=broken_search, embed_fn=_embed)
    try:
        pipeline.run("q")
    except ConnectionError:
        pass
    assert pipeline.last_stats is not None
    assert pipeline.last_stats.returned == 0
    assert pipeline.stats.runs == 1


def test_pipeline_stats_aggregate_and_summarise():
    pipeline = Pipeline(search_fn=_search, embed_fn=_embed, llm_fn=lambda q, item: "Score: 8")
    for query in ("a", "b", "c"):
        pipeline.run(query)
    summary = pipeline.stats.summary()
    assert summary["runs"] == 3 and summary["searched"] == 3 and summary["cache_hits"] == 0
    assert summary["funnel"]["fetched"] == 9
    assert summary["funnel"]["llm_evaluated"] == 3 and summary["funnel"]["llm_accepted"] == 3
    assert summary["llm_share_of_candidates"] == round(3 / 9, 4)
    assert summary["llm_calls_per_query"] == 1.0
    assert summary["llm_calls_avoided"] == 0
    assert summary["avg_latency_ms"] >= 0 and "llm" in summary["avg_stage_ms"]
    assert [r.query for r in pipeline.stats.recent()] == ["a", "b", "c"]
    assert [r.query for r in pipeline.stats.recent(2)] == ["b", "c"]

    pipeline.stats.reset()
    assert pipeline.stats.summary()["runs"] == 0 and pipeline.stats.recent() == []


def test_pipeline_stats_without_llm_count_avoided_calls_and_keep_recent_limit():
    stats = PipelineStats(keep_recent=2)
    for i in range(3):
        stats.record(RunStats(query=f"q{i}", fetched=10, needs_review=4, llm_evaluated=1, errors=["llm_fn"]))
    summary = stats.summary()
    assert summary["llm_calls_avoided"] == 9  # 12 review-band candidates, 3 LLM calls
    assert summary["errors_by_stage"] == {"llm_fn": 3}
    assert [r.query for r in stats.recent()] == ["q1", "q2"]
    assert PipelineStats(keep_recent=0).recent() == []
