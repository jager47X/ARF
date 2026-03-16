"""Tests for cost tracking."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.cost_tracker import CostTracker


class TestCostTracker:
    def test_basic_flow(self):
        t = CostTracker()
        t.start_query("test query")
        t.record_embedding_call(tokens=128)
        t.record_llm_call("rerank", input_tokens=500, output_tokens=50)
        t.record_cache_miss()
        result = t.end_query()

        assert result["query"] == "test query"
        assert result["embedding_tokens"] == 128
        assert result["llm_calls"] == 1
        assert result["cache_misses"] == 1
        assert result["estimated_cost_usd"] > 0

    def test_cache_hit_rate(self):
        t = CostTracker()
        t.start_query("q1")
        t.record_cache_hit()
        t.record_cache_hit()
        t.record_cache_miss()
        result = t.end_query()

        assert result["cache_hit_rate"] == 2 / 3

    def test_summary(self):
        t = CostTracker()

        t.start_query("q1")
        t.record_embedding_call(tokens=100)
        t.end_query()

        t.start_query("q2")
        t.record_embedding_call(tokens=200)
        t.record_llm_call("rerank", input_tokens=500, output_tokens=50)
        t.end_query()

        s = t.summary()
        assert s["total_queries"] == 2
        assert s["total_llm_calls"] == 1
        assert s["llm_rerank_frequency"] == 0.5  # 1 out of 2 queries used rerank

    def test_empty_summary(self):
        t = CostTracker()
        assert t.summary() == {"total_queries": 0}
