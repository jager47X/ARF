"""
Cost tracking for ARF query pipeline.

Wraps query processing to measure token usage, LLM call frequency,
cache hit rates, and estimated cost per query.

Usage:
    tracker = CostTracker()
    tracker.start_query("some query")
    # ... run pipeline ...
    tracker.record_embedding_call(tokens=128)
    tracker.record_llm_call(call_type="rerank", input_tokens=500, output_tokens=50)
    tracker.record_cache_hit()
    summary = tracker.end_query()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Approximate pricing (USD per 1M tokens) as of 2025
PRICING = {
    "voyage-3-large": {"input": 0.06},          # embedding
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # moderation / fix_query
    "gpt-4o": {"input": 2.50, "output": 10.00},      # reranking / summaries
    "o3-mini": {"input": 1.10, "output": 4.40},       # reasoning
}


@dataclass
class QueryCost:
    query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    embedding_tokens: int = 0
    embedding_calls: int = 0
    llm_calls: List[Dict] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on token counts."""
        # Embedding cost
        cost = self.embedding_tokens * PRICING["voyage-3-large"]["input"] / 1_000_000

        # LLM costs (assume gpt-4o for reranking/summaries, gpt-4o-mini for others)
        for call in self.llm_calls:
            model = "gpt-4o" if call.get("type") in ("rerank", "summary") else "gpt-4o-mini"
            pricing = PRICING[model]
            cost += call.get("input_tokens", 0) * pricing["input"] / 1_000_000
            cost += call.get("output_tokens", 0) * pricing["output"] / 1_000_000

        return cost

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "latency_ms": round(self.latency_ms, 1),
            "embedding_tokens": self.embedding_tokens,
            "embedding_calls": self.embedding_calls,
            "llm_calls": len(self.llm_calls),
            "llm_call_breakdown": {
                call_type: sum(1 for c in self.llm_calls if c.get("type") == call_type)
                for call_type in {c.get("type") for c in self.llm_calls}
            },
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class CostTracker:
    """Accumulates cost data across multiple queries."""

    def __init__(self):
        self._current: Optional[QueryCost] = None
        self.queries: List[QueryCost] = []

    def start_query(self, query: str) -> None:
        self._current = QueryCost(query=query, start_time=time.time())

    def record_embedding_call(self, tokens: int = 0) -> None:
        if self._current:
            self._current.embedding_calls += 1
            self._current.embedding_tokens += tokens
            self._current.total_input_tokens += tokens

    def record_llm_call(
        self, call_type: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        if self._current:
            self._current.llm_calls.append({
                "type": call_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            self._current.total_input_tokens += input_tokens
            self._current.total_output_tokens += output_tokens

    def record_cache_hit(self) -> None:
        if self._current:
            self._current.cache_hits += 1

    def record_cache_miss(self) -> None:
        if self._current:
            self._current.cache_misses += 1

    def end_query(self) -> dict:
        if self._current:
            self._current.end_time = time.time()
            self.queries.append(self._current)
            result = self._current.to_dict()
            self._current = None
            return result
        return {}

    def summary(self) -> dict:
        """Aggregate stats across all tracked queries."""
        if not self.queries:
            return {"total_queries": 0}

        total = len(self.queries)
        total_cost = sum(q.estimated_cost_usd for q in self.queries)
        total_llm = sum(len(q.llm_calls) for q in self.queries)
        total_cache_hits = sum(q.cache_hits for q in self.queries)
        total_cache_misses = sum(q.cache_misses for q in self.queries)
        avg_latency = sum(q.latency_ms for q in self.queries) / total

        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_query_usd": round(total_cost / total, 6),
            "total_llm_calls": total_llm,
            "avg_llm_calls_per_query": round(total_llm / total, 2),
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "overall_cache_hit_rate": (
                round(total_cache_hits / (total_cache_hits + total_cache_misses), 3)
                if (total_cache_hits + total_cache_misses) > 0
                else 0.0
            ),
            "llm_rerank_frequency": round(
                sum(1 for q in self.queries if any(c.get("type") == "rerank" for c in q.llm_calls)) / total, 3
            ),
        }
