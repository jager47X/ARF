"""Per-query and aggregate pipeline statistics.

Every :meth:`~arf.pipeline.Pipeline.run` produces a :class:`RunStats`
describing what each stage did — how many candidates it saw, accepted and
rejected, how long it took, which optional stages failed.
:class:`PipelineStats` accumulates them so the questions that matter for a
cost-first pipeline — *what share of candidates reached the LLM?*, *what is
the cache hit rate?*, *where does the latency go?* — are answered from live
traffic rather than from a README claim.

No external dependencies.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Optional

# Per-run counters that PipelineStats sums across runs.
_COUNTERS = (
    "fetched",
    "keyword_hits",
    "expansions",
    "rephrase_attempts",
    "accepted_direct",
    "needs_review",
    "rejected",
    "mlp_evaluated",
    "mlp_accepted",
    "mlp_rejected",
    "rerank_evaluated",
    "rerank_accepted",
    "rerank_rejected",
    "llm_evaluated",
    "llm_accepted",
    "llm_errors",
    "unreviewed_kept",
    "gap_dropped",
    "returned",
)

_FUNNEL = (
    "fetched",
    "accepted_direct",
    "needs_review",
    "rejected",
    "mlp_evaluated",
    "mlp_accepted",
    "mlp_rejected",
    "rerank_evaluated",
    "rerank_accepted",
    "rerank_rejected",
    "llm_evaluated",
    "llm_accepted",
    "llm_errors",
    "unreviewed_kept",
    "gap_dropped",
    "returned",
)


@dataclass
class RunStats:
    """What one :meth:`~arf.pipeline.Pipeline.run` did, stage by stage.

    Candidate counters describe the pool *after* keyword merging and query
    expansion (``fetched``) and how triage, the MLP, the re-ranker and the
    LLM split it.  ``timings_ms`` holds per-stage wall-clock time
    (``embed``, ``search``, ``keyword``, ``cache``, ``mlp``, ``rerank``,
    ``llm``, ``output``, ``total``).  ``errors`` lists the stages whose
    failure was swallowed (see ``Pipeline(on_error=...)``).
    """

    query: str = ""
    top_k: int = 0
    started_at: float = field(default_factory=time.time)
    blocked: bool = False
    cache_hit: bool = False
    cache_hops: int = 0
    semantic_cache_hit: bool = False
    rephrase_attempts: int = 0
    expansions: int = 0
    fetched: int = 0
    keyword_hits: int = 0
    accepted_direct: int = 0
    needs_review: int = 0
    rejected: int = 0
    mlp_evaluated: int = 0
    mlp_accepted: int = 0
    mlp_rejected: int = 0
    rerank_evaluated: int = 0
    rerank_accepted: int = 0
    rerank_rejected: int = 0
    llm_evaluated: int = 0
    llm_accepted: int = 0
    llm_errors: int = 0
    unreviewed_kept: int = 0
    gap_dropped: int = 0
    returned: int = 0
    errors: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def timer(self, stage: str) -> Iterator[None]:
        """Accumulate wall-clock milliseconds for *stage*."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.timings_ms[stage] = round(self.timings_ms.get(stage, 0.0) + elapsed, 3)

    @property
    def total_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)

    @property
    def llm_share(self) -> float:
        """Share of fetched candidates that reached the LLM (0.0 when nothing was fetched)."""
        return self.llm_evaluated / self.fetched if self.fetched else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["llm_share"] = round(self.llm_share, 4)
        data["total_ms"] = self.total_ms
        return data


class PipelineStats:
    """Thread-safe aggregate of :class:`RunStats` over a pipeline's lifetime.

    Args:
        keep_recent: How many of the latest :class:`RunStats` to keep for
            :meth:`recent` (0 keeps none).
    """

    def __init__(self, keep_recent: int = 100):
        self._lock = threading.Lock()
        self._recent: Deque[RunStats] = deque(maxlen=max(0, keep_recent))
        self.runs = 0
        self.blocked = 0
        self.cache_hits = 0
        self.semantic_cache_hits = 0
        self.total_ms = 0.0
        self.counters: Dict[str, int] = {name: 0 for name in _COUNTERS}
        self.errors_by_stage: Dict[str, int] = {}
        self.timings_ms: Dict[str, float] = {}

    def reset(self) -> None:
        with self._lock:
            self.runs = 0
            self.blocked = 0
            self.cache_hits = 0
            self.semantic_cache_hits = 0
            self.total_ms = 0.0
            self.counters = {name: 0 for name in _COUNTERS}
            self.errors_by_stage = {}
            self.timings_ms = {}
            self._recent.clear()

    def record(self, run: RunStats) -> None:
        with self._lock:
            self.runs += 1
            self.blocked += int(run.blocked)
            self.cache_hits += int(run.cache_hit)
            self.semantic_cache_hits += int(run.semantic_cache_hit)
            self.total_ms += run.total_ms
            for name in _COUNTERS:
                self.counters[name] += int(getattr(run, name))
            for stage in run.errors:
                self.errors_by_stage[stage] = self.errors_by_stage.get(stage, 0) + 1
            for stage, ms in run.timings_ms.items():
                self.timings_ms[stage] = self.timings_ms.get(stage, 0.0) + ms
            if self._recent.maxlen:
                self._recent.append(run)

    def recent(self, n: Optional[int] = None) -> List[RunStats]:
        """The latest runs, oldest first."""
        with self._lock:
            runs = list(self._recent)
        return runs[-n:] if n else runs

    def summary(self) -> Dict[str, Any]:
        """Rates and totals suitable for a dashboard or a benchmark table.

        ``llm_share_of_candidates`` is the number the R-Flow design is about:
        the fraction of fetched candidates that needed an LLM call.
        ``llm_calls_avoided`` counts review-band candidates that the MLP,
        the re-ranker or the unreviewed policy resolved without one.
        """

        def rate(numerator: float, denominator: float) -> float:
            return round(numerator / denominator, 4) if denominator else 0.0

        with self._lock:
            runs = self.runs
            c = dict(self.counters)
            searched = runs - self.cache_hits - self.blocked
            return {
                "runs": runs,
                "blocked": self.blocked,
                "searched": searched,
                "cache_hits": self.cache_hits,
                "semantic_cache_hits": self.semantic_cache_hits,
                "cache_hit_rate": rate(self.cache_hits, runs),
                "funnel": {name: c[name] for name in _FUNNEL},
                "llm_share_of_candidates": rate(c["llm_evaluated"], c["fetched"]),
                "llm_calls_per_query": rate(c["llm_evaluated"], searched),
                "llm_calls_avoided": c["needs_review"] - c["llm_evaluated"],
                "keyword_hits": c["keyword_hits"],
                "expansions": c["expansions"],
                "rephrase_attempts": c["rephrase_attempts"],
                "avg_latency_ms": rate(self.total_ms, runs),
                "avg_stage_ms": {stage: rate(ms, runs) for stage, ms in self.timings_ms.items()},
                "errors_by_stage": dict(self.errors_by_stage),
            }
