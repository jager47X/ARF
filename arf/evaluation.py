"""Offline evaluation of a :class:`~arf.pipeline.Pipeline` on labelled queries.

Pure Python.  A dataset is any iterable of dicts::

    {"query": "what is due process?", "relevant": ["doc-14", "doc-5"]}
    {"query": "...",                  "relevant": {"doc-14": 3, "doc-5": 1}}   # graded

:func:`evaluate` runs every query through the pipeline and reports the
standard ranking metrics (recall@k, precision@k, hit-rate@k, MRR, nDCG@k)
together with the pipeline's own stage statistics — so one run says both
how good the results are *and* what they cost (LLM calls per query, cache
hit rate, latency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from arf.stats import PipelineStats

Relevance = Union[Sequence[Any], Mapping[Any, float]]


def _grades(relevant: Relevance) -> Dict[str, float]:
    """Normalise a list of ids or an ``{id: grade}`` mapping to ``{str(id): grade}``."""
    if isinstance(relevant, Mapping):
        return {str(k): float(v) for k, v in relevant.items() if float(v) > 0}
    return {str(k): 1.0 for k in relevant}


def precision_at_k(retrieved: Sequence[Any], relevant: Relevance, k: int) -> float:
    grades = _grades(relevant)
    top = [str(r) for r in retrieved[:k]]
    return sum(1 for r in top if r in grades) / k if k > 0 else 0.0


def recall_at_k(retrieved: Sequence[Any], relevant: Relevance, k: int) -> float:
    grades = _grades(relevant)
    if not grades:
        return 0.0
    top = {str(r) for r in retrieved[:k]}
    return sum(1 for r in grades if r in top) / len(grades)


def hit_rate_at_k(retrieved: Sequence[Any], relevant: Relevance, k: int) -> float:
    grades = _grades(relevant)
    return 1.0 if any(str(r) in grades for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[Any], relevant: Relevance) -> float:
    grades = _grades(relevant)
    for i, r in enumerate(retrieved):
        if str(r) in grades:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[Any], relevant: Relevance, k: int) -> float:
    """Normalised DCG with ``gain = 2^grade - 1`` (binary lists use grade 1)."""
    grades = _grades(relevant)
    if not grades:
        return 0.0
    dcg = sum((2 ** grades[str(r)] - 1) / math.log2(i + 2) for i, r in enumerate(retrieved[:k]) if str(r) in grades)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def metrics_for(retrieved: Sequence[Any], relevant: Relevance, ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """All metrics for one query, keyed like ``recall@5`` / ``mrr``."""
    out: Dict[str, float] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return out


@dataclass
class QueryEval:
    query: str
    retrieved: List[str]
    relevant: Dict[str, float]
    metrics: Dict[str, float]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Outcome of :func:`evaluate`.

    Attributes:
        n_queries: Number of queries evaluated.
        metrics: Mean of each metric over all queries.
        per_query: One :class:`QueryEval` per dataset entry.
        stats: :meth:`~arf.stats.PipelineStats.summary` over the evaluation
            runs only (LLM share, cache hit rate, latency ...).
    """

    n_queries: int
    metrics: Dict[str, float]
    per_query: List[QueryEval]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "metrics": dict(self.metrics),
            "stats": dict(self.stats),
            "per_query": [
                {
                    "query": q.query,
                    "retrieved": list(q.retrieved),
                    "relevant": dict(q.relevant),
                    "metrics": dict(q.metrics),
                    "stats": dict(q.stats),
                }
                for q in self.per_query
            ],
        }

    def markdown(self) -> str:
        """A compact table — paste it into a README or a PR."""
        lines = ["| metric | value |", "|---|---|"]
        for name, value in self.metrics.items():
            lines.append(f"| {name} | {value:.4f} |")
        s = self.stats
        lines.append(f"| llm_share_of_candidates | {s.get('llm_share_of_candidates', 0.0):.4f} |")
        lines.append(f"| llm_calls_per_query | {s.get('llm_calls_per_query', 0.0):.4f} |")
        lines.append(f"| cache_hit_rate | {s.get('cache_hit_rate', 0.0):.4f} |")
        lines.append(f"| avg_latency_ms | {s.get('avg_latency_ms', 0.0):.1f} |")
        lines.append(f"| queries | {self.n_queries} |")
        return "\n".join(lines)


def _default_id_fn(result: Dict[str, Any]) -> str:
    doc = result.get("document")
    doc_id = getattr(doc, "id", None)
    if doc_id is None and isinstance(doc, dict):
        doc_id = doc.get("_id", doc.get("id"))
    return str(doc_id)


def evaluate(
    pipeline: Any,
    dataset: Iterable[Mapping[str, Any]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    top_k: Optional[int] = None,
    id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    progress_fn: Optional[Callable[[int, QueryEval], None]] = None,
) -> EvalResult:
    """Run every dataset query through *pipeline* and score the results.

    Args:
        pipeline: A :class:`~arf.pipeline.Pipeline` (anything with
            ``run(query, top_k=...)`` and ``last_stats``).
        dataset: Dicts with ``"query"`` and ``"relevant"`` (a list of ids
            or an ``{id: grade}`` mapping; ``"relevant_ids"`` is accepted
            too).
        ks: Cut-offs for the ``@k`` metrics.
        top_k: Results to request per query (default ``max(ks)``).
        id_fn: ``(result_dict) -> id`` — defaults to ``result["document"].id``.
        progress_fn: ``(index, QueryEval) -> None`` called after each query.
    """
    id_of = id_fn or _default_id_fn
    k_results = top_k or max(ks)
    agg = PipelineStats(keep_recent=0)
    per_query: List[QueryEval] = []

    for index, entry in enumerate(dataset):
        query = str(entry["query"])
        relevant = _grades(entry.get("relevant", entry.get("relevant_ids", [])) or [])
        results = pipeline.run(query, top_k=k_results)
        retrieved = [id_of(r) for r in results]
        run_stats = getattr(pipeline, "last_stats", None)
        if run_stats is not None:
            agg.record(run_stats)
        q = QueryEval(
            query=query,
            retrieved=retrieved,
            relevant=relevant,
            metrics=metrics_for(retrieved, relevant, ks),
            stats=run_stats.to_dict() if run_stats is not None else {},
        )
        per_query.append(q)
        if progress_fn is not None:
            progress_fn(index, q)

    means: Dict[str, float] = {}
    if per_query:
        for name in per_query[0].metrics:
            means[name] = round(sum(q.metrics[name] for q in per_query) / len(per_query), 4)

    return EvalResult(n_queries=len(per_query), metrics=means, per_query=per_query, stats=agg.summary())
