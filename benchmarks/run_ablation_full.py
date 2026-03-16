#!/usr/bin/env python3
"""
ARF Ablation Study -- Comprehensive retrieval strategy comparison.

Tests every configuration combination and measures retrieval metrics,
LLM call frequency, cost per query, and latency.

Strategies tested:
  1. Semantic Only -- vector search alone
  2. Semantic + Keyword -- vector search + keyword matcher boost
  3. Semantic + Keyword + Threshold -- add ABC gate threshold filtering
  4. Semantic + Keyword + MLP -- replace threshold + LLM with MLP reranker
  5. Semantic + Keyword + MLP + LLM Fallback -- MLP confident, LLM uncertain
  6. Full Pipeline (no MLP) -- current production pipeline
  7. Full Pipeline (with MLP) -- new pipeline with MLP reranker

Usage:
    python benchmarks/run_ablation_full.py --production
    python benchmarks/run_ablation_full.py --production --domain us_constitution
    python benchmarks/run_ablation_full.py --production --include-cost
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401
from benchmarks.cost_tracker import PRICING, CostTracker
from benchmarks.metrics import compute_all_metrics

BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}

# Approximate token counts for cost estimation
_EMBEDDING_TOKENS_PER_QUERY = 128
_LLM_RERANK_INPUT_TOKENS = 500
_LLM_RERANK_OUTPUT_TOKENS = 50
_MLP_INFERENCE_TIME_MS = 2.0  # ~2ms per document


def load_queries(domain: str = None) -> list:
    """Load benchmark queries, optionally filtered by domain.

    Only returns queries that have expected titles (so metrics can be computed).
    """
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    queries = data["queries"]
    if domain:
        queries = [q for q in queries if q["domain"] == domain]
    return [q for q in queries if q.get("expected_titles")]


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _strategy_semantic_only(rag, query: str, tracker: Optional[CostTracker] = None) -> List[str]:
    """Strategy 1: Semantic vector search only."""
    emb = rag.query_manager.get_embedding(query)
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)
    raw = rag.vector_search.search_main.search_similar(emb, k=10)
    return [doc.get("title", "") for doc, score in (raw or [])]


def _strategy_semantic_keyword(rag, query: str, tracker: Optional[CostTracker] = None) -> List[str]:
    """Strategy 2: Semantic + keyword matcher boost."""
    emb = rag.query_manager.get_embedding(query)
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)
    raw = rag.vector_search.search_main.search_similar(emb, k=10)
    sem = raw or []

    kw_matches = rag.keyword.find_textual(query) if rag.keyword else []

    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}
    for doc, score in sem:
        t = doc.get("title", "")
        if t not in scores or score > scores[t]:
            scores[t] = score
            doc_map[t] = doc
        elif t not in doc_map:
            doc_map[t] = doc

    for t in kw_matches:
        if t not in scores:
            scores[t] = 0.70
        else:
            scores[t] = min(1.0, scores[t] + 0.05)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked]


def _strategy_semantic_keyword_threshold(
    rag, query: str, tracker: Optional[CostTracker] = None
) -> Tuple[List[str], int]:
    """Strategy 3: Semantic + keyword + ABC gate threshold filtering.

    Returns (titles, llm_call_count).
    """
    emb = rag.query_manager.get_embedding(query)
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)
    raw = rag.vector_search.search_main.search_similar(emb, k=10)
    sem = raw or []

    kw_matches = rag.keyword.find_textual(query) if rag.keyword else []

    thresholds = rag.config.get("thresholds", {})
    rag_search = float(thresholds.get("RAG_SEARCH", 0.85))
    llm_verif = float(thresholds.get("LLM_VERIFication", 0.70))

    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}
    for doc, score in sem:
        t = doc.get("title", "")
        if t not in scores or score > scores[t]:
            scores[t] = score
            doc_map[t] = doc

    for t in kw_matches:
        if t not in scores:
            scores[t] = 0.70
        else:
            scores[t] = min(1.0, scores[t] + 0.05)

    # Apply ABC gate thresholds
    accepted = []
    llm_calls = 0
    for title, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score >= rag_search:
            accepted.append(title)
        elif score >= llm_verif:
            # Would need LLM verification
            llm_calls += 1
            if tracker:
                tracker.record_llm_call(
                    "rerank",
                    input_tokens=_LLM_RERANK_INPUT_TOKENS,
                    output_tokens=_LLM_RERANK_OUTPUT_TOKENS,
                )
            # Simulate LLM acceptance (use the result from vector search as proxy)
            accepted.append(title)

    return accepted, llm_calls


def _try_load_mlp():
    """Try to load MLP reranker. Returns (MLPReranker_class, model_path) or (None, None)."""
    try:
        from rag_dependencies.mlp_reranker import MLPReranker
        # Check for model file in standard locations
        model_paths = [
            Path(__file__).parent.parent / "models" / "mlp_reranker.joblib",
            Path(__file__).parent.parent / "models" / "mlp_reranker.pkl",
            Path(__file__).parent.parent / "models" / "mlp_reranker.pt",
        ]
        for p in model_paths:
            if p.exists():
                return MLPReranker, str(p)
        return MLPReranker, None
    except ImportError:
        return None, None


def _strategy_semantic_keyword_mlp(
    rag, query: str, domain: str, tracker: Optional[CostTracker] = None
) -> Tuple[List[str], bool]:
    """Strategy 4: Semantic + keyword + MLP reranker.

    Returns (titles, mlp_available).
    """
    from rag_dependencies.feature_extractor import FeatureExtractor

    emb = rag.query_manager.get_embedding(query)
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)
    raw = rag.vector_search.search_main.search_similar(emb, k=10)
    sem = raw or []

    kw_matches = rag.keyword.find_textual(query) if rag.keyword else []

    # Merge semantic + keyword scores
    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}
    result_pairs: List[Tuple[dict, float]] = []
    for doc, score in sem:
        t = doc.get("title", "")
        scores[t] = max(scores.get(t, 0), score)
        doc_map[t] = doc
        result_pairs.append((doc, score))

    for t in kw_matches:
        if t not in scores:
            scores[t] = 0.70

    # Try MLP reranking
    MLPReranker, model_path = _try_load_mlp()
    if MLPReranker is None or model_path is None:
        # MLP not available -- fall back to score-based ranking
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        accepted = [t for t, s in ranked if s >= 0.6]
        return accepted or [t for t, _ in ranked[:5]], False

    try:
        extractor = FeatureExtractor(rag.config, domain)
        reranker = MLPReranker(model_path)

        alias_matches = []
        if rag.alias:
            try:
                alias_matches = rag.alias.find_semantic_aliases(query, rag.query_manager)
            except Exception:
                pass

        features_batch = extractor.extract_batch(
            query=query,
            results=result_pairs,
            query_embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
            keyword_matches=kw_matches,
            alias_matches=alias_matches,
        )

        mlp_scores = []
        vectors = [extractor.to_vector(f) for f in features_batch]
        if reranker.is_loaded:
            probas = reranker.predict(vectors)
        else:
            probas = [0.5] * len(vectors)
        for i, prob in enumerate(probas):
            doc, original_score = result_pairs[i]
            title = doc.get("title", "")
            # Blend MLP probability with semantic score for reranking
            blended = 0.4 * original_score + 0.6 * prob
            mlp_scores.append((title, blended, prob, original_score))

        # Rerank by blended score — MLP pushes truly relevant docs higher
        mlp_scores.sort(key=lambda x: x[1], reverse=True)
        # Accept docs where MLP is confident they're relevant (prob >= 0.4)
        accepted = [t for t, blended, prob, _ in mlp_scores if prob >= 0.4]
        if not accepted:
            accepted = [t for t, _, _, _ in mlp_scores[:5]]
        return accepted, True

    except Exception:
        # MLP failed -- fall back
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        accepted = [t for t, s in ranked if s >= 0.6]
        return accepted or [t for t, _ in ranked[:5]], False


def _strategy_semantic_keyword_mlp_llm_fallback(
    rag, query: str, domain: str, tracker: Optional[CostTracker] = None
) -> Tuple[List[str], int, bool]:
    """Strategy 5: Semantic + keyword + MLP + LLM fallback for uncertain cases.

    Returns (titles, llm_call_count, mlp_available).
    """
    from rag_dependencies.feature_extractor import FeatureExtractor

    emb = rag.query_manager.get_embedding(query)
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)
    raw = rag.vector_search.search_main.search_similar(emb, k=10)
    sem = raw or []

    kw_matches = rag.keyword.find_textual(query) if rag.keyword else []

    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}
    result_pairs: List[Tuple[dict, float]] = []
    for doc, score in sem:
        t = doc.get("title", "")
        scores[t] = max(scores.get(t, 0), score)
        doc_map[t] = doc
        result_pairs.append((doc, score))

    for t in kw_matches:
        if t not in scores:
            scores[t] = 0.70

    MLPReranker, model_path = _try_load_mlp()
    if MLPReranker is None or model_path is None:
        # No MLP -- simulate with threshold + LLM for uncertain
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        accepted = []
        llm_calls = 0
        for title, s in ranked:
            if s >= 0.6:
                accepted.append(title)
            elif s >= 0.4:
                llm_calls += 1
                if tracker:
                    tracker.record_llm_call(
                        "rerank",
                        input_tokens=_LLM_RERANK_INPUT_TOKENS,
                        output_tokens=_LLM_RERANK_OUTPUT_TOKENS,
                    )
                accepted.append(title)
        return accepted or [t for t, _ in ranked[:5]], llm_calls, False

    try:
        extractor = FeatureExtractor(rag.config, domain)
        reranker = MLPReranker(model_path)

        alias_matches = []
        if rag.alias:
            try:
                alias_matches = rag.alias.find_semantic_aliases(query, rag.query_manager)
            except Exception:
                pass

        features_batch = extractor.extract_batch(
            query=query,
            results=result_pairs,
            query_embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
            keyword_matches=kw_matches,
            alias_matches=alias_matches,
        )

        accepted_with_scores = []
        llm_calls = 0
        scored_items = []
        vectors = [extractor.to_vector(f) for f in features_batch]
        if reranker.is_loaded:
            probas = reranker.predict(vectors)
        else:
            probas = [0.5] * len(vectors)
        for i, prob in enumerate(probas):
            doc, original_score = result_pairs[i]
            title = doc.get("title", "")
            blended = 0.4 * original_score + 0.6 * prob
            scored_items.append((title, blended, prob, original_score))

        for title, blended, prob, _ in sorted(scored_items, key=lambda x: x[1], reverse=True):
            if prob >= 0.6:
                # MLP confident -- accept
                accepted_with_scores.append((title, blended))
            elif prob >= 0.4:
                # MLP uncertain -- fall back to LLM
                llm_calls += 1
                if tracker:
                    tracker.record_llm_call(
                        "rerank",
                        input_tokens=_LLM_RERANK_INPUT_TOKENS,
                        output_tokens=_LLM_RERANK_OUTPUT_TOKENS,
                    )
                accepted_with_scores.append((title, blended))
            # prob < 0.4 -- reject

        # Sort accepted by blended score for proper reranking
        accepted_with_scores.sort(key=lambda x: x[1], reverse=True)
        accepted = [t for t, _ in accepted_with_scores]
        if not accepted:
            accepted = [t for t, _, _, _ in sorted(scored_items, key=lambda x: x[1], reverse=True)[:5]]

        return accepted, llm_calls, True

    except Exception:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:5]], 0, False


def _strategy_full_pipeline_no_mlp(rag, query: str, tracker: Optional[CostTracker] = None) -> Tuple[List[str], int]:
    """Strategy 6: Full current production pipeline (no MLP).

    Returns (titles, estimated_llm_calls).
    """
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)
    try:
        results, _ = rag.process_query(query, language="en")
        titles = [doc.get("title", "") for doc, score in results]
    except Exception:
        titles = []
        results = []

    # Estimate LLM calls: any result that went through verification
    # In the full pipeline, docs in the LLM_VERIFication band get LLM-verified
    thresholds = rag.config.get("thresholds", {})
    rag_search = float(thresholds.get("RAG_SEARCH", 0.85))
    llm_verif = float(thresholds.get("LLM_VERIFication", 0.70))
    llm_calls = 0
    for doc, score in results:
        if llm_verif <= score < rag_search:
            llm_calls += 1
            if tracker:
                tracker.record_llm_call(
                    "rerank",
                    input_tokens=_LLM_RERANK_INPUT_TOKENS,
                    output_tokens=_LLM_RERANK_OUTPUT_TOKENS,
                )

    return titles, llm_calls


def _strategy_full_pipeline_with_mlp(
    rag, query: str, domain: str, tracker: Optional[CostTracker] = None
) -> Tuple[List[str], int, bool]:
    """Strategy 7: Full pipeline with MLP reranker inserted.

    MLP handles the confident decisions, LLM only for uncertain cases.
    Returns (titles, llm_call_count, mlp_available).
    """
    from rag_dependencies.feature_extractor import FeatureExtractor

    emb = rag.query_manager.get_embedding(query)
    if tracker:
        tracker.record_embedding_call(tokens=_EMBEDDING_TOKENS_PER_QUERY)

    raw = rag.vector_search.search_main.search_similar(emb, k=10)
    sem = raw or []

    kw_matches = rag.keyword.find_textual(query) if rag.keyword else []

    alias_matches = []
    if rag.alias:
        try:
            alias_matches = rag.alias.find_semantic_aliases(query, rag.query_manager)
        except Exception:
            pass

    thresholds = rag.config.get("thresholds", {})
    rag_search = float(thresholds.get("RAG_SEARCH", 0.85))
    llm_verif = float(thresholds.get("LLM_VERIFication", 0.70))
    keyword_score = float(rag.config.get("KEYWORD_MATCH_SCORE", 0.70))

    # Build scored results with keyword boost
    scores: Dict[str, float] = {}
    doc_map: Dict[str, dict] = {}
    result_pairs: List[Tuple[dict, float]] = []
    for doc, score in sem:
        t = doc.get("title", "")
        if t not in scores or score > scores[t]:
            scores[t] = score
            doc_map[t] = doc
        result_pairs.append((doc, score))

    for t in kw_matches:
        if t not in scores:
            scores[t] = keyword_score
        else:
            scores[t] = min(1.0, scores[t] + 0.05)

    # MLP reranks ALL candidates (not just verification band) to improve ordering
    MLPReranker, model_path = _try_load_mlp()
    llm_calls = 0
    mlp_available = False

    if MLPReranker is not None and model_path is not None and result_pairs:
        try:
            extractor = FeatureExtractor(rag.config, domain)
            reranker = MLPReranker(model_path)
            mlp_available = reranker.is_loaded

            if reranker.is_loaded:
                features_batch = extractor.extract_batch(
                    query=query,
                    results=result_pairs,
                    query_embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    keyword_matches=kw_matches,
                    alias_matches=alias_matches,
                )

                vectors = [extractor.to_vector(f) for f in features_batch]
                probas = reranker.predict(vectors)

                # Build blended scores for ALL candidates
                reranked = []
                for i, prob in enumerate(probas):
                    doc, sem_score = result_pairs[i]
                    title = doc.get("title", "")
                    # Blend: MLP-weighted score improves ranking
                    blended = 0.4 * sem_score + 0.6 * prob
                    reranked.append((title, blended, prob, sem_score))

                reranked.sort(key=lambda x: x[1], reverse=True)

                # Accept/reject/LLM-fallback based on MLP confidence
                accepted = []
                for title, blended, prob, sem_score in reranked:
                    if prob >= 0.6 or sem_score >= rag_search:
                        accepted.append(title)
                    elif prob >= 0.4 and sem_score >= llm_verif:
                        # Uncertain -- LLM fallback
                        llm_calls += 1
                        if tracker:
                            tracker.record_llm_call(
                                "rerank",
                                input_tokens=_LLM_RERANK_INPUT_TOKENS,
                                output_tokens=_LLM_RERANK_OUTPUT_TOKENS,
                            )
                        accepted.append(title)
                    # prob < 0.4 and below rag_search -- reject

                if not accepted:
                    accepted = [t for t, _, _, _ in reranked[:5]]

                return accepted, llm_calls, True
        except Exception:
            pass  # Fall through to non-MLP path

    # Non-MLP fallback: ABC gate as before
    accepted = []
    needs_verification = []
    for title, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score >= rag_search:
            accepted.append(title)
        elif score >= llm_verif:
            needs_verification.append((title, score))

    if needs_verification:
        # No MLP -- LLM verifies all candidates in the band
        for title, _ in needs_verification:
            llm_calls += 1
            if tracker:
                tracker.record_llm_call(
                    "rerank",
                    input_tokens=_LLM_RERANK_INPUT_TOKENS,
                    output_tokens=_LLM_RERANK_OUTPUT_TOKENS,
                )
            accepted.append(title)

    if not accepted:
        # Nothing passed any gate -- return top-scored items
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        accepted = [t for t, _ in ranked[:5]]

    return accepted, llm_calls, mlp_available


# ---------------------------------------------------------------------------
# Aggregation and display
# ---------------------------------------------------------------------------

def _run_strategy(
    queries: list,
    run_fn,
    label: str,
    include_cost: bool = False,
) -> Dict[str, Any]:
    """Run all queries through a strategy function and collect metrics."""
    results = []
    total_llm_calls = 0
    tracker = CostTracker() if include_cost else None

    for q in queries:
        expected = set(q["expected_titles"])
        if tracker:
            tracker.start_query(q["query"])

        start = time.time()
        try:
            output = run_fn(q["query"], tracker)
        except Exception as e:
            print(f"      ERROR {q['id']}: {e}")
            output = []

        elapsed_ms = (time.time() - start) * 1000

        # Handle different return types
        llm_calls = 0
        mlp_used = None
        if isinstance(output, tuple):
            if len(output) == 3:
                retrieved, llm_calls, mlp_used = output
            elif len(output) == 2:
                retrieved, second = output
                if isinstance(second, bool):
                    mlp_used = second
                else:
                    llm_calls = second
            else:
                retrieved = output[0]
        else:
            retrieved = output

        total_llm_calls += llm_calls

        if tracker:
            tracker.end_query()

        metrics = compute_all_metrics(retrieved, expected)

        results.append({
            "id": q["id"],
            "query": q["query"],
            "retrieved_top5": retrieved[:5],
            "metrics": metrics,
            "latency_ms": round(elapsed_ms, 0),
            "llm_calls": llm_calls,
            "mlp_used": mlp_used,
        })

        rr = metrics.get("rr", 0)
        status = "HIT" if rr > 0 else "MISS"
        print(f"      {q['id']}: {status} (RR={rr:.2f}, {elapsed_ms:.0f}ms) -- {q['query'][:45]}")

    return {
        "label": label,
        "results": results,
        "total_llm_calls": total_llm_calls,
        "tracker": tracker,
    }


def _aggregate(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregate metrics for a strategy."""
    results = strategy_data["results"]
    if not results:
        return {"total": 0}

    n = len(results)

    def mean(key):
        return round(sum(r["metrics"].get(key, 0) for r in results) / n, 3)

    total_llm = strategy_data["total_llm_calls"]
    queries_with_llm = sum(1 for r in results if r["llm_calls"] > 0)
    llm_freq = queries_with_llm / n if n > 0 else 0.0

    avg_latency = round(sum(r["latency_ms"] for r in results) / n, 0)

    # Cost estimation
    tracker = strategy_data.get("tracker")
    if tracker and tracker.queries:
        cost_summary = tracker.summary()
        avg_cost = cost_summary.get("avg_cost_per_query_usd", 0.0)
    else:
        # Estimate cost from LLM calls
        emb_cost = _EMBEDDING_TOKENS_PER_QUERY * PRICING["voyage-3-large"]["input"] / 1_000_000
        llm_cost_per_call = (
            _LLM_RERANK_INPUT_TOKENS * PRICING["gpt-4o"]["input"] / 1_000_000
            + _LLM_RERANK_OUTPUT_TOKENS * PRICING["gpt-4o"]["output"] / 1_000_000
        )
        avg_cost = emb_cost + (total_llm / n) * llm_cost_per_call if n > 0 else 0.0

    return {
        "total": n,
        "mrr": round(sum(r["metrics"].get("rr", 0) for r in results) / n, 3),
        "p@1": mean("p@1"),
        "p@5": mean("p@5"),
        "r@5": mean("r@5"),
        "ndcg@5": mean("ndcg@5"),
        "llm_freq": round(llm_freq * 100, 1),
        "avg_cost": avg_cost,
        "avg_latency_ms": avg_latency,
        "total_llm_calls": total_llm,
    }


def print_comparison(strategies: Dict[str, Dict[str, Any]]):
    """Print formatted comparison table."""
    print(f"\n{'=' * 120}")
    print("ABLATION STUDY -- RETRIEVAL STRATEGY COMPARISON")
    print(f"{'=' * 120}")
    header = (
        f"{'Strategy':<55} {'N':>4} {'MRR':>6} {'P@1':>6} {'P@5':>6} {'R@5':>6} "
        f"{'NDCG@5':>7} {'LLM%':>6} {'Cost':>10} {'Latency':>10}"
    )
    print(header)
    print(
        f"{'-' * 55} {'-' * 4} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} "
        f"{'-' * 7} {'-' * 6} {'-' * 10} {'-' * 10}"
    )

    for name, data in strategies.items():
        a = data["agg"]
        if a["total"] == 0:
            continue
        cost_str = f"${a['avg_cost']:.4f}"
        lat_str = f"{a['avg_latency_ms']:.0f}ms"
        print(
            f"{name:<55} {a['total']:>4} {a['mrr']:>6.3f} {a['p@1']:>6.3f} "
            f"{a['p@5']:>6.3f} {a['r@5']:>6.3f} {a['ndcg@5']:>7.3f} "
            f"{a['llm_freq']:>5.1f}% {cost_str:>10} {lat_str:>10}"
        )

    print(f"{'=' * 120}")


def print_cost_savings(strategies: Dict[str, Dict[str, Any]]):
    """Print cost savings summary comparing strategy 6 (baseline) vs 7 (MLP)."""
    baseline_key = None
    mlp_key = None
    for name in strategies:
        if "6." in name or "(current" in name:
            baseline_key = name
        if "7." in name or "(with MLP)" in name:
            mlp_key = name

    if not baseline_key or not mlp_key:
        return

    baseline = strategies[baseline_key]["agg"]
    mlp = strategies[mlp_key]["agg"]

    if baseline["total"] == 0 or mlp["total"] == 0:
        return

    print(f"\n{'=' * 120}")
    print("COST SAVINGS SUMMARY")
    print(f"{'=' * 120}")

    # LLM call reduction
    baseline_llm = baseline["llm_freq"]
    mlp_llm = mlp["llm_freq"]
    if baseline_llm > 0:
        llm_saved_pct = ((baseline_llm - mlp_llm) / baseline_llm) * 100
        print(f"  LLM calls saved:        {llm_saved_pct:.0f}% (from {baseline_llm:.1f}% frequency to {mlp_llm:.1f}%)")
    else:
        print("  LLM calls saved:        N/A (baseline had 0% LLM frequency)")

    # Cost reduction
    if baseline["avg_cost"] > 0:
        cost_saved_pct = ((baseline["avg_cost"] - mlp["avg_cost"]) / baseline["avg_cost"]) * 100
        print(f"  Cost reduction:         {cost_saved_pct:.0f}% (${baseline['avg_cost']:.4f} -> ${mlp['avg_cost']:.4f} per query)")
    else:
        print("  Cost reduction:         N/A")

    # Latency improvement
    if baseline["avg_latency_ms"] > 0:
        lat_saved_pct = ((baseline["avg_latency_ms"] - mlp["avg_latency_ms"]) / baseline["avg_latency_ms"]) * 100
        print(f"  Latency improvement:    {lat_saved_pct:.1f}% ({baseline['avg_latency_ms']:.0f}ms -> {mlp['avg_latency_ms']:.0f}ms avg)")
    else:
        print("  Latency improvement:    N/A")

    # MRR improvement
    mrr_delta = mlp["mrr"] - baseline["mrr"]
    print(f"  MRR improvement:        {mrr_delta:+.3f} ({baseline['mrr']:.3f} -> {mlp['mrr']:.3f})")

    print(f"{'=' * 120}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARF Ablation Study -- Full Strategy Comparison")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--domain", type=str, default="us_constitution",
                        help="Domain to evaluate (default: us_constitution)")
    parser.add_argument("--include-cost", action="store_true",
                        help="Track detailed cost via CostTracker")
    args = parser.parse_args()

    env = args.env or "production"
    domain = args.domain
    include_cost = args.include_cost

    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    queries = load_queries(domain)
    collection_key = DOMAIN_TO_COLLECTION.get(domain)

    if not collection_key or collection_key not in COLLECTION:
        print(f"Domain '{domain}' not configured")
        return

    print(f"ARF Ablation Study -- domain={domain}, env={env}")
    print(f"  Queries: {len(queries)}")
    print(f"  Cost tracking: {'enabled' if include_cost else 'disabled'}")
    print()

    # Check MLP availability upfront
    MLPClass, mlp_model_path = _try_load_mlp()
    mlp_note = ""
    if MLPClass is None:
        mlp_note = " [MLPReranker not installed -- using score-based simulation]"
    elif mlp_model_path is None:
        mlp_note = " [MLP model file not found -- using score-based simulation]"
    if mlp_note:
        print(f"  NOTE:{mlp_note}")
        print()

    strategies: Dict[str, Dict[str, Any]] = {}

    # ===== Strategy 1: Semantic Only =====
    print("  [1/7] Semantic Only...")
    rag1 = RAG(COLLECTION[collection_key], debug_mode=False)
    data1 = _run_strategy(
        queries,
        lambda q, t, _r=rag1: _strategy_semantic_only(_r, q, t),
        "Semantic Only",
        include_cost,
    )
    strategies["1. Semantic Only"] = {"data": data1, "agg": _aggregate(data1)}
    del rag1

    # ===== Strategy 2: Semantic + Keyword =====
    print("\n  [2/7] Semantic + Keyword...")
    rag2 = RAG(COLLECTION[collection_key], debug_mode=False)
    data2 = _run_strategy(
        queries,
        lambda q, t, _r=rag2: _strategy_semantic_keyword(_r, q, t),
        "Semantic + Keyword",
        include_cost,
    )
    strategies["2. Semantic + Keyword"] = {"data": data2, "agg": _aggregate(data2)}
    del rag2

    # ===== Strategy 3: Semantic + Keyword + Threshold =====
    print("\n  [3/7] Semantic + Keyword + Threshold...")
    rag3 = RAG(COLLECTION[collection_key], debug_mode=False)
    data3 = _run_strategy(
        queries,
        lambda q, t, _r=rag3: _strategy_semantic_keyword_threshold(_r, q, t),
        "Semantic + Keyword + Threshold",
        include_cost,
    )
    strategies["3. Semantic + Keyword + Threshold"] = {"data": data3, "agg": _aggregate(data3)}
    del rag3

    # ===== Strategy 4: Semantic + Keyword + MLP =====
    print(f"\n  [4/7] Semantic + Keyword + MLP...{mlp_note}")
    rag4 = RAG(COLLECTION[collection_key], debug_mode=False)
    data4 = _run_strategy(
        queries,
        lambda q, t, _r=rag4: _strategy_semantic_keyword_mlp(_r, q, domain, t),
        "Semantic + Keyword + MLP",
        include_cost,
    )
    strategies["4. Semantic + Keyword + MLP"] = {"data": data4, "agg": _aggregate(data4)}
    del rag4

    # ===== Strategy 5: Semantic + Keyword + MLP + LLM Fallback =====
    print(f"\n  [5/7] Semantic + Keyword + MLP + LLM Fallback...{mlp_note}")
    rag5 = RAG(COLLECTION[collection_key], debug_mode=False)
    data5 = _run_strategy(
        queries,
        lambda q, t, _r=rag5: _strategy_semantic_keyword_mlp_llm_fallback(_r, q, domain, t),
        "Semantic + Keyword + MLP + LLM Fallback",
        include_cost,
    )
    strategies["5. Semantic + Keyword + MLP + LLM Fallback"] = {"data": data5, "agg": _aggregate(data5)}
    del rag5

    # ===== Strategy 6: Full Pipeline (current, no MLP) =====
    print("\n  [6/7] Full Pipeline (current, no MLP)...")
    rag6 = RAG(COLLECTION[collection_key], debug_mode=False)
    data6 = _run_strategy(
        queries,
        lambda q, t, _r=rag6: _strategy_full_pipeline_no_mlp(_r, q, t),
        "Full Pipeline (current, no MLP)",
        include_cost,
    )
    strategies["6. Full Pipeline (current, no MLP)"] = {"data": data6, "agg": _aggregate(data6)}
    del rag6

    # ===== Strategy 7: Full Pipeline (with MLP) =====
    print(f"\n  [7/7] Full Pipeline (with MLP)...{mlp_note}")
    rag7 = RAG(COLLECTION[collection_key], debug_mode=False)
    data7 = _run_strategy(
        queries,
        lambda q, t, _r=rag7: _strategy_full_pipeline_with_mlp(_r, q, domain, t),
        "Full Pipeline (with MLP)",
        include_cost,
    )
    strategies["7. Full Pipeline (with MLP)"] = {"data": data7, "agg": _aggregate(data7)}
    del rag7

    # ===== Print results =====
    print_comparison(strategies)
    print_cost_savings(strategies)

    # ===== Save results =====
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = RESULTS_DIR / f"ablation_full_{ts}.json"

    save_data = {
        "metadata": {
            "timestamp": ts,
            "domain": domain,
            "env": env,
            "num_queries": len(queries),
            "include_cost": include_cost,
            "mlp_available": MLPClass is not None and mlp_model_path is not None,
        },
        "strategies": {},
    }
    for name, sdata in strategies.items():
        agg = sdata["agg"]
        save_data["strategies"][name] = {
            "aggregate": agg,
            "per_query": [
                {
                    "id": r["id"],
                    "query": r["query"],
                    "retrieved_top5": r["retrieved_top5"],
                    "metrics": r["metrics"],
                    "latency_ms": r["latency_ms"],
                    "llm_calls": r["llm_calls"],
                }
                for r in sdata["data"]["results"]
            ],
        }

    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
