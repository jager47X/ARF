#!/usr/bin/env python3
"""
Baseline Measurement Script for the ARF RAG Pipeline

Runs the current threshold-based system (no MLP reranker) against the
eval dataset and measures retrieval quality, cost, and latency:

    - Precision@k, Recall@k for k=1,3,5,10
    - MRR, NDCG@k
    - Hallucination rate (via FaithfulnessEvaluator)
    - LLM reranking call frequency
    - Cost-per-query (via CostTracker)
    - Latency distribution (p50, p95, p99)

Usage:
    python benchmarks/run_baseline.py --production
    python benchmarks/run_baseline.py --production --domain us_constitution
    python benchmarks/run_baseline.py --production --eval-faithfulness

Results are saved to benchmarks/results/baseline_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path and set up services.rag import shim
sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401
from benchmarks.cost_tracker import CostTracker
from benchmarks.metrics import compute_all_metrics, mrr

logger = logging.getLogger(__name__)

BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}


def load_benchmark_queries(
    domain: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> List[dict]:
    """Load queries from a dataset file or the default benchmark file."""
    path = Path(dataset_path) if dataset_path else BENCHMARK_FILE
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    else:
        raise ValueError(f"Unrecognised format in {path}")

    if domain:
        queries = [q for q in queries if q.get("domain") == domain]

    return queries


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    arr = np.array(values)
    return float(np.percentile(arr, p))


def run_baseline(
    queries: List[dict],
    env: str = "production",
    eval_faithfulness: bool = False,
) -> Dict[str, Any]:
    """Run baseline evaluation against the live pipeline.

    Returns a comprehensive results dict.
    """
    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    tracker = CostTracker()
    per_query_results: List[Dict[str, Any]] = []
    all_rr_pairs = []
    latencies: List[float] = []
    llm_call_counts: List[int] = []
    total_candidates = 0
    total_llm_verified = 0

    # Group by domain
    by_domain: Dict[str, List[dict]] = {}
    for q in queries:
        by_domain.setdefault(q["domain"], []).append(q)

    # Per-domain RAG instances (reuse across queries)
    rag_instances: Dict[str, Any] = {}

    for domain, domain_queries in by_domain.items():
        collection_key = DOMAIN_TO_COLLECTION.get(domain)
        if not collection_key or collection_key not in COLLECTION:
            logger.warning("Skipping domain %s: collection not configured", domain)
            continue

        print(f"\n  Evaluating {domain} ({len(domain_queries)} queries)...")

        if domain not in rag_instances:
            rag_instances[domain] = RAG(COLLECTION[collection_key], debug_mode=False)
        rag = rag_instances[domain]

        for q in domain_queries:
            query_id = q["id"]
            query_text = q["query"]
            expected = set(q.get("expected_titles", []))

            tracker.start_query(query_text)
            start = time.time()

            try:
                results, current_query = rag.process_query(query_text, language="en")
                retrieved_titles = [doc.get("title", "") for doc, score in results]
                retrieved_scores = [score for doc, score in results]
            except Exception as e:
                logger.error("Query %s failed: %s", query_id, e)
                results = []
                retrieved_titles = []
                retrieved_scores = []

            elapsed = time.time() - start
            cost_data = tracker.end_query()
            latencies.append(elapsed)

            # Count LLM calls for this query
            n_llm = cost_data.get("llm_calls", 0)
            llm_call_counts.append(n_llm)
            total_candidates += len(results)

            # Rerank-type LLM calls specifically
            llm_breakdown = cost_data.get("llm_call_breakdown", {})
            n_rerank = llm_breakdown.get("rerank", 0)
            total_llm_verified += n_rerank

            # Compute retrieval metrics
            if expected:
                metrics = compute_all_metrics(retrieved_titles, expected)
                all_rr_pairs.append((retrieved_titles, expected))
            else:
                metrics = {"rr": None, "note": "no expected titles defined"}

            entry = {
                "id": query_id,
                "domain": domain,
                "query": query_text,
                "expected": list(expected),
                "retrieved_top5": retrieved_titles[:5],
                "retrieved_scores_top5": [round(s, 4) for s in retrieved_scores[:5]],
                "num_results": len(results),
                "latency_s": round(elapsed, 3),
                "llm_calls": n_llm,
                "llm_rerank_calls": n_rerank,
                "metrics": metrics,
                "cost": cost_data,
            }
            per_query_results.append(entry)

            # Print progress
            status = "HIT" if metrics.get("rr", 0) and metrics["rr"] > 0 else "MISS"
            rr_val = metrics.get("rr", "N/A")
            rr_display = f"{rr_val:.2f}" if isinstance(rr_val, (int, float)) else rr_val
            print(f"    {query_id}: {status} (RR={rr_display}, LLM={n_llm}, {elapsed:.1f}s) -- {query_text[:50]}")

    # =================================================================
    # Aggregate metrics
    # =================================================================
    scored_pairs = [(r, rel) for r, rel in all_rr_pairs]
    n_scored = len(scored_pairs)

    # Mean retrieval metrics
    mean_metrics: Dict[str, float] = {}
    if n_scored > 0:
        metric_sums: Dict[str, float] = defaultdict(float)
        for entry in per_query_results:
            m = entry.get("metrics", {})
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    metric_sums[key] += val
        mean_metrics = {k: round(v / n_scored, 4) for k, v in metric_sums.items()}

    # Latency distribution
    latency_dist = {
        "p50_s": round(_percentile(latencies, 50), 3),
        "p95_s": round(_percentile(latencies, 95), 3),
        "p99_s": round(_percentile(latencies, 99), 3),
        "mean_s": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        "min_s": round(float(np.min(latencies)), 3) if latencies else 0.0,
        "max_s": round(float(np.max(latencies)), 3) if latencies else 0.0,
    }

    # LLM call stats
    n_queries = len(per_query_results)
    llm_stats = {
        "total_llm_calls": sum(llm_call_counts),
        "avg_llm_calls_per_query": round(sum(llm_call_counts) / max(n_queries, 1), 2),
        "queries_with_llm_calls": sum(1 for c in llm_call_counts if c > 0),
        "llm_call_frequency": round(sum(1 for c in llm_call_counts if c > 0) / max(n_queries, 1), 3),
        "total_candidates_evaluated": total_candidates,
        "total_llm_rerank_calls": total_llm_verified,
        "llm_rerank_rate": round(total_llm_verified / max(total_candidates, 1), 4),
    }

    aggregate = {
        "total_queries": n_queries,
        "queries_with_expected": n_scored,
        "mrr": round(mrr(scored_pairs), 4) if scored_pairs else None,
        "mean_metrics": mean_metrics,
        "latency_distribution": latency_dist,
        "llm_stats": llm_stats,
        "cost_summary": tracker.summary(),
    }

    # =================================================================
    # Per-domain breakdown
    # =================================================================
    domain_breakdowns: Dict[str, Dict[str, Any]] = {}
    for domain in by_domain:
        domain_entries = [e for e in per_query_results if e["domain"] == domain]
        domain_scored = [e for e in domain_entries if e["metrics"].get("rr") is not None]
        if domain_scored:
            d_metric_sums: Dict[str, float] = defaultdict(float)
            for e in domain_scored:
                for k, v in e["metrics"].items():
                    if isinstance(v, (int, float)):
                        d_metric_sums[k] += v
            nd = len(domain_scored)
            domain_breakdowns[domain] = {
                "n_queries": len(domain_entries),
                "n_scored": nd,
                "mean_metrics": {k: round(v / nd, 4) for k, v in d_metric_sums.items()},
                "avg_latency_s": round(np.mean([e["latency_s"] for e in domain_entries]), 3),
                "avg_llm_calls": round(np.mean([e["llm_calls"] for e in domain_entries]), 2),
            }
        else:
            domain_breakdowns[domain] = {
                "n_queries": len(domain_entries),
                "n_scored": 0,
                "note": "no expected titles for scoring",
            }

    aggregate["per_domain"] = domain_breakdowns

    # =================================================================
    # Faithfulness evaluation (optional)
    # =================================================================
    if eval_faithfulness:
        print("\n  Running faithfulness evaluation...")
        try:
            from benchmarks.hallucination_eval import FaithfulnessEvaluator
            evaluator = FaithfulnessEvaluator()

            faith_items = []
            for entry in per_query_results:
                if entry["num_results"] == 0:
                    continue
                try:
                    domain = entry["domain"]
                    rag = rag_instances.get(domain)
                    if not rag:
                        continue

                    results, _ = rag.process_query(entry["query"], language="en")
                    if not results:
                        continue

                    doc, score = results[0]
                    summary = rag.process_summary(
                        query=entry["query"],
                        result_list=results,
                        index=0,
                        language="en",
                    )
                    source_text = doc.get("text", "") or doc.get("summary", "")
                    if summary and source_text:
                        faith_items.append({
                            "source_text": source_text,
                            "generated_summary": summary,
                            "query": entry["query"],
                        })
                except Exception as e:
                    logger.warning("Faithfulness skip for %s: %s", entry["id"], e)

            if faith_items:
                faith_results = evaluator.evaluate_batch(faith_items)
                aggregate["faithfulness"] = faith_results
                print(f"    Faithfulness rate: {faith_results['faithfulness_rate']:.1%}")
                print(f"    Hallucination rate: {faith_results['hallucination_rate']:.1%}")
                print(f"    Avg faith. score: {faith_results['avg_faithfulness_score']:.3f}")
            else:
                print("    No items eligible for faithfulness evaluation.")
        except Exception as e:
            logger.error("Faithfulness evaluation failed: %s", e)
            print(f"    Faithfulness evaluation failed: {e}")

    return {"queries": per_query_results, "aggregate": aggregate}


def save_results(results: Dict[str, Any]) -> Path:
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "baseline_report.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary."""
    agg = results.get("aggregate", {})
    print(f"\n{'='*60}")
    print("BASELINE MEASUREMENT REPORT")
    print(f"{'='*60}")
    print(f"  Total queries:         {agg.get('total_queries', 0)}")
    print(f"  With expected titles:  {agg.get('queries_with_expected', 0)}")

    if agg.get("mrr") is not None:
        print(f"  MRR:                   {agg['mrr']}")

    mean = agg.get("mean_metrics", {})
    if mean:
        print("\n  Mean Retrieval Metrics:")
        for key in sorted(mean):
            print(f"    {key:>10}: {mean[key]:.4f}")

    lat = agg.get("latency_distribution", {})
    if lat:
        print("\n  Latency Distribution:")
        print(f"    p50:   {lat.get('p50_s', 0):.3f} s")
        print(f"    p95:   {lat.get('p95_s', 0):.3f} s")
        print(f"    p99:   {lat.get('p99_s', 0):.3f} s")
        print(f"    mean:  {lat.get('mean_s', 0):.3f} s")

    llm = agg.get("llm_stats", {})
    if llm:
        print("\n  LLM Verification Stats:")
        print(f"    Total LLM calls:        {llm.get('total_llm_calls', 0)}")
        print(f"    Avg per query:           {llm.get('avg_llm_calls_per_query', 0):.2f}")
        print(f"    LLM call frequency:      {llm.get('llm_call_frequency', 0):.1%}")
        print(f"    Total rerank calls:      {llm.get('total_llm_rerank_calls', 0)}")
        print(f"    Rerank rate (per cand.): {llm.get('llm_rerank_rate', 0):.2%}")

    cost = agg.get("cost_summary", {})
    if cost.get("total_queries"):
        print("\n  Cost Analysis:")
        print(f"    Avg cost/query:   ${cost.get('avg_cost_per_query_usd', 0):.6f}")
        print(f"    Total cost:       ${cost.get('total_cost_usd', 0):.6f}")
        print(f"    Cache hit rate:   {cost.get('overall_cache_hit_rate', 0):.1%}")

    # Per-domain
    per_domain = agg.get("per_domain", {})
    if per_domain:
        print("\n  Per-Domain Breakdown:")
        for domain, info in sorted(per_domain.items()):
            n = info.get("n_queries", 0)
            dm = info.get("mean_metrics", {})
            ndcg5 = dm.get("ndcg@5", "N/A")
            print(f"    {domain:<35} queries={n}  NDCG@5={ndcg5}  avg_latency={info.get('avg_latency_s', 'N/A')}s")

    faith = agg.get("faithfulness")
    if faith:
        print("\n  Faithfulness (LLM-as-judge):")
        print(f"    Faithfulness rate:   {faith['faithfulness_rate']:.1%}")
        print(f"    Hallucination rate:  {faith['hallucination_rate']:.1%}")
        print(f"    Avg faith. score:    {faith['avg_faithfulness_score']:.3f}")

    print(f"{'='*60}\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ARF Baseline Measurement")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--domain", type=str, default=None, help="Filter to a specific domain")
    parser.add_argument("--dataset", type=str, default=None, help="Custom dataset path")
    parser.add_argument("--eval-faithfulness", action="store_true", help="Run hallucination evaluation")
    args = parser.parse_args()

    env = args.env or "production"

    queries = load_benchmark_queries(domain=args.domain, dataset_path=args.dataset)
    print(f"\nARF Baseline Measurement -- env={env}, queries={len(queries)}")

    results = run_baseline(queries, env=env, eval_faithfulness=args.eval_faithfulness)

    path = save_results(results)
    print_summary(results)
    print(f"  Full results saved to: {path}\n")


if __name__ == "__main__":
    main()
