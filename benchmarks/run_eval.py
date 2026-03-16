#!/usr/bin/env python3
"""
ARF Benchmark Runner

Runs the benchmark query set against live ARF and computes retrieval
metrics (Precision@k, Recall@k, MRR, NDCG), cost estimates, and
optionally hallucination/faithfulness scores.

Usage:
    # Full evaluation (requires API keys + MongoDB)
    python benchmarks/run_eval.py --production

    # Specific domain only
    python benchmarks/run_eval.py --production --domain us_constitution

    # With hallucination evaluation
    python benchmarks/run_eval.py --production --eval-faithfulness

    # With ablation (compare strategies)
    python benchmarks/run_eval.py --production --ablation

    # Dry run (just validate benchmark queries, no API calls)
    python benchmarks/run_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import compute_all_metrics, mrr
from benchmarks.cost_tracker import CostTracker


BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Domain -> collection key mapping
DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}


def load_benchmark_queries(domain: str = None) -> list:
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    queries = data["queries"]
    if domain:
        queries = [q for q in queries if q["domain"] == domain]
    return queries


def run_dry(queries: list) -> None:
    """Validate benchmark queries without making API calls."""
    print(f"\n{'='*60}")
    print(f"DRY RUN — validating {len(queries)} benchmark queries")
    print(f"{'='*60}\n")

    by_domain = {}
    for q in queries:
        by_domain.setdefault(q["domain"], []).append(q)

    for domain, qs in sorted(by_domain.items()):
        has_expected = sum(1 for q in qs if q.get("expected_titles"))
        print(f"  {domain}: {len(qs)} queries ({has_expected} with expected results)")

    print(f"\n  Total: {len(queries)} queries across {len(by_domain)} domains")
    print("  Benchmark file is valid.\n")


def run_evaluation(
    queries: list,
    env: str = "production",
    eval_faithfulness: bool = False,
    ablation: bool = False,
) -> dict:
    """
    Run full evaluation against live ARF.

    Returns a results dict with per-query metrics and aggregates.
    """
    # Late imports — these require API keys and MongoDB
    from config import COLLECTION, load_environment
    load_environment(env)

    from RAG_interface import RAG

    tracker = CostTracker()
    all_results = []
    all_rr_pairs = []

    # Group queries by domain for efficient collection reuse
    by_domain = {}
    for q in queries:
        by_domain.setdefault(q["domain"], []).append(q)

    for domain, domain_queries in by_domain.items():
        collection_key = DOMAIN_TO_COLLECTION.get(domain)
        if not collection_key or collection_key not in COLLECTION:
            print(f"  SKIP {domain}: collection not configured")
            continue

        print(f"\n  Evaluating {domain} ({len(domain_queries)} queries)...")
        rag = RAG(COLLECTION[collection_key], debug_mode=False)

        for q in domain_queries:
            query_id = q["id"]
            query_text = q["query"]
            expected = set(q.get("expected_titles", []))

            tracker.start_query(query_text)
            start = time.time()

            try:
                results, _ = rag.process_query(query_text, language="en")
                retrieved_titles = [
                    doc.get("title", "") for doc, score in results
                ]
            except Exception as e:
                print(f"    ERROR {query_id}: {e}")
                retrieved_titles = []
                results = []

            elapsed = time.time() - start
            cost_data = tracker.end_query()

            # Compute metrics (only if we have expected results)
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
                "num_results": len(results),
                "latency_s": round(elapsed, 2),
                "metrics": metrics,
                "cost": cost_data,
            }
            all_results.append(entry)

            # Print progress
            status = "HIT" if metrics.get("rr", 0) and metrics["rr"] > 0 else "MISS"
            rr_val = metrics.get("rr", "N/A")
            rr_display = f"{rr_val:.2f}" if isinstance(rr_val, float) else rr_val
            print(f"    {query_id}: {status} (RR={rr_display}, {elapsed:.1f}s) — {query_text[:50]}")

    # Aggregate metrics
    scored_pairs = [(r, rel) for r, rel in all_rr_pairs]
    aggregate = {
        "total_queries": len(all_results),
        "queries_with_expected": len(scored_pairs),
        "mrr": round(mrr(scored_pairs), 3) if scored_pairs else None,
        "cost_summary": tracker.summary(),
    }

    # Compute mean metrics across scored queries
    if scored_pairs:
        from collections import defaultdict
        metric_sums = defaultdict(float)
        for entry in all_results:
            m = entry.get("metrics", {})
            for key, val in m.items():
                if isinstance(val, (int, float)):
                    metric_sums[key] += val
        n = len(scored_pairs)
        aggregate["mean_metrics"] = {k: round(v / n, 3) for k, v in metric_sums.items()}

    # Faithfulness evaluation
    if eval_faithfulness:
        print("\n  Running faithfulness evaluation...")
        try:
            from benchmarks.hallucination_eval import FaithfulnessEvaluator
            evaluator = FaithfulnessEvaluator()

            faith_items = []
            for entry in all_results:
                if entry["num_results"] > 0:
                    # Get summary for the top result
                    try:
                        domain = entry["domain"]
                        collection_key = DOMAIN_TO_COLLECTION.get(domain)
                        rag = RAG(COLLECTION[collection_key], debug_mode=False)
                        results, _ = rag.process_query(entry["query"], language="en")
                        if results:
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
                        print(f"    Faithfulness skip: {e}")

            if faith_items:
                faith_results = evaluator.evaluate_batch(faith_items)
                aggregate["faithfulness"] = faith_results
                print(f"    Faithfulness rate: {faith_results['faithfulness_rate']}")
                print(f"    Hallucination rate: {faith_results['hallucination_rate']}")
        except Exception as e:
            print(f"    Faithfulness evaluation failed: {e}")

    return {"queries": all_results, "aggregate": aggregate}


def save_results(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def print_summary(results: dict) -> None:
    agg = results.get("aggregate", {})
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total queries:        {agg.get('total_queries', 0)}")
    print(f"  With expected titles: {agg.get('queries_with_expected', 0)}")

    if agg.get("mrr") is not None:
        print(f"  MRR:                  {agg['mrr']}")

    mean = agg.get("mean_metrics", {})
    if mean:
        print(f"\n  Mean Metrics:")
        for key in sorted(mean):
            print(f"    {key:>10}: {mean[key]:.3f}")

    cost = agg.get("cost_summary", {})
    if cost.get("total_queries"):
        print(f"\n  Cost Analysis:")
        print(f"    Avg latency:        {cost.get('avg_latency_ms', 0):.0f} ms")
        print(f"    Avg cost/query:     ${cost.get('avg_cost_per_query_usd', 0):.6f}")
        print(f"    Total cost:         ${cost.get('total_cost_usd', 0):.6f}")
        print(f"    Cache hit rate:     {cost.get('overall_cache_hit_rate', 0):.1%}")
        print(f"    LLM rerank freq:    {cost.get('llm_rerank_frequency', 0):.1%}")

    faith = agg.get("faithfulness")
    if faith:
        print(f"\n  Faithfulness (LLM-as-judge):")
        print(f"    Faithfulness rate:  {faith['faithfulness_rate']:.1%}")
        print(f"    Hallucination rate: {faith['hallucination_rate']:.1%}")
        print(f"    Avg faith. score:   {faith['avg_faithfulness_score']:.3f}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ARF Benchmark Runner")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--domain", type=str, default=None, help="Filter to a specific domain")
    parser.add_argument("--eval-faithfulness", action="store_true", help="Run hallucination evaluation")
    parser.add_argument("--ablation", action="store_true", help="Compare retrieval strategies")
    parser.add_argument("--dry-run", action="store_true", help="Validate queries without API calls")
    args = parser.parse_args()

    queries = load_benchmark_queries(args.domain)

    if args.dry_run:
        run_dry(queries)
        return

    env = args.env or "production"
    print(f"ARF Benchmark — env={env}, queries={len(queries)}")

    results = run_evaluation(
        queries, env=env,
        eval_faithfulness=args.eval_faithfulness,
        ablation=args.ablation,
    )

    path = save_results(results)
    print_summary(results)
    print(f"  Full results saved to: {path}")


if __name__ == "__main__":
    main()
