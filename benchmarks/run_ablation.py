#!/usr/bin/env python3
"""
ARF Ablation Study — Compare retrieval strategies.

Runs the same benchmark queries under different retrieval configurations:
  1. MongoDB Atlas Semantic Only — raw $vectorSearch, no reranking/alias/keyword
  2. MongoDB Atlas Hybrid — semantic + keyword/alias matching (no LLM rerank)
  3. Full ARF Pipeline — semantic + keyword/alias + LLM reranking + rephrasing + caching

Usage:
    python benchmarks/run_ablation.py --production
    python benchmarks/run_ablation.py --production --domain us_constitution
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401

from benchmarks.metrics import compute_all_metrics, mrr
from benchmarks.cost_tracker import CostTracker

BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}


def load_queries(domain: str = None) -> list:
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    queries = data["queries"]
    if domain:
        queries = [q for q in queries if q["domain"] == domain]
    # Only queries with expected titles
    return [q for q in queries if q.get("expected_titles")]


def run_semantic_only(rag, queries: list) -> dict:
    """
    MongoDB Atlas Semantic Only — call vector search directly, skip all
    reranking, alias, keyword, caching, and rephrasing.
    """
    results = []
    tracker = CostTracker()

    for q in queries:
        tracker.start_query(q["query"])
        start = time.time()
        expected = set(q["expected_titles"])

        try:
            # Get embedding
            emb = rag.query_manager.get_embedding(q["query"])
            tracker.record_embedding_call(tokens=128)

            # Raw vector search — no threshold gates, no reranking
            raw = rag.vector_search.search_main.search_similar(emb, k=10)
            retrieved = [doc.get("title", "") for doc, score in (raw or [])]
        except Exception as e:
            print(f"    ERROR {q['id']}: {e}")
            retrieved = []

        elapsed = time.time() - start
        cost = tracker.end_query()
        metrics = compute_all_metrics(retrieved, expected)

        results.append({
            "id": q["id"],
            "query": q["query"],
            "retrieved_top5": retrieved[:5],
            "metrics": metrics,
            "latency_s": round(elapsed, 2),
        })

        rr = metrics.get("rr", 0)
        status = "HIT" if rr > 0 else "MISS"
        print(f"    {q['id']}: {status} (RR={rr:.2f}, {elapsed:.1f}s) — {q['query'][:50]}")

    return _aggregate(results)


def run_hybrid(rag, queries: list) -> dict:
    """
    MongoDB Atlas Hybrid — semantic search + keyword/alias matching.
    No LLM reranking, no rephrasing.
    """
    results = []
    tracker = CostTracker()

    for q in queries:
        tracker.start_query(q["query"])
        start = time.time()
        expected = set(q["expected_titles"])

        try:
            emb = rag.query_manager.get_embedding(q["query"])
            tracker.record_embedding_call(tokens=128)

            # Semantic search
            raw = rag.vector_search.search_main.search_similar(emb, k=10)
            sem_results = raw or []

            # Keyword/alias matching (if enabled)
            kw_titles = []
            if rag.keyword:
                kw_titles = rag.keyword.find_textual(q["query"])

            # Merge: boost semantic results that also match keywords
            title_scores = {}
            for doc, score in sem_results:
                title = doc.get("title", "")
                title_scores[title] = max(title_scores.get(title, 0), score)

            # Add keyword matches with a fixed score if not already present
            for kw_title in kw_titles:
                if kw_title not in title_scores:
                    title_scores[kw_title] = 0.70  # KEYWORD_MATCH_SCORE from config
                else:
                    # Boost existing score slightly for keyword overlap
                    title_scores[kw_title] = min(1.0, title_scores[kw_title] + 0.05)

            # Sort by score
            ranked = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)
            retrieved = [title for title, _ in ranked]

        except Exception as e:
            print(f"    ERROR {q['id']}: {e}")
            retrieved = []

        elapsed = time.time() - start
        cost = tracker.end_query()
        metrics = compute_all_metrics(retrieved, expected)

        results.append({
            "id": q["id"],
            "query": q["query"],
            "retrieved_top5": retrieved[:5],
            "metrics": metrics,
            "latency_s": round(elapsed, 2),
        })

        rr = metrics.get("rr", 0)
        status = "HIT" if rr > 0 else "MISS"
        print(f"    {q['id']}: {status} (RR={rr:.2f}, {elapsed:.1f}s) — {q['query'][:50]}")

    return _aggregate(results)


def run_full_pipeline(rag, queries: list) -> dict:
    """Full ARF Pipeline — everything enabled."""
    results = []
    tracker = CostTracker()

    for q in queries:
        tracker.start_query(q["query"])
        start = time.time()
        expected = set(q["expected_titles"])

        try:
            pipeline_results, _ = rag.process_query(q["query"], language="en")
            retrieved = [doc.get("title", "") for doc, score in pipeline_results]
        except Exception as e:
            print(f"    ERROR {q['id']}: {e}")
            retrieved = []

        elapsed = time.time() - start
        cost = tracker.end_query()
        metrics = compute_all_metrics(retrieved, expected)

        results.append({
            "id": q["id"],
            "query": q["query"],
            "retrieved_top5": retrieved[:5],
            "metrics": metrics,
            "latency_s": round(elapsed, 2),
        })

        rr = metrics.get("rr", 0)
        status = "HIT" if rr > 0 else "MISS"
        print(f"    {q['id']}: {status} (RR={rr:.2f}, {elapsed:.1f}s) — {q['query'][:50]}")

    return _aggregate(results)


def _aggregate(results: list) -> dict:
    rr_pairs = [
        ([r["retrieved_top5"][i] if i < len(r["retrieved_top5"]) else "" for i in range(10)],
         set())
        for r in results
    ]
    # Recalculate from full retrieved lists
    all_rr = []
    for r in results:
        all_rr.append(r["metrics"].get("rr", 0))

    n = len(results)
    mean = lambda key: round(sum(r["metrics"].get(key, 0) for r in results) / n, 3) if n else 0

    return {
        "total": n,
        "mrr": round(sum(all_rr) / n, 3) if n else 0,
        "p@1": mean("p@1"),
        "p@5": mean("p@5"),
        "r@5": mean("r@5"),
        "r@10": mean("r@10"),
        "ndcg@5": mean("ndcg@5"),
        "avg_latency_ms": round(sum(r["latency_s"] for r in results) / n * 1000, 0) if n else 0,
        "details": results,
    }


def print_comparison(strategies: dict):
    print(f"\n{'='*90}")
    print("ABLATION STUDY — RETRIEVAL STRATEGY COMPARISON")
    print(f"{'='*90}")
    print(f"{'Strategy':<45} {'MRR':>6} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'NDCG@5':>7} {'Latency':>10}")
    print(f"{'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")

    for name, agg in strategies.items():
        lat = f"{agg['avg_latency_ms']:.0f} ms"
        print(f"{name:<45} {agg['mrr']:>6.3f} {agg['p@1']:>6.3f} {agg['p@5']:>6.3f} {agg['r@5']:>6.3f} {agg['ndcg@5']:>7.3f} {lat:>10}")

    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="ARF Ablation Study")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--domain", type=str, default="us_constitution")
    args = parser.parse_args()

    env = args.env or "production"

    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    queries = load_queries(args.domain)
    collection_key = DOMAIN_TO_COLLECTION.get(args.domain)

    if not collection_key or collection_key not in COLLECTION:
        print(f"Domain '{args.domain}' not configured")
        return

    print(f"ARF Ablation Study — {args.domain}, {len(queries)} queries\n")

    rag = RAG(COLLECTION[collection_key], debug_mode=False)

    strategies = {}

    print("  [1/3] MongoDB Atlas Semantic Only...")
    strategies["MongoDB Atlas (Semantic Only)"] = run_semantic_only(rag, queries)

    print(f"\n  [2/3] MongoDB Atlas Hybrid (Semantic + Keyword/Alias)...")
    strategies["MongoDB Atlas (Hybrid)"] = run_hybrid(rag, queries)

    print(f"\n  [3/3] Full ARF Pipeline...")
    strategies["Full ARF Pipeline"] = run_full_pipeline(rag, queries)

    print_comparison(strategies)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"ablation_{ts}.json"
    with open(path, "w") as f:
        json.dump(strategies, f, indent=2, default=str)
    print(f"  Results saved to: {path}")


if __name__ == "__main__":
    main()
