#!/usr/bin/env python3
"""
ARF Ablation Study — Compare retrieval strategies.

Runs the same benchmark queries under different retrieval configurations:
  1. MongoDB Atlas Semantic Only — raw $vectorSearch, no reranking/alias/keyword
  2. MongoDB Atlas Hybrid — semantic + keyword/alias matching (no LLM rerank)
  3. Full ARF Pipeline — semantic + keyword/alias + LLM reranking + rephrasing + caching
  4. Full ARF Pipeline (with 50% duplicate queries) — measures cache hit latency

Usage:
    python benchmarks/run_ablation.py --production
    python benchmarks/run_ablation.py --production --domain us_constitution
"""

from __future__ import annotations

import argparse
import json
import random
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


def build_duplicate_set(queries: list, seed: int = 42) -> tuple:
    """
    Build a query set with 50% random duplicates appended.
    Returns (full_query_list, duplicate_ids_set).
    """
    rng = random.Random(seed)
    n_dup = len(queries) // 2
    duplicates = rng.sample(queries, k=min(n_dup, len(queries)))
    dup_ids = {q["id"] for q in duplicates}
    # Tag duplicates
    tagged = []
    for q in duplicates:
        tagged.append({**q, "id": q["id"] + "-DUP", "_is_duplicate": True})
    full = queries + tagged
    return full, dup_ids


def _run_strategy(rag, queries: list, mode: str) -> dict:
    """
    Run queries through a specific retrieval mode.
    mode: 'semantic' | 'hybrid' | 'full'
    """
    results = []
    tracker = CostTracker()

    for q in queries:
        is_dup = q.get("_is_duplicate", False)
        tracker.start_query(q["query"])
        start = time.time()
        expected = set(q["expected_titles"])
        retrieved = []

        try:
            if mode == "semantic":
                emb = rag.query_manager.get_embedding(q["query"])
                tracker.record_embedding_call(tokens=128)
                raw = rag.vector_search.search_main.search_similar(emb, k=10)
                retrieved = [doc.get("title", "") for doc, score in (raw or [])]

            elif mode == "hybrid":
                emb = rag.query_manager.get_embedding(q["query"])
                tracker.record_embedding_call(tokens=128)
                raw = rag.vector_search.search_main.search_similar(emb, k=10)
                sem_results = raw or []
                kw_titles = []
                if rag.keyword:
                    kw_titles = rag.keyword.find_textual(q["query"])
                title_scores = {}
                for doc, score in sem_results:
                    title = doc.get("title", "")
                    title_scores[title] = max(title_scores.get(title, 0), score)
                for kw_title in kw_titles:
                    if kw_title not in title_scores:
                        title_scores[kw_title] = 0.70
                    else:
                        title_scores[kw_title] = min(1.0, title_scores[kw_title] + 0.05)
                ranked = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)
                retrieved = [title for title, _ in ranked]

            elif mode == "full":
                pipeline_results, _ = rag.process_query(q["query"], language="en")
                retrieved = [doc.get("title", "") for doc, score in pipeline_results]
                # Check if cache was used
                if pipeline_results:
                    tracker.record_cache_hit() if is_dup else tracker.record_cache_miss()
                else:
                    tracker.record_cache_miss()

        except Exception as e:
            print(f"    ERROR {q['id']}: {e}")
            retrieved = []

        elapsed = time.time() - start
        cost = tracker.end_query()
        metrics = compute_all_metrics(retrieved, expected)

        dup_tag = " [CACHED]" if is_dup else ""
        results.append({
            "id": q["id"],
            "query": q["query"],
            "is_duplicate": is_dup,
            "retrieved_top5": retrieved[:5],
            "metrics": metrics,
            "latency_s": round(elapsed, 3),
        })

        rr = metrics.get("rr", 0)
        status = "HIT" if rr > 0 else "MISS"
        print(f"    {q['id']}: {status} (RR={rr:.2f}, {elapsed*1000:.0f}ms){dup_tag} — {q['query'][:45]}")

    return _aggregate_with_splits(results, tracker)


def _aggregate_with_splits(results: list, tracker: CostTracker = None) -> dict:
    """Aggregate metrics, splitting unique vs duplicate queries."""
    unique = [r for r in results if not r.get("is_duplicate")]
    dups = [r for r in results if r.get("is_duplicate")]
    n = len(results)

    def _agg(subset):
        if not subset:
            return {}
        nn = len(subset)
        mean = lambda key: round(sum(r["metrics"].get(key, 0) for r in subset) / nn, 3)
        all_rr = [r["metrics"].get("rr", 0) for r in subset]
        return {
            "total": nn,
            "mrr": round(sum(all_rr) / nn, 3),
            "p@1": mean("p@1"),
            "p@5": mean("p@5"),
            "r@5": mean("r@5"),
            "r@10": mean("r@10"),
            "ndcg@5": mean("ndcg@5"),
            "avg_latency_ms": round(sum(r["latency_s"] for r in subset) / nn * 1000, 0),
        }

    agg = _agg(results)
    agg["unique_queries"] = _agg(unique)
    agg["duplicate_queries"] = _agg(dups) if dups else {}
    agg["details"] = results

    if tracker:
        agg["cost"] = tracker.summary()

    return agg


def print_comparison(strategies: dict):
    print(f"\n{'='*100}")
    print("ABLATION STUDY — RETRIEVAL STRATEGY COMPARISON")
    print(f"{'='*100}")
    header = f"{'Strategy':<50} {'Queries':>7} {'MRR':>6} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'NDCG@5':>7} {'Latency':>10}"
    print(header)
    print(f"{'-'*50} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")

    for name, agg in strategies.items():
        # Main row (unique queries only for fair comparison)
        uq = agg.get("unique_queries", agg)
        lat = f"{uq['avg_latency_ms']:.0f} ms"
        total = uq.get("total", agg.get("total", 0))
        print(f"{name:<50} {total:>7} {uq['mrr']:>6.3f} {uq['p@1']:>6.3f} {uq['p@5']:>6.3f} {uq['r@5']:>6.3f} {uq['ndcg@5']:>7.3f} {lat:>10}")

        # Duplicate row if present
        dq = agg.get("duplicate_queries", {})
        if dq:
            lat_dup = f"{dq['avg_latency_ms']:.0f} ms"
            dup_label = f"  └─ {name} (cached/duplicate)"
            if len(dup_label) > 50:
                dup_label = f"  └─ cached/duplicate queries"
            print(f"{dup_label:<50} {dq['total']:>7} {dq['mrr']:>6.3f} {dq['p@1']:>6.3f} {dq['p@5']:>6.3f} {dq['r@5']:>6.3f} {dq['ndcg@5']:>7.3f} {lat_dup:>10}")

    print(f"{'='*100}")

    # Cache summary
    for name, agg in strategies.items():
        cost = agg.get("cost", {})
        if cost.get("total_queries"):
            print(f"\n  {name} — Cost & Cache:")
            print(f"    Cache hit rate:      {cost.get('overall_cache_hit_rate', 0):.1%}")
            print(f"    LLM rerank freq:     {cost.get('llm_rerank_frequency', 0):.1%}")
            print(f"    Avg LLM calls/query: {cost.get('avg_llm_calls_per_query', 0):.1f}")

    print()


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

    # Build duplicate set (50% random duplicates appended)
    queries_with_dups, dup_ids = build_duplicate_set(queries)

    print(f"ARF Ablation Study — {args.domain}")
    print(f"  Unique queries: {len(queries)}")
    print(f"  Duplicate queries: {len(queries_with_dups) - len(queries)} (50% random sample for cache latency test)")
    print(f"  Total queries per strategy: {len(queries_with_dups)}\n")

    rag = RAG(COLLECTION[collection_key], debug_mode=False)

    strategies = {}

    print("  [1/3] MongoDB Atlas Semantic Only...")
    strategies["MongoDB Atlas (Semantic Only)"] = _run_strategy(rag, queries_with_dups, "semantic")

    print(f"\n  [2/3] MongoDB Atlas Hybrid (Semantic + Keyword/Alias)...")
    strategies["MongoDB Atlas (Hybrid)"] = _run_strategy(rag, queries_with_dups, "hybrid")

    print(f"\n  [3/3] Full ARF Pipeline (with caching)...")
    strategies["Full ARF Pipeline"] = _run_strategy(rag, queries_with_dups, "full")

    print_comparison(strategies)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"ablation_{ts}.json"
    # Strip details for cleaner JSON
    save_data = {}
    for name, agg in strategies.items():
        save_data[name] = {k: v for k, v in agg.items() if k != "details"}
        save_data[name]["details"] = agg.get("details", [])
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to: {path}")


if __name__ == "__main__":
    main()
