#!/usr/bin/env python3
"""
ARF Cost Analysis — Simulate 100→1000 queries to measure cost at scale.

Takes the 100-query US Constitution test set, runs 100 unique queries,
then runs 900 similar queries (rephrased variants) to simulate production
traffic where ~90% of queries are variations of previously seen questions.

Measures: embedding calls, LLM calls, cache hit rate, latency, API costs.

Usage:
    python benchmarks/run_cost_analysis.py --production
    python benchmarks/run_cost_analysis.py --production --total 500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401

CSV_FILE = Path(__file__).parent / "us_constitution_test_set.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# Voyage AI pricing (per 1M tokens)
VOYAGE_PRICE_PER_1M = 0.06  # voyage-3-large input
# OpenAI pricing (per 1M tokens) — for reranking/rephrasing
OPENAI_INPUT_PER_1M = 2.50   # gpt-4o input
OPENAI_OUTPUT_PER_1M = 10.00  # gpt-4o output
# Avg tokens per query embedding
AVG_QUERY_TOKENS = 25
# Avg tokens per LLM rerank call
AVG_RERANK_INPUT_TOKENS = 800
AVG_RERANK_OUTPUT_TOKENS = 50
# Avg tokens per rephrase call
AVG_REPHRASE_INPUT_TOKENS = 200
AVG_REPHRASE_OUTPUT_TOKENS = 30


def load_csv_queries() -> list:
    queries = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query": row["Question"].strip(),
                "difficulty": int(row.get("Case", 1)),
                "related": row.get("Related to", ""),
            })
    return queries


REPHRASE_TEMPLATES = [
    lambda q: q.replace("What", "How").replace("?", "?") if q.startswith("What") else q + " explained",
    lambda q: q.replace("does", "do").replace("is", "are") if "does" in q or "is " in q else "tell me about " + q,
    lambda q: q.rstrip("?") + " in detail?",
    lambda q: q.replace("rights", "freedoms").replace("right", "freedom") if "right" in q.lower() else q + " law",
    lambda q: q.replace("Amendment", "amendment").replace("Article", "article") if "Amendment" in q else q,
    lambda q: "explain " + q.lower().rstrip("?"),
    lambda q: q.replace("protect", "guarantee").replace("say", "state") if "protect" in q or "say" in q else q + " overview",
    lambda q: q.replace("government", "gov").replace("federal", "national") if "government" in q else q,
    lambda q: "what are the " + q.lower().rstrip("?") + "?",
]


def generate_similar(query: str, variant: int, rng: random.Random) -> str:
    """Generate a slightly different version of a query."""
    template = REPHRASE_TEMPLATES[variant % len(REPHRASE_TEMPLATES)]
    return template(query)


def main():
    parser = argparse.ArgumentParser(description="ARF Cost Analysis")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--total", type=int, default=1000, help="Total queries to simulate")
    args = parser.parse_args()

    env = args.env or "production"

    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    base_queries = load_csv_queries()
    total = args.total
    n_unique = len(base_queries)
    n_similar = total - n_unique

    print(f"ARF Cost Analysis — {total} total queries")
    print(f"  Unique (cold): {n_unique}")
    print(f"  Similar (should hit cache): {n_similar}")
    print()

    rag = RAG(COLLECTION["US_CONSTITUTION_SET"], debug_mode=False)

    # --- Phase 1: Run unique queries ---
    print(f"  Phase 1: Running {n_unique} unique queries...")
    cold_times = []
    cold_results = []
    embedding_calls = 0
    llm_rerank_calls = 0
    llm_rephrase_calls = 0
    cache_hits = 0
    cache_misses = 0

    for i, q in enumerate(base_queries):
        t0 = time.time()
        try:
            results, _ = rag.process_query(q["query"], language="en")
            n_results = len(results)
        except Exception as e:
            n_results = 0
        elapsed = (time.time() - t0) * 1000
        cold_times.append(elapsed)
        cold_results.append(n_results)
        embedding_calls += 1  # always 1 embedding call
        cache_misses += 1  # first time = miss

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_unique} done... avg={sum(cold_times)/len(cold_times):.0f}ms")

    avg_cold = sum(cold_times) / len(cold_times)
    print(f"  Phase 1 complete: avg={avg_cold:.0f}ms, total={sum(cold_times)/1000:.1f}s")

    # --- Phase 2: Run similar queries ---
    print(f"\n  Phase 2: Running {n_similar} similar queries...")
    rng = random.Random(42)
    warm_times = []

    for i in range(n_similar):
        base_q = base_queries[i % n_unique]
        variant = i // n_unique
        similar_q = generate_similar(base_q["query"], variant, rng)

        t0 = time.time()
        try:
            results, _ = rag.process_query(similar_q, language="en")
            n_results = len(results)
        except Exception as e:
            n_results = 0
        elapsed = (time.time() - t0) * 1000
        warm_times.append(elapsed)
        embedding_calls += 1

        # Heuristic: if fast (<2s), likely cache hit
        if elapsed < 2000:
            cache_hits += 1
        else:
            cache_misses += 1
            # If slow and went through rephrase, count LLM calls
            if elapsed > 5000:
                llm_rephrase_calls += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_similar} done... avg={sum(warm_times)/len(warm_times):.0f}ms")

    avg_warm = sum(warm_times) / len(warm_times) if warm_times else 0

    # --- Cost calculations ---
    total_embedding_cost = embedding_calls * AVG_QUERY_TOKENS * VOYAGE_PRICE_PER_1M / 1_000_000
    total_rerank_cost = llm_rerank_calls * (
        AVG_RERANK_INPUT_TOKENS * OPENAI_INPUT_PER_1M / 1_000_000 +
        AVG_RERANK_OUTPUT_TOKENS * OPENAI_OUTPUT_PER_1M / 1_000_000
    )
    total_rephrase_cost = llm_rephrase_calls * (
        AVG_REPHRASE_INPUT_TOKENS * OPENAI_INPUT_PER_1M / 1_000_000 +
        AVG_REPHRASE_OUTPUT_TOKENS * OPENAI_OUTPUT_PER_1M / 1_000_000
    )
    total_api_cost = total_embedding_cost + total_rerank_cost + total_rephrase_cost
    cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0

    all_times = cold_times + warm_times

    # --- Print results ---
    print(f"\n{'='*70}")
    print("COST ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"  Total queries:              {total}")
    print(f"  Unique (cold):              {n_unique}")
    print(f"  Similar (warm):             {n_similar}")
    print(f"")
    print(f"  Embedding calls:            {embedding_calls}")
    print(f"  Embedding calls/query:      {embedding_calls/total:.2f}")
    print(f"  LLM rerank calls:           {llm_rerank_calls}")
    print(f"  LLM rephrase calls:         {llm_rephrase_calls}")
    print(f"  LLM calls/query:            {(llm_rerank_calls + llm_rephrase_calls)/total:.3f}")
    print(f"  LLM rerank frequency:       {llm_rerank_calls/total:.1%}")
    print(f"")
    print(f"  Cache hits:                 {cache_hits}")
    print(f"  Cache misses:               {cache_misses}")
    print(f"  Cache hit rate:             {cache_hit_rate:.1%}")
    print(f"")
    print(f"  Avg latency (cold):         {avg_cold:.0f} ms")
    print(f"  Avg latency (similar):      {avg_warm:.0f} ms")
    print(f"  Avg latency (overall):      {sum(all_times)/len(all_times):.0f} ms")
    print(f"  P50 latency:                {sorted(all_times)[len(all_times)//2]:.0f} ms")
    print(f"  P99 latency:                {sorted(all_times)[int(len(all_times)*0.99)]:.0f} ms")
    print(f"")
    print(f"  API Costs:")
    print(f"    Embedding (Voyage):       ${total_embedding_cost:.6f}")
    print(f"    Reranking (OpenAI):       ${total_rerank_cost:.6f}")
    print(f"    Rephrasing (OpenAI):      ${total_rephrase_cost:.6f}")
    print(f"    Total API cost:           ${total_api_cost:.6f}")
    print(f"    Cost per query:           ${total_api_cost/total:.8f}")
    print(f"{'='*70}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"cost_analysis_{ts}.json"
    result = {
        "total_queries": total,
        "unique_queries": n_unique,
        "similar_queries": n_similar,
        "embedding_calls": embedding_calls,
        "llm_rerank_calls": llm_rerank_calls,
        "llm_rephrase_calls": llm_rephrase_calls,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": round(cache_hit_rate, 3),
        "avg_latency_cold_ms": round(avg_cold, 0),
        "avg_latency_warm_ms": round(avg_warm, 0),
        "avg_latency_overall_ms": round(sum(all_times) / len(all_times), 0),
        "p50_latency_ms": round(sorted(all_times)[len(all_times) // 2], 0),
        "p99_latency_ms": round(sorted(all_times)[int(len(all_times) * 0.99)], 0),
        "total_api_cost_usd": round(total_api_cost, 6),
        "cost_per_query_usd": round(total_api_cost / total, 8),
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to: {path}")


if __name__ == "__main__":
    main()
