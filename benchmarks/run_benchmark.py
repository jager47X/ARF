#!/usr/bin/env python3
"""
ARF Benchmark — Compare retrieval strategies in isolated environments.

Each strategy runs in its own isolated context:
  1. MongoDB Atlas (Semantic Only) — direct $vectorSearch, fresh embedding per query
  2. MongoDB Atlas (Hybrid) — $vectorSearch + keyword matching, no caching
  3. Full ARF Pipeline — first pass (cold, builds cache)
  4. Full ARF Pipeline — second pass with similar queries (~1 word changed)
     to demonstrate cache speed + accuracy improvement over time

Usage:
    python benchmarks/run_benchmark.py --production
    python benchmarks/run_benchmark.py --production --domain us_constitution
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

BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}

# Slightly rephrased queries (~1 word changed, ~90% embedding similarity)
SIMILAR_MAP = {
    "What does the 14th Amendment say about equal protection?":
        "What does the 14th Amendment mention about equal protection?",
    "freedom of speech":
        "freedom of expression",
    "right to bear arms":
        "right to carry arms",
    "unreasonable search and seizure":
        "unreasonable search and arrest",
    "double jeopardy due process":
        "double jeopardy fair process",
    "powers not delegated to the federal government":
        "powers not granted to the federal government",
    "abolition of slavery":
        "elimination of slavery",
    "supremacy clause federal law preemption":
        "supremacy clause federal law priority",
    "congressional power to tax and regulate commerce":
        "congressional authority to tax and regulate commerce",
    "Article 1 Section 8":
        "Article I Section 8",
    "presidential executive power":
        "presidential executive authority",
    "14th amendment":
        "fourteenth amendment",
    "first amendment rights":
        "first amendment freedoms",
    "immigration naturalization law":
        "immigration naturalization statute",
    "citizen rights privileges immunities":
        "citizen rights privileges protections",
}


def load_queries(domain: str = None) -> list:
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    queries = data["queries"]
    if domain:
        queries = [q for q in queries if q["domain"] == domain]
    return [q for q in queries if q.get("expected_titles")]


def pick_similar(queries: list, seed: int = 42) -> list:
    rng = random.Random(seed)
    n = max(1, len(queries) // 2)
    picked = rng.sample(queries, k=min(n, len(queries)))
    return [{
        **q,
        "id": q["id"] + "-SIM",
        "query": SIMILAR_MAP.get(q["query"], q["query"] + " law"),
        "original_query": q["query"],
        "_is_similar": True,
    } for q in picked]


def _run_and_collect(queries, run_fn, label=""):
    """Run queries through a function, collect results with metrics."""
    results = []
    for q in queries:
        is_sim = q.get("_is_similar", False)
        start = time.time()
        expected = set(q["expected_titles"])

        try:
            retrieved = run_fn(q["query"])
        except Exception as e:
            print(f"    ERROR {q['id']}: {e}")
            retrieved = []

        elapsed_ms = (time.time() - start) * 1000
        metrics = compute_all_metrics(retrieved, expected)

        results.append({
            "id": q["id"],
            "query": q["query"],
            "is_similar": is_sim,
            "retrieved_top5": retrieved[:5],
            "metrics": metrics,
            "latency_ms": round(elapsed_ms, 0),
        })

        rr = metrics.get("rr", 0)
        status = "HIT" if rr > 0 else "MISS"
        tag = " [SIMILAR]" if is_sim else ""
        print(f"    {q['id']}: {status} (RR={rr:.2f}, {elapsed_ms:.0f}ms){tag} — {q['query'][:45]}")

    return results


def _agg(results):
    if not results:
        return {}
    n = len(results)
    all_rr = [r["metrics"].get("rr", 0) for r in results]
    def mean(key):
        return round(sum(r["metrics"].get(key, 0) for r in results) / n, 3)
    return {
        "total": n,
        "mrr": round(sum(all_rr) / n, 3),
        "p@1": mean("p@1"),
        "p@5": mean("p@5"),
        "r@5": mean("r@5"),
        "r@10": mean("r@10"),
        "ndcg@5": mean("ndcg@5"),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 0),
    }


def print_comparison(strategies: dict):
    print(f"\n{'='*105}")
    print("BENCHMARK — RETRIEVAL STRATEGY COMPARISON")
    print(f"{'='*105}")
    header = f"{'Strategy':<55} {'N':>4} {'MRR':>6} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'NDCG@5':>7} {'Latency':>10}"
    print(header)
    print(f"{'-'*55} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")

    for name, data in strategies.items():
        a = data["agg"]
        lat = f"{a['avg_latency_ms']:.0f} ms"
        print(f"{name:<55} {a['total']:>4} {a['mrr']:>6.3f} {a['p@1']:>6.3f} {a['p@5']:>6.3f} {a['r@5']:>6.3f} {a['ndcg@5']:>7.3f} {lat:>10}")

    print(f"{'='*105}\n")


def main():
    parser = argparse.ArgumentParser(description="ARF Benchmark")
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
    similar = pick_similar(queries)
    collection_key = DOMAIN_TO_COLLECTION.get(args.domain)

    if not collection_key or collection_key not in COLLECTION:
        print(f"Domain '{args.domain}' not configured")
        return

    print(f"ARF Benchmark — {args.domain}")
    print(f"  Unique queries: {len(queries)}")
    print(f"  Similar queries: {len(similar)} (50% random pick, ~1 word changed)")
    print()

    strategies = {}

    # ===== 1) MongoDB Atlas Semantic Only — isolated RAG instance =====
    print("  [1/4] MongoDB Atlas (Semantic Only) — isolated, no cache...")
    rag_sem = RAG(COLLECTION[collection_key], debug_mode=False)

    def semantic_only(query, _rag=rag_sem):
        emb = _rag.query_manager.get_embedding(query)
        raw = _rag.vector_search.search_main.search_similar(emb, k=10)
        return [doc.get("title", "") for doc, score in (raw or [])]

    sem_results = _run_and_collect(queries, semantic_only)
    strategies["MongoDB Atlas (Semantic Only)"] = {"agg": _agg(sem_results), "details": sem_results}
    del rag_sem  # tear down

    # ===== 2) MongoDB Atlas Hybrid — isolated RAG instance =====
    print("\n  [2/4] MongoDB Atlas (Hybrid) — isolated, no cache...")
    rag_hyb = RAG(COLLECTION[collection_key], debug_mode=False)

    def hybrid(query, _rag=rag_hyb):
        emb = _rag.query_manager.get_embedding(query)
        raw = _rag.vector_search.search_main.search_similar(emb, k=10)
        sem = raw or []
        kw = _rag.keyword.find_textual(query) if _rag.keyword else []
        scores = {}
        for doc, score in sem:
            t = doc.get("title", "")
            scores[t] = max(scores.get(t, 0), score)
        for t in kw:
            if t not in scores:
                scores[t] = 0.70
            else:
                scores[t] = min(1.0, scores[t] + 0.05)
        return [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    hyb_results = _run_and_collect(queries, hybrid)
    strategies["MongoDB Atlas (Hybrid)"] = {"agg": _agg(hyb_results), "details": hyb_results}
    del rag_hyb  # tear down

    # ===== 3) Full ARF Pipeline — unique queries =====
    print("\n  [3/4] Full ARF Pipeline — unique queries...")
    rag_full = RAG(COLLECTION[collection_key], debug_mode=False)

    def full_pipeline(query):
        results, _ = rag_full.process_query(query, language="en")
        return [doc.get("title", "") for doc, score in results]

    full_results = _run_and_collect(queries, full_pipeline)
    strategies["Full ARF Pipeline"] = {"agg": _agg(full_results), "details": full_results}

    # ===== 4) Full ARF Pipeline — similar queries (test cache) =====
    # Uses SAME rag_full instance so cache from step 3 is available
    print("\n  [4/4] Full ARF Pipeline — similar queries (should leverage cache)...")
    sim_results = _run_and_collect(similar, full_pipeline)
    strategies["Full ARF Pipeline (similar queries)"] = {"agg": _agg(sim_results), "details": sim_results}

    print_comparison(strategies)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"benchmark_{ts}.json"
    save = {name: {k: v for k, v in d.items() if k != "details"} for name, d in strategies.items()}
    with open(path, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  Results saved to: {path}")


if __name__ == "__main__":
    main()
