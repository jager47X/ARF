#!/usr/bin/env python3
"""
ARF Cost Analysis — Measure real API calls across cold, cached, and similar queries.

Instruments every external API call (Voyage embed, OpenAI chat, OpenAI moderation)
and measures latency, cache hit rate, and cost. Runs a manageable sample then
extrapolates to 1000 queries.

Usage:
    python benchmarks/run_cost_analysis.py --production
    python benchmarks/run_cost_analysis.py --production --sample 20
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

# Pricing (USD per 1M tokens)
VOYAGE_PRICE = 0.06        # voyage-3-large input
OPENAI_INPUT = 2.50        # gpt-4o input
OPENAI_OUTPUT = 10.00      # gpt-4o output
MODERATION_PRICE = 0.001   # per call (approx)


SIMILAR_TRANSFORMS = [
    lambda q: q.replace("What", "Which") if q.startswith("What") else "explain " + q.lower().rstrip("?"),
    lambda q: q.rstrip("?") + " in the US?",
    lambda q: q.replace("rights", "freedoms").replace("right", "freedom") if "right" in q.lower() else q + " overview",
    lambda q: q.replace("does", "did") if "does" in q else q.replace("is", "was") if " is " in q else q + " today",
    lambda q: q.replace("Amendment", "amendment") if "Amendment" in q else q.replace("Article", "article") if "Article" in q else q,
    lambda q: "tell me about " + q.lower().rstrip("?"),
    lambda q: q.replace("protect", "guarantee").replace("say", "state") if "protect" in q or "say" in q else q + " explained",
    lambda q: q.replace("government", "state").replace("federal", "national") if "government" in q else q,
    lambda q: q.replace("the ", "a ", 1) if q.startswith("What") else "how " + q.lower().rstrip("?"),
]


def load_csv():
    queries = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries.append(row["Question"].strip())
    return queries


def instrument_rag(rag):
    """Patch all external API calls to count them."""
    import rag_dependencies.ai_service as ai_mod

    counters = {
        "voyage_calls": 0,
        "voyage_texts": 0,
        "voyage_tokens_est": 0,
        "openai_chat": 0,
        "openai_input_tokens_est": 0,
        "openai_output_tokens_est": 0,
        "moderation": 0,
    }

    # Patch Voyage
    orig_voyage = rag.query_manager.openAI.emb_backend.embed
    def p_voyage(texts):
        counters["voyage_calls"] += 1
        counters["voyage_texts"] += len(texts)
        counters["voyage_tokens_est"] += sum(len(t.split()) * 2 for t in texts)  # rough token est
        return orig_voyage(texts)
    rag.query_manager.openAI.emb_backend.embed = p_voyage

    # Patch OpenAI chat
    orig_chat = ai_mod.openai_client.chat.completions.create
    def p_chat(*a, **kw):
        counters["openai_chat"] += 1
        msgs = kw.get("messages", [])
        counters["openai_input_tokens_est"] += sum(len(m.get("content", "").split()) * 2 for m in msgs)
        counters["openai_output_tokens_est"] += 100  # rough estimate
        return orig_chat(*a, **kw)
    ai_mod.openai_client.chat.completions.create = p_chat

    # Patch moderation
    orig_mod = ai_mod.openai_client.moderations.create
    def p_mod(*a, **kw):
        counters["moderation"] += 1
        return orig_mod(*a, **kw)
    ai_mod.openai_client.moderations.create = p_mod

    return counters


def run_phase(rag, queries, counters, label):
    """Run queries, track per-query stats."""
    results = []
    for i, q in enumerate(queries):
        # Snapshot counters before
        before = dict(counters)
        t0 = time.time()

        try:
            r, _ = rag.process_query(q, language="en")
            n = len(r)
        except Exception:
            n = 0

        elapsed = (time.time() - t0) * 1000

        # Diff counters
        voyage_this = counters["voyage_calls"] - before["voyage_calls"]
        voyage_texts_this = counters["voyage_texts"] - before["voyage_texts"]
        openai_this = counters["openai_chat"] - before["openai_chat"]
        mod_this = counters["moderation"] - before["moderation"]
        is_cache_hit = (voyage_this == 0 and openai_this == 0 and mod_this == 0)

        results.append({
            "query": q,
            "results": n,
            "latency_ms": round(elapsed),
            "voyage_calls": voyage_this,
            "voyage_texts": voyage_texts_this,
            "openai_calls": openai_this,
            "moderation_calls": mod_this,
            "cache_hit": is_cache_hit,
        })

        tag = "CACHE" if is_cache_hit else f"V={voyage_this} O={openai_this} M={mod_this}"
        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"    {label} {i+1}/{len(queries)}: {elapsed:.0f}ms [{tag}] — {q[:40]}")

    return results


def compute_cost(counters):
    voyage_cost = counters["voyage_tokens_est"] * VOYAGE_PRICE / 1_000_000
    openai_cost = (
        counters["openai_input_tokens_est"] * OPENAI_INPUT / 1_000_000 +
        counters["openai_output_tokens_est"] * OPENAI_OUTPUT / 1_000_000
    )
    mod_cost = counters["moderation"] * MODERATION_PRICE
    return {
        "voyage_usd": round(voyage_cost, 6),
        "openai_usd": round(openai_cost, 6),
        "moderation_usd": round(mod_cost, 6),
        "total_usd": round(voyage_cost + openai_cost + mod_cost, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="ARF Cost Analysis")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--sample", type=int, default=20, help="Number of unique queries to run (default 20)")
    args = parser.parse_args()

    env = args.env or "production"

    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    all_queries = load_csv()
    rng = random.Random(42)
    sample = rng.sample(all_queries, min(args.sample, len(all_queries)))

    # Generate similar queries (9x the sample to simulate 1:9 ratio)
    similar = []
    for i, q in enumerate(sample):
        for j in range(9):
            transform = SIMILAR_TRANSFORMS[(i + j) % len(SIMILAR_TRANSFORMS)]
            similar.append(transform(q))

    print("ARF Cost Analysis")
    print(f"  Unique queries (cold): {len(sample)}")
    print(f"  Similar queries (warm): {len(similar)}")
    print(f"  Total: {len(sample) + len(similar)}")
    print()

    rag = RAG(COLLECTION["US_CONSTITUTION_SET"], debug_mode=False)
    counters = instrument_rag(rag)

    # Phase 1: Cold queries
    print(f"  Phase 1: {len(sample)} unique (cold) queries...")
    cold_results = run_phase(rag, sample, counters, "COLD")

    # Phase 2: Similar queries
    print(f"\n  Phase 2: {len(similar)} similar queries...")
    warm_results = run_phase(rag, similar, counters, "WARM")

    # Stats
    cold_hits = sum(1 for r in cold_results if r["cache_hit"])
    warm_hits = sum(1 for r in warm_results if r["cache_hit"])
    cold_latencies = [r["latency_ms"] for r in cold_results]
    warm_latencies = [r["latency_ms"] for r in warm_results]
    all_latencies = cold_latencies + warm_latencies

    cold_voyage = sum(r["voyage_calls"] for r in cold_results)
    cold_openai = sum(r["openai_calls"] for r in cold_results)
    cold_mod = sum(r["moderation_calls"] for r in cold_results)
    warm_voyage = sum(r["voyage_calls"] for r in warm_results)
    warm_openai = sum(r["openai_calls"] for r in warm_results)
    warm_mod = sum(r["moderation_calls"] for r in warm_results)

    total_cost = compute_cost(counters)
    n_total = len(cold_results) + len(warm_results)

    print(f"\n{'='*70}")
    print("COST ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"  Queries:          {n_total} total ({len(sample)} cold + {len(similar)} similar)")
    print()
    print(f"  --- COLD ({len(sample)} unique queries) ---")
    print(f"  Voyage embed calls:     {cold_voyage} ({cold_voyage/len(sample):.1f}/query)")
    print(f"  OpenAI chat calls:      {cold_openai} ({cold_openai/len(sample):.2f}/query)")
    print(f"  Moderation calls:       {cold_mod} ({cold_mod/len(sample):.2f}/query)")
    print(f"  Cache hits:             {cold_hits}/{len(sample)} ({cold_hits/len(sample):.0%})")
    print(f"  Avg latency:            {sum(cold_latencies)/len(cold_latencies):.0f} ms")
    print()
    print(f"  --- SIMILAR ({len(similar)} rephrased queries) ---")
    print(f"  Voyage embed calls:     {warm_voyage} ({warm_voyage/len(similar):.2f}/query)")
    print(f"  OpenAI chat calls:      {warm_openai} ({warm_openai/len(similar):.2f}/query)")
    print(f"  Moderation calls:       {warm_mod} ({warm_mod/len(similar):.2f}/query)")
    print(f"  Cache hits:             {warm_hits}/{len(similar)} ({warm_hits/len(similar):.0%})")
    print(f"  Avg latency:            {sum(warm_latencies)/len(warm_latencies):.0f} ms")
    print()
    print("  --- OVERALL ---")
    print(f"  Total Voyage calls:     {counters['voyage_calls']}")
    print(f"  Total Voyage texts:     {counters['voyage_texts']}")
    print(f"  Total OpenAI calls:     {counters['openai_chat']}")
    print(f"  Total Moderation:       {counters['moderation']}")
    print(f"  Cache hit rate:         {(cold_hits + warm_hits)/n_total:.1%}")
    print(f"  Avg latency:            {sum(all_latencies)/len(all_latencies):.0f} ms")
    print(f"  P50 latency:            {sorted(all_latencies)[len(all_latencies)//2]:.0f} ms")
    print(f"  P99 latency:            {sorted(all_latencies)[int(len(all_latencies)*0.99)]:.0f} ms")
    print()
    print("  --- API COST ---")
    print(f"  Voyage embedding:       ${total_cost['voyage_usd']:.6f}")
    print(f"  OpenAI chat:            ${total_cost['openai_usd']:.6f}")
    print(f"  OpenAI moderation:      ${total_cost['moderation_usd']:.6f}")
    print(f"  Total:                  ${total_cost['total_usd']:.6f}")
    print(f"  Cost/query (overall):   ${total_cost['total_usd']/n_total:.8f}")
    print(f"  Cost/query (cold only): ${total_cost['total_usd']/len(sample):.8f}")
    print()

    # Extrapolation to 1000
    cold_cost_per = total_cost['total_usd'] / len(sample) if len(sample) > 0 else 0
    print("  --- EXTRAPOLATION TO 1000 QUERIES (100 cold + 900 similar) ---")
    print(f"  100 cold queries:       ${cold_cost_per * 100:.4f}")
    print("  900 cached (est $0):    $0.0000")
    print(f"  Total est. 1000:        ${cold_cost_per * 100:.4f}")
    print(f"  Est. cost/query @1000:  ${cold_cost_per * 100 / 1000:.6f}")
    print(f"{'='*70}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"cost_analysis_{ts}.json"
    save_data = {
        "cold": {"count": len(sample), "cache_hits": cold_hits, "voyage": cold_voyage, "openai": cold_openai, "moderation": cold_mod, "avg_latency_ms": round(sum(cold_latencies)/len(cold_latencies))},
        "similar": {"count": len(similar), "cache_hits": warm_hits, "voyage": warm_voyage, "openai": warm_openai, "moderation": warm_mod, "avg_latency_ms": round(sum(warm_latencies)/len(warm_latencies))},
        "total_cost": total_cost,
        "counters": dict(counters),
    }
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
