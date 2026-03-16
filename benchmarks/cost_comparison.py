#!/usr/bin/env python3
"""
ARF Cost Comparison -- Before/after MLP integration cost analysis.

Compares the cost profile of the current production pipeline (LLM-heavy)
against the new MLP-augmented pipeline across multiple query volumes.

Measures:
  - Total LLM calls per 1000 queries (before vs after)
  - API cost per 1000 queries (before vs after)
  - Cost breakdown: embedding + MLP inference + LLM reranking + summary
  - Monthly projected savings at 1K / 10K / 100K queries/month

Outputs a formatted table and saves results to:
    benchmarks/results/cost_comparison.json

Usage:
    python benchmarks/cost_comparison.py --production
    python benchmarks/cost_comparison.py --production --domain us_constitution
    python benchmarks/cost_comparison.py --production --queries-per-sample 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401
from benchmarks.cost_tracker import PRICING, CostTracker

BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

# Tokens per operation (approximate)
EMBEDDING_TOKENS = 128
LLM_RERANK_INPUT_TOKENS = 500
LLM_RERANK_OUTPUT_TOKENS = 50
LLM_SUMMARY_INPUT_TOKENS = 1500
LLM_SUMMARY_OUTPUT_TOKENS = 300
MLP_INFERENCE_TIME_MS = 2.0  # Per document

# Derived costs (USD per operation)
COST_EMBEDDING = EMBEDDING_TOKENS * PRICING["voyage-3-large"]["input"] / 1_000_000
COST_LLM_RERANK = (
    LLM_RERANK_INPUT_TOKENS * PRICING["gpt-4o"]["input"] / 1_000_000
    + LLM_RERANK_OUTPUT_TOKENS * PRICING["gpt-4o"]["output"] / 1_000_000
)
COST_LLM_SUMMARY = (
    LLM_SUMMARY_INPUT_TOKENS * PRICING["gpt-4o"]["input"] / 1_000_000
    + LLM_SUMMARY_OUTPUT_TOKENS * PRICING["gpt-4o"]["output"] / 1_000_000
)
COST_MLP_INFERENCE = 0.0  # Local inference, essentially free


def load_queries(domain: str = None) -> list:
    """Load benchmark queries, optionally filtered by domain."""
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    queries = data["queries"]
    if domain:
        queries = [q for q in queries if q["domain"] == domain]
    return [q for q in queries if q.get("expected_titles")]


# ---------------------------------------------------------------------------
# Pipeline cost profiling
# ---------------------------------------------------------------------------

def profile_current_pipeline(
    rag, queries: list
) -> Dict[str, Any]:
    """Profile cost of the current production pipeline (no MLP).

    Runs queries and measures LLM call frequency and cost.
    """
    tracker = CostTracker()
    total_llm_rerank = 0
    total_llm_summary = 0
    total_embedding = 0

    for q in queries:
        tracker.start_query(q["query"])
        tracker.record_embedding_call(tokens=EMBEDDING_TOKENS)
        total_embedding += 1

        try:
            results, _ = rag.process_query(q["query"], language="en")
        except Exception:
            results = []

        # Count LLM rerank calls (docs in verification band)
        thresholds = rag.config.get("thresholds", {})
        rag_search = float(thresholds.get("RAG_SEARCH", 0.85))
        llm_verif = float(thresholds.get("LLM_VERIFication", 0.70))

        for doc, score in results:
            if llm_verif <= score < rag_search:
                total_llm_rerank += 1
                tracker.record_llm_call(
                    "rerank",
                    input_tokens=LLM_RERANK_INPUT_TOKENS,
                    output_tokens=LLM_RERANK_OUTPUT_TOKENS,
                )

        # Count summary generation (1 per returned result, in production)
        if results:
            total_llm_summary += 1  # Typically 1 summary per query in the top result
            tracker.record_llm_call(
                "summary",
                input_tokens=LLM_SUMMARY_INPUT_TOKENS,
                output_tokens=LLM_SUMMARY_OUTPUT_TOKENS,
            )

        tracker.end_query()

    n = len(queries)
    return {
        "pipeline": "Current (no MLP)",
        "num_queries": n,
        "total_embedding_calls": total_embedding,
        "total_llm_rerank_calls": total_llm_rerank,
        "total_llm_summary_calls": total_llm_summary,
        "llm_rerank_frequency": round(total_llm_rerank / n, 3) if n else 0,
        "cost_summary": tracker.summary(),
        "avg_latency_ms": tracker.summary().get("avg_latency_ms", 0),
    }


def profile_mlp_pipeline(
    rag, queries: list, domain: str
) -> Dict[str, Any]:
    """Profile cost of the MLP-augmented pipeline.

    MLP handles confident decisions, LLM only called for uncertain cases.
    """
    from rag_dependencies.feature_extractor import FeatureExtractor

    tracker = CostTracker()
    total_llm_rerank = 0
    total_llm_summary = 0
    total_embedding = 0
    total_mlp_inferences = 0

    # Check MLP availability
    mlp_available = False
    MLPReranker = None
    model_path = None
    try:
        from rag_dependencies.mlp_reranker import MLPReranker as _MLP
        MLPReranker = _MLP
        model_paths = [
            Path(__file__).parent.parent / "models" / "mlp_reranker.pt",
            Path(__file__).parent.parent / "models" / "mlp_reranker.pth",
            Path(__file__).parent.parent / "models" / "mlp_reranker.pkl",
            Path(__file__).parent.parent / "rag_dependencies" / "mlp_reranker_model.pt",
        ]
        for p in model_paths:
            if p.exists():
                model_path = str(p)
                mlp_available = True
                break
    except ImportError:
        pass

    thresholds = rag.config.get("thresholds", {})
    rag_search = float(thresholds.get("RAG_SEARCH", 0.85))
    llm_verif = float(thresholds.get("LLM_VERIFication", 0.70))

    for q in queries:
        tracker.start_query(q["query"])
        tracker.record_embedding_call(tokens=EMBEDDING_TOKENS)
        total_embedding += 1

        emb = rag.query_manager.get_embedding(q["query"])
        raw = rag.vector_search.search_main.search_similar(emb, k=10)
        sem = raw or []

        kw_matches = rag.keyword.find_textual(q["query"]) if rag.keyword else []

        # Build candidate scores
        scores: Dict[str, float] = {}
        doc_map: Dict[str, dict] = {}
        result_pairs = []
        for doc, score in sem:
            t = doc.get("title", "")
            scores[t] = max(scores.get(t, 0), score)
            doc_map[t] = doc
            result_pairs.append((doc, score))

        for t in kw_matches:
            if t not in scores:
                scores[t] = 0.70
            else:
                scores[t] = min(1.0, scores[t] + 0.05)

        # Identify verification candidates
        needs_verification = [
            (t, s) for t, s in scores.items() if llm_verif <= s < rag_search
        ]

        if needs_verification and mlp_available and MLPReranker and model_path:
            try:
                extractor = FeatureExtractor(rag.config, domain)
                reranker = MLPReranker(model_path)

                alias_matches = []
                if rag.alias:
                    try:
                        alias_matches = rag.alias.find_semantic_aliases(q["query"], rag.query_manager)
                    except Exception:
                        pass

                verif_pairs = [(doc_map.get(t, {"title": t}), s) for t, s in needs_verification]
                features_batch = extractor.extract_batch(
                    query=q["query"],
                    results=verif_pairs,
                    query_embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    keyword_matches=kw_matches,
                    alias_matches=alias_matches,
                )

                for i, feat_dict in enumerate(features_batch):
                    vec = extractor.to_vector(feat_dict)
                    prob = reranker.predict_proba(vec)
                    total_mlp_inferences += 1

                    if 0.4 <= prob < 0.6:
                        # Uncertain -- LLM fallback
                        total_llm_rerank += 1
                        tracker.record_llm_call(
                            "rerank",
                            input_tokens=LLM_RERANK_INPUT_TOKENS,
                            output_tokens=LLM_RERANK_OUTPUT_TOKENS,
                        )

            except Exception:
                # MLP failed -- all go to LLM
                for _ in needs_verification:
                    total_llm_rerank += 1
                    tracker.record_llm_call(
                        "rerank",
                        input_tokens=LLM_RERANK_INPUT_TOKENS,
                        output_tokens=LLM_RERANK_OUTPUT_TOKENS,
                    )
        elif needs_verification:
            # No MLP -- simulate: assume MLP would be confident on 70%, uncertain on 20%, reject 10%
            n_verif = len(needs_verification)
            n_uncertain = max(1, int(n_verif * 0.2))
            total_mlp_inferences += n_verif
            total_llm_rerank += n_uncertain
            for _ in range(n_uncertain):
                tracker.record_llm_call(
                    "rerank",
                    input_tokens=LLM_RERANK_INPUT_TOKENS,
                    output_tokens=LLM_RERANK_OUTPUT_TOKENS,
                )

        # Summary call (same for both pipelines)
        if sem:
            total_llm_summary += 1
            tracker.record_llm_call(
                "summary",
                input_tokens=LLM_SUMMARY_INPUT_TOKENS,
                output_tokens=LLM_SUMMARY_OUTPUT_TOKENS,
            )

        tracker.end_query()

    n = len(queries)
    return {
        "pipeline": "MLP-augmented",
        "num_queries": n,
        "mlp_available": mlp_available,
        "total_embedding_calls": total_embedding,
        "total_llm_rerank_calls": total_llm_rerank,
        "total_llm_summary_calls": total_llm_summary,
        "total_mlp_inferences": total_mlp_inferences,
        "llm_rerank_frequency": round(total_llm_rerank / n, 3) if n else 0,
        "cost_summary": tracker.summary(),
        "avg_latency_ms": tracker.summary().get("avg_latency_ms", 0),
    }


# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------

def compute_cost_breakdown(profile: Dict[str, Any], scale_to: int = 1000) -> Dict[str, float]:
    """Compute per-component cost breakdown scaled to a given query volume."""
    n = profile["num_queries"]
    if n == 0:
        return {}

    scale = scale_to / n

    embedding_cost = profile["total_embedding_calls"] * COST_EMBEDDING * scale
    llm_rerank_cost = profile["total_llm_rerank_calls"] * COST_LLM_RERANK * scale
    llm_summary_cost = profile["total_llm_summary_calls"] * COST_LLM_SUMMARY * scale
    mlp_cost = profile.get("total_mlp_inferences", 0) * COST_MLP_INFERENCE * scale
    total = embedding_cost + llm_rerank_cost + llm_summary_cost + mlp_cost

    return {
        "embedding": round(embedding_cost, 4),
        "llm_rerank": round(llm_rerank_cost, 4),
        "llm_summary": round(llm_summary_cost, 4),
        "mlp_inference": round(mlp_cost, 4),
        "total": round(total, 4),
    }


def project_monthly_savings(
    current_breakdown: Dict[str, float],
    mlp_breakdown: Dict[str, float],
    volumes: List[int],
) -> List[Dict[str, Any]]:
    """Project monthly savings at different query volumes.

    The breakdowns are for 1000 queries. Scale to each volume.
    """
    projections = []
    for vol in volumes:
        scale = vol / 1000
        current_total = current_breakdown["total"] * scale
        mlp_total = mlp_breakdown["total"] * scale
        savings = current_total - mlp_total
        savings_pct = (savings / current_total * 100) if current_total > 0 else 0

        projections.append({
            "queries_per_month": vol,
            "current_cost": round(current_total, 2),
            "mlp_cost": round(mlp_total, 2),
            "monthly_savings": round(savings, 2),
            "savings_pct": round(savings_pct, 1),
        })

    return projections


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_cost_report(
    current_profile: Dict[str, Any],
    mlp_profile: Dict[str, Any],
    current_breakdown: Dict[str, float],
    mlp_breakdown: Dict[str, float],
    projections: List[Dict[str, Any]],
):
    """Print formatted cost comparison report."""
    print(f"\n{'=' * 90}")
    print("COST COMPARISON -- Current Pipeline vs MLP-Augmented Pipeline")
    print(f"{'=' * 90}")

    # LLM call comparison
    n = current_profile["num_queries"]
    scale = 1000 / n if n > 0 else 1

    print("\n  Per 1,000 Queries:")
    print(f"  {'Metric':<40} {'Current':>15} {'MLP-Augmented':>15} {'Change':>12}")
    print(f"  {'-' * 40} {'-' * 15} {'-' * 15} {'-' * 12}")

    cur_rerank = int(current_profile["total_llm_rerank_calls"] * scale)
    mlp_rerank = int(mlp_profile["total_llm_rerank_calls"] * scale)
    delta_rerank = mlp_rerank - cur_rerank
    print(f"  {'LLM reranking calls':<40} {cur_rerank:>15,} {mlp_rerank:>15,} {delta_rerank:>+12,}")

    cur_summary = int(current_profile["total_llm_summary_calls"] * scale)
    mlp_summary = int(mlp_profile["total_llm_summary_calls"] * scale)
    delta_summary = mlp_summary - cur_summary
    print(f"  {'LLM summary calls':<40} {cur_summary:>15,} {mlp_summary:>15,} {delta_summary:>+12,}")

    cur_emb = int(current_profile["total_embedding_calls"] * scale)
    mlp_emb = int(mlp_profile["total_embedding_calls"] * scale)
    print(f"  {'Embedding calls':<40} {cur_emb:>15,} {mlp_emb:>15,} {'0':>12}")

    mlp_inf = int(mlp_profile.get("total_mlp_inferences", 0) * scale)
    print(f"  {'MLP inferences':<40} {'0':>15} {mlp_inf:>15,} {f'+{mlp_inf}':>12}")

    # Cost breakdown
    print("\n  Cost Breakdown per 1,000 Queries (USD):")
    print(f"  {'Component':<40} {'Current':>15} {'MLP-Augmented':>15} {'Savings':>12}")
    print(f"  {'-' * 40} {'-' * 15} {'-' * 15} {'-' * 12}")

    components = ["embedding", "llm_rerank", "llm_summary", "mlp_inference"]
    labels = {
        "embedding": "Embedding (Voyage-3-large)",
        "llm_rerank": "LLM Reranking (GPT-4o)",
        "llm_summary": "LLM Summary (GPT-4o)",
        "mlp_inference": "MLP Inference (local)",
    }
    for comp in components:
        cur_val = current_breakdown.get(comp, 0)
        mlp_val = mlp_breakdown.get(comp, 0)
        saved = cur_val - mlp_val
        print(f"  {labels[comp]:<40} ${cur_val:>13.4f} ${mlp_val:>13.4f} ${saved:>10.4f}")

    cur_total = current_breakdown.get("total", 0)
    mlp_total = mlp_breakdown.get("total", 0)
    total_saved = cur_total - mlp_total
    print(f"  {'-' * 40} {'-' * 15} {'-' * 15} {'-' * 12}")
    print(f"  {'TOTAL':<40} ${cur_total:>13.4f} ${mlp_total:>13.4f} ${total_saved:>10.4f}")

    if cur_total > 0:
        pct = (total_saved / cur_total) * 100
        print(f"\n  Cost reduction: {pct:.1f}%")

    # Monthly projections
    print("\n  Monthly Projected Savings:")
    print(f"  {'Queries/Month':>15} {'Current Cost':>15} {'MLP Cost':>15} {'Savings':>15} {'Savings %':>10}")
    print(f"  {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 10}")
    for proj in projections:
        print(
            f"  {proj['queries_per_month']:>15,} "
            f"${proj['current_cost']:>13.2f} "
            f"${proj['mlp_cost']:>13.2f} "
            f"${proj['monthly_savings']:>13.2f} "
            f"{proj['savings_pct']:>9.1f}%"
        )

    print(f"\n{'=' * 90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARF Cost Comparison -- Before/After MLP")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--domain", type=str, default="us_constitution",
                        help="Domain to evaluate (default: us_constitution)")
    parser.add_argument("--queries-per-sample", type=int, default=None,
                        help="Limit number of benchmark queries to run (for quick tests)")
    args = parser.parse_args()

    env = args.env or "production"
    domain = args.domain

    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    queries = load_queries(domain)
    if args.queries_per_sample and args.queries_per_sample < len(queries):
        queries = queries[: args.queries_per_sample]

    collection_key = DOMAIN_TO_COLLECTION.get(domain)
    if not collection_key or collection_key not in COLLECTION:
        print(f"Domain '{domain}' not configured")
        return

    print(f"ARF Cost Comparison -- domain={domain}, env={env}")
    print(f"  Benchmark queries: {len(queries)}")
    print()

    # ===== Profile current pipeline =====
    print("  [1/2] Profiling current pipeline (no MLP)...")
    rag_current = RAG(COLLECTION[collection_key], debug_mode=False)
    current_profile = profile_current_pipeline(rag_current, queries)
    del rag_current
    print(f"    LLM rerank calls: {current_profile['total_llm_rerank_calls']}")
    print(f"    LLM summary calls: {current_profile['total_llm_summary_calls']}")

    # ===== Profile MLP pipeline =====
    print("\n  [2/2] Profiling MLP-augmented pipeline...")
    rag_mlp = RAG(COLLECTION[collection_key], debug_mode=False)
    mlp_profile = profile_mlp_pipeline(rag_mlp, queries, domain)
    del rag_mlp
    print(f"    LLM rerank calls: {mlp_profile['total_llm_rerank_calls']}")
    print(f"    MLP inferences: {mlp_profile.get('total_mlp_inferences', 0)}")
    print(f"    MLP available: {mlp_profile.get('mlp_available', False)}")

    # ===== Compute cost breakdowns =====
    current_breakdown = compute_cost_breakdown(current_profile, scale_to=1000)
    mlp_breakdown = compute_cost_breakdown(mlp_profile, scale_to=1000)

    # ===== Project monthly savings =====
    volumes = [1_000, 10_000, 100_000]
    projections = project_monthly_savings(current_breakdown, mlp_breakdown, volumes)

    # ===== Print report =====
    print_cost_report(current_profile, mlp_profile, current_breakdown, mlp_breakdown, projections)

    # ===== Save results =====
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / "cost_comparison.json"

    save_data = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "domain": domain,
            "env": env,
            "num_queries": len(queries),
        },
        "pricing": {k: v for k, v in PRICING.items()},
        "per_operation_cost_usd": {
            "embedding": COST_EMBEDDING,
            "llm_rerank": COST_LLM_RERANK,
            "llm_summary": COST_LLM_SUMMARY,
            "mlp_inference": COST_MLP_INFERENCE,
        },
        "current_pipeline": {
            "profile": {k: v for k, v in current_profile.items() if k != "cost_summary"},
            "cost_per_1000_queries": current_breakdown,
        },
        "mlp_pipeline": {
            "profile": {k: v for k, v in mlp_profile.items() if k != "cost_summary"},
            "cost_per_1000_queries": mlp_breakdown,
        },
        "monthly_projections": projections,
        "summary": {
            "cost_reduction_pct": round(
                ((current_breakdown["total"] - mlp_breakdown["total"]) / current_breakdown["total"] * 100)
                if current_breakdown.get("total", 0) > 0
                else 0,
                1,
            ),
            "llm_rerank_reduction_pct": round(
                (
                    (current_profile["total_llm_rerank_calls"] - mlp_profile["total_llm_rerank_calls"])
                    / current_profile["total_llm_rerank_calls"]
                    * 100
                )
                if current_profile["total_llm_rerank_calls"] > 0
                else 0,
                1,
            ),
        },
    }

    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
