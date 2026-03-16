#!/usr/bin/env python3
"""
Monthly Automated Retraining Pipeline for the MLP Reranker

Collects recent LLM verifier judgments from MongoDB, generates features,
merges with existing training data, retrains the MLP, and deploys the new
model only if performance improves.

Usage:
    # Dry run -- check what would change without modifying anything
    python benchmarks/retrain_monthly.py --production --dry-run

    # Actually retrain and deploy
    python benchmarks/retrain_monthly.py --production

    # Retrain with a custom lookback window (default: 30 days)
    python benchmarks/retrain_monthly.py --production --lookback-days 60

    # Force deploy even if metrics decrease
    python benchmarks/retrain_monthly.py --production --force

Pipeline steps:
    1. Export recent LLM verifier judgments from MongoDB (User_queries collection)
    2. Generate features for new query-document pairs
    3. Merge with existing training data (features cache)
    4. Retrain the MLP with the expanded dataset
    5. Validate on a held-out test set
    6. Only deploy new model if performance >= old model
    7. Log retraining metrics to benchmarks/results/retrain_log.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path and set up services.rag import shim
sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RETRAIN_LOG_FILE = RESULTS_DIR / "retrain_log.json"
DEFAULT_FEATURES_CACHE = Path(__file__).parent / "features_cache.json"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "mlp_reranker.joblib"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}


# ======================================================================
# Step 1: Export recent LLM judgments from MongoDB
# ======================================================================

def export_recent_judgments(
    env: str = "production",
    lookback_days: int = 30,
) -> List[Dict[str, Any]]:
    """Query MongoDB for recent LLM-verified query-document pairs.

    Looks in the User_queries collection for queries that have results
    with LLM verification scores, processed within the lookback window.
    """
    from config import MONGO_URI, QUERY_COLLECTION_NAME, load_environment
    load_environment(env)

    from pymongo import MongoClient

    client = MongoClient(MONGO_URI)
    db = client["public"]
    query_collection = db[QUERY_COLLECTION_NAME]

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=lookback_days)

    logger.info("Querying MongoDB for judgments since %s...", cutoff.isoformat())

    # Find queries with results that have been processed recently
    # We look for documents with 'results' array and 'updated_at' or 'searched_datetime'
    pipeline = [
        {
            "$match": {
                "$or": [
                    {"updated_at": {"$gte": cutoff}},
                    {"searched_datetime": {"$gte": cutoff.isoformat()}},
                    {"last_search_at": {"$gte": cutoff}},
                ],
                "results": {"$exists": True, "$ne": []},
            }
        },
        {
            "$project": {
                "query": 1,
                "results": 1,
                "updated_at": 1,
                "avg_relevance_score": 1,
            }
        },
        {"$limit": 5000},  # Safety limit
    ]

    judgments = []
    try:
        cursor = query_collection.aggregate(pipeline, maxTimeMS=60000)
        for doc in cursor:
            query_text = doc.get("query", "")
            if not query_text:
                continue

            results = doc.get("results", [])
            for result in results:
                if not isinstance(result, dict):
                    continue

                title = result.get("title", "")
                score = result.get("score") or result.get("relevance_score")
                if title and score is not None:
                    judgments.append({
                        "query": query_text,
                        "title": title,
                        "score": float(score),
                        "verified": True,
                        "source": "mongodb_judgments",
                    })

        logger.info("Exported %d judgment pairs from MongoDB", len(judgments))
    except Exception as e:
        logger.error("Failed to export judgments from MongoDB: %s", e)
    finally:
        client.close()

    return judgments


# ======================================================================
# Step 2: Generate features for new judgment pairs
# ======================================================================

def generate_features_for_judgments(
    judgments: List[Dict[str, Any]],
    env: str = "production",
) -> List[Dict[str, Any]]:
    """Generate feature vectors for judgment pairs by running vector search."""
    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    # We need vector search to get scores and doc details
    # Group judgments by likely domain (heuristic: try each domain)
    rag_instances = {}
    for domain, collection_key in DOMAIN_TO_COLLECTION.items():
        if collection_key in COLLECTION:
            try:
                rag_instances[domain] = RAG(COLLECTION[collection_key], debug_mode=False)
            except Exception as e:
                logger.warning("Could not initialize RAG for %s: %s", domain, e)

    rows = []
    seen_pairs = set()

    for judgment in judgments:
        query_text = judgment["query"]
        target_title = judgment["title"]
        verified_score = judgment["score"]
        pair_key = (query_text.lower().strip(), target_title.lower().strip())

        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Label: score >= 0.7 means the LLM verifier considered it relevant
        # This maps to our binary label: 1 = relevant, 0 = not relevant
        label = 1 if verified_score >= 0.7 else 0

        # Try each domain to find the document
        for domain, rag in rag_instances.items():
            try:
                vec = rag.query_manager.get_embedding(query_text)
                if hasattr(rag, "vector_search") and rag.vector_search:
                    results = rag.vector_search.search_main.search_similar(vec, k=20)
                else:
                    continue

                if not results:
                    continue

                # Find the target document in results
                top_score = results[0][1]
                for rank, (doc, v_score) in enumerate(results):
                    doc_title = doc.get("title", "")
                    if doc_title.lower().strip() == target_title.lower().strip():
                        # Found the document -- extract features
                        from benchmarks.train_reranker import _extract_features_for_pair, _features_to_vector
                        features = _extract_features_for_pair(query_text, doc, v_score, domain)
                        features["score_rank"] = float(rank)
                        features["score_gap_to_top"] = float(top_score - v_score)
                        vector, names = _features_to_vector(features)

                        rows.append({
                            "vector": vector,
                            "feature_names": names,
                            "label": label,
                            "query_id": f"retrain_{len(rows)}",
                            "domain": domain,
                            "title": doc_title,
                            "score": float(v_score),
                            "source": "llm_judgment",
                        })
                        break  # Found the doc, move to next judgment

                    # Also add negatives from this query's results
                    if rank < 5 and doc_title.lower().strip() != target_title.lower().strip():
                        neg_pair = (query_text.lower().strip(), doc_title.lower().strip())
                        if neg_pair not in seen_pairs:
                            seen_pairs.add(neg_pair)
                            features = _extract_features_for_pair(query_text, doc, v_score, domain)
                            features["score_rank"] = float(rank)
                            features["score_gap_to_top"] = float(top_score - v_score)
                            vector, names = _features_to_vector(features)

                            rows.append({
                                "vector": vector,
                                "feature_names": names,
                                "label": 0,
                                "query_id": f"retrain_neg_{len(rows)}",
                                "domain": domain,
                                "title": doc_title,
                                "score": float(v_score),
                                "source": "llm_judgment_negative",
                            })

                break  # Found the domain, move to next judgment

            except Exception as e:
                logger.debug("Feature gen failed for domain %s, query %s: %s",
                             domain, query_text[:40], e)
                continue

    logger.info("Generated %d feature rows from %d judgments", len(rows), len(judgments))
    return rows


# ======================================================================
# Step 3: Merge with existing data
# ======================================================================

def merge_features(
    existing_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge existing and new feature rows, deduplicating by (query, title)."""
    seen = set()
    merged = []

    # New data takes priority (more recent judgments)
    for row in new_rows:
        key = (row.get("query_id", ""), row.get("title", "").lower())
        if key not in seen:
            seen.add(key)
            merged.append(row)

    # Add existing data that isn't duplicated
    for row in existing_rows:
        key = (row.get("query_id", ""), row.get("title", "").lower())
        if key not in seen:
            seen.add(key)
            merged.append(row)

    logger.info("Merged: %d existing + %d new = %d total (after dedup)",
                len(existing_rows), len(new_rows), len(merged))
    return merged


# ======================================================================
# Step 4-5: Retrain and validate
# ======================================================================

def retrain_and_validate(
    rows: List[Dict[str, Any]],
    old_model_path: str,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Retrain the MLP and validate against the old model.

    Returns:
        (new_reranker, metrics_report) -- reranker is None if old model is better
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from rag_dependencies.mlp_reranker import MLPReranker

    feature_names = rows[0]["feature_names"]
    X = np.array([r["vector"] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int32)

    # Split into train/test
    if len(X) < 10:
        logger.warning("Very few samples (%d) -- using all for training", len(X))
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, stratify=y, random_state=random_state,
        )

    # Evaluate old model on test set (if it exists)
    old_metrics = {"f1": 0.0, "accuracy": 0.0}
    old_model_exists = Path(old_model_path).is_file()
    if old_model_exists:
        try:
            old_reranker = MLPReranker.load(old_model_path)
            old_probas = old_reranker.predict(X_test.tolist())
            old_preds = [1 if p >= 0.5 else 0 for p in old_probas]
            old_metrics["f1"] = f1_score(y_test, old_preds, zero_division=0)
            old_metrics["accuracy"] = accuracy_score(y_test, old_preds)
            if len(np.unique(y_test)) > 1:
                old_metrics["auc_roc"] = roc_auc_score(y_test, old_probas)
            logger.info("Old model test metrics: %s", old_metrics)
        except Exception as e:
            logger.warning("Could not evaluate old model: %s", e)
            old_model_exists = False

    # Train new model
    new_reranker = MLPReranker()
    new_train_metrics = new_reranker.train(
        X_train, y_train,
        hidden_layer_sizes=(128, 64, 32),
        calibrate=True,
        feature_names=feature_names,
    )

    # Evaluate new model on test set
    new_probas = new_reranker.predict(X_test.tolist())
    new_preds = [1 if p >= 0.5 else 0 for p in new_probas]
    new_metrics = {
        "f1": round(f1_score(y_test, new_preds, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, new_preds), 4),
        "train_metrics": new_train_metrics,
    }
    if len(np.unique(y_test)) > 1:
        new_metrics["auc_roc"] = round(roc_auc_score(y_test, new_probas), 4)

    logger.info("New model test metrics: %s", {k: v for k, v in new_metrics.items() if k != "train_metrics"})

    # Compare: deploy new model only if it improves F1
    improved = new_metrics["f1"] >= old_metrics.get("f1", 0)

    report = {
        "old_model_existed": old_model_exists,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "improved": improved,
        "improvement_delta": round(new_metrics["f1"] - old_metrics.get("f1", 0), 4),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_rate": round(float(np.mean(y)), 3),
    }

    if improved:
        return new_reranker, report
    else:
        return None, report


# ======================================================================
# Step 7: Logging
# ======================================================================

def append_retrain_log(entry: Dict[str, Any]) -> None:
    """Append a retraining entry to the persistent log file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log = []
    if RETRAIN_LOG_FILE.is_file():
        try:
            with open(RETRAIN_LOG_FILE) as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            log = []

    log.append(entry)

    with open(RETRAIN_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2, default=str)
    logger.info("Retrain log updated at %s", RETRAIN_LOG_FILE)


# ======================================================================
# Main
# ======================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Monthly MLP Reranker Retraining")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--dry-run", action="store_true", help="Check what would change without modifying anything")
    parser.add_argument("--force", action="store_true", help="Deploy new model even if metrics decrease")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of judgments to consider")
    parser.add_argument("--features-cache", type=str, default=str(DEFAULT_FEATURES_CACHE),
                        help="Path to existing features cache")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to current model (and where to save new one)")
    args = parser.parse_args()

    env = args.env or "production"

    print(f"\n{'='*60}")
    print("ARF MLP Reranker Monthly Retraining")
    print(f"{'='*60}")
    print(f"  Environment:     {env}")
    print(f"  Lookback:        {args.lookback_days} days")
    print(f"  Dry run:         {args.dry_run}")
    print(f"  Force deploy:    {args.force}")
    print(f"  Model path:      {args.model_path}")
    print(f"  Features cache:  {args.features_cache}")
    print()

    # Step 1: Export recent judgments
    print("  Step 1: Exporting recent LLM judgments...")
    judgments = export_recent_judgments(env=env, lookback_days=args.lookback_days)
    print(f"    Found {len(judgments)} judgment pairs")

    if not judgments:
        print("\n  No new judgments found. Nothing to retrain.")
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "skipped",
            "reason": "no_new_judgments",
            "lookback_days": args.lookback_days,
        }
        if not args.dry_run:
            append_retrain_log(log_entry)
        return

    # Step 2: Generate features for new data
    print("\n  Step 2: Generating features for new judgment pairs...")
    new_rows = generate_features_for_judgments(judgments, env=env)
    print(f"    Generated {len(new_rows)} feature rows (positive: {sum(r['label'] for r in new_rows)})")

    if not new_rows:
        print("\n  No features generated. Check MongoDB connectivity and domain configuration.")
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "skipped",
            "reason": "no_features_generated",
            "n_judgments": len(judgments),
        }
        if not args.dry_run:
            append_retrain_log(log_entry)
        return

    # Step 3: Merge with existing training data
    print("\n  Step 3: Merging with existing training data...")
    existing_rows = []
    if Path(args.features_cache).is_file():
        try:
            from benchmarks.train_reranker import load_features_cache
            existing_rows = load_features_cache(args.features_cache)
            print(f"    Loaded {len(existing_rows)} existing rows from cache")
        except Exception as e:
            logger.warning("Could not load existing features: %s", e)

    merged_rows = merge_features(existing_rows, new_rows)
    print(f"    Total training data: {len(merged_rows)} rows")

    n_positive = sum(r["label"] for r in merged_rows)
    n_negative = len(merged_rows) - n_positive
    print(f"    Positive: {n_positive}, Negative: {n_negative}")

    if n_positive < 2 or n_negative < 2:
        print("\n  Insufficient labelled data for training (need >=2 per class).")
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "skipped",
            "reason": "insufficient_data",
            "n_positive": n_positive,
            "n_negative": n_negative,
        }
        if not args.dry_run:
            append_retrain_log(log_entry)
        return

    if args.dry_run:
        print(f"\n  DRY RUN: Would retrain with {len(merged_rows)} samples.")
        print(f"  DRY RUN: New data adds {len(new_rows)} rows to {len(existing_rows)} existing rows.")
        print("  DRY RUN: No model changes made.\n")
        return

    # Step 4-5: Retrain and validate
    print("\n  Step 4-5: Retraining and validating...")
    new_reranker, report = retrain_and_validate(
        merged_rows, args.model_path,
    )

    print("\n  Validation Results:")
    print(f"    Old model F1:   {report['old_metrics'].get('f1', 'N/A')}")
    print(f"    New model F1:   {report['new_metrics']['f1']}")
    print(f"    Improved:       {report['improved']}")
    print(f"    Delta:          {report['improvement_delta']:+.4f}")

    # Step 6: Deploy if improved (or forced)
    deploy = report["improved"] or args.force
    if deploy and new_reranker is not None:
        print(f"\n  Step 6: Deploying new model to {args.model_path}")
        new_reranker.save(args.model_path)

        # Update features cache with merged data
        try:
            from benchmarks.train_reranker import save_features_cache
            save_features_cache(merged_rows, args.features_cache)
            print(f"    Updated features cache: {args.features_cache}")
        except Exception as e:
            logger.warning("Could not update features cache: %s", e)

        deploy_status = "deployed"
    elif args.force and new_reranker is None:
        # Force was requested but retrain_and_validate returned None -- need to rebuild
        from rag_dependencies.mlp_reranker import MLPReranker
        forced_reranker = MLPReranker()
        X = np.array([r["vector"] for r in merged_rows], dtype=np.float64)
        y_arr = np.array([r["label"] for r in merged_rows], dtype=np.int32)
        forced_reranker.train(X, y_arr, hidden_layer_sizes=(128, 64, 32), calibrate=True,
                              feature_names=merged_rows[0]["feature_names"])
        forced_reranker.save(args.model_path)
        print("\n  Step 6: FORCE deployed new model (metrics decreased)")
        deploy_status = "force_deployed"
    else:
        print("\n  Step 6: Skipping deployment (old model is better)")
        deploy_status = "skipped_no_improvement"

    # Step 7: Log
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": deploy_status,
        "n_judgments": len(judgments),
        "n_new_features": len(new_rows),
        "n_existing_features": len(existing_rows),
        "n_merged": len(merged_rows),
        "report": report,
    }
    append_retrain_log(log_entry)

    print(f"\n{'='*60}")
    print("RETRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Status:           {deploy_status}")
    print(f"  New judgments:     {len(judgments)}")
    print(f"  Training samples:  {len(merged_rows)}")
    print(f"  Old F1:           {report['old_metrics'].get('f1', 'N/A')}")
    print(f"  New F1:           {report['new_metrics']['f1']}")
    print(f"  Model path:       {args.model_path}")
    print(f"  Retrain log:      {RETRAIN_LOG_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
