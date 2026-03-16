#!/usr/bin/env python3
"""
MLP Reranker Training Pipeline

Trains, evaluates, and saves an MLP reranker model for the ARF legal RAG
pipeline.  Supports multiple training modes:

    # Train from eval dataset (generates features via MongoDB vector search)
    python benchmarks/train_reranker.py --dataset benchmarks/eval_dataset.json --production

    # Train from eval dataset with feature caching
    python benchmarks/train_reranker.py --dataset benchmarks/eval_dataset.json --features-cache benchmarks/features_cache.json --production

    # Train from benchmark_queries.json (default dataset)
    python benchmarks/train_reranker.py --production

    # Retrain from previously cached features (no MongoDB needed)
    python benchmarks/train_reranker.py --retrain --features-cache benchmarks/features_cache.json

The pipeline:
    1. Loads evaluation queries (benchmark_queries.json or eval_dataset.json)
    2. Generates features via MongoDB vector search for each query-doc pair
       (or loads pre-computed features from cache)
    3. Trains multiple model architectures and a logistic regression baseline
    4. Performs stratified K-fold cross-validation per domain
    5. Reports accuracy, precision, recall, F1, AUC-ROC per model
    6. Trains the best architecture on the full dataset with isotonic calibration
    7. Saves the calibrated model to models/mlp_reranker.joblib
    8. Generates a training report to benchmarks/results/training_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path and set up services.rag import shim
sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401

logger = logging.getLogger(__name__)

BENCHMARK_FILE = Path(__file__).parent / "benchmark_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "mlp_reranker.joblib"

# Domain -> collection key mapping (mirrors run_eval.py)
DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}

# Model architectures to compare
MODEL_CONFIGS = {
    "logistic_regression": {"type": "logistic"},
    "mlp_2layer_64_32": {"type": "mlp", "hidden_layer_sizes": (64, 32)},
    "mlp_3layer_128_64_32": {"type": "mlp", "hidden_layer_sizes": (128, 64, 32)},
}


# ======================================================================
# Feature Generation
# ======================================================================

def _extract_features_for_pair(
    query_text: str,
    doc: dict,
    vector_score: float,
    domain: str,
) -> Dict[str, Any]:
    """Extract features for a single (query, document) pair.

    Tries to use FeatureExtractor if available; otherwise falls back to
    a minimal hand-crafted feature set that captures the most important
    signals for the reranker.
    """
    features: Dict[str, float] = {}

    # --- Core retrieval score ---
    features["vector_score"] = float(vector_score)
    features["vector_score_sq"] = float(vector_score ** 2)

    # --- Text overlap signals ---
    query_lower = query_text.lower()
    title = (doc.get("title") or "").lower()
    text = (doc.get("text") or doc.get("summary") or "").lower()

    query_tokens = set(query_lower.split())
    title_tokens = set(title.split())
    text_tokens = set(text.split()) if text else set()

    # Token overlap ratios
    if query_tokens:
        features["title_token_overlap"] = len(query_tokens & title_tokens) / len(query_tokens)
        features["text_token_overlap"] = len(query_tokens & text_tokens) / len(query_tokens) if text_tokens else 0.0
    else:
        features["title_token_overlap"] = 0.0
        features["text_token_overlap"] = 0.0

    # Exact substring match
    features["query_in_title"] = 1.0 if query_lower in title else 0.0
    features["title_in_query"] = 1.0 if title and title in query_lower else 0.0

    # --- Length features ---
    features["title_length"] = float(len(title.split()))
    features["query_length"] = float(len(query_lower.split()))
    features["text_length_log"] = float(np.log1p(len(text.split()))) if text else 0.0

    # --- Structural features ---
    features["has_article"] = 1.0 if doc.get("article") else 0.0
    features["has_section"] = 1.0 if doc.get("section") else 0.0
    features["has_chapter"] = 1.0 if doc.get("chapter") else 0.0
    features["has_clauses"] = 1.0 if doc.get("clauses") else 0.0
    features["has_sections_array"] = 1.0 if doc.get("sections") else 0.0
    features["num_clauses"] = float(len(doc.get("clauses", []) or []))
    features["num_sections"] = float(len(doc.get("sections", []) or []))

    # --- Domain encoding (one-hot) ---
    for d in DOMAIN_TO_COLLECTION:
        features[f"domain_{d}"] = 1.0 if domain == d else 0.0

    # --- Score distribution features (will be filled per-query later) ---
    features["score_rank"] = 0.0  # placeholder, filled in batch
    features["score_gap_to_top"] = 0.0  # placeholder

    # Try FeatureExtractor (another agent creates it)
    try:
        from rag_dependencies.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        extracted = extractor.extract(query_text, doc, vector_score, domain)
        if isinstance(extracted, dict):
            # Merge extracted features (they take priority)
            features.update(extracted)
    except (ImportError, Exception):
        pass  # FeatureExtractor not available yet — use hand-crafted features

    return features


def _features_to_vector(features: Dict[str, float]) -> Tuple[List[float], List[str]]:
    """Convert a feature dict to a sorted numeric vector + ordered name list."""
    names = sorted(features.keys())
    vector = [float(features.get(n, 0.0)) for n in names]
    return vector, names


def generate_features_for_query(
    query_entry: dict,
    rag_instance,
    domain: str,
) -> List[Dict[str, Any]]:
    """Generate labelled feature rows for a single benchmark query.

    Runs vector search, then labels each candidate:
        - label=1 if candidate title is in expected_titles with relevance >= 2
        - label=0 otherwise (negatives)

    Returns list of dicts with keys: vector, label, query_id, domain, title, score.
    """
    query_text = query_entry["query"]
    expected_titles = set(query_entry.get("expected_titles", []))
    # For benchmark_queries.json format — default relevance=3 for expected docs
    expected_relevance = {t: query_entry.get("relevance", {}).get(t, 3) for t in expected_titles}

    try:
        # Generate embedding
        vec = rag_instance.query_manager.get_embedding(query_text)

        # Run vector search to get candidates
        if hasattr(rag_instance, "vector_search") and rag_instance.vector_search:
            results = rag_instance.vector_search.search_main.search_similar(vec, k=20)
        else:
            results = None

        if not results:
            logger.warning("No vector search results for query: %s", query_text[:60])
            return []

    except Exception as e:
        logger.error("Feature generation failed for %s: %s", query_entry.get("id", "?"), e)
        return []

    # Extract features for each candidate
    rows = []
    top_score = results[0][1] if results else 0.0

    for rank, (doc, score) in enumerate(results):
        title = doc.get("title", "")
        features = _extract_features_for_pair(query_text, doc, score, domain)
        features["score_rank"] = float(rank)
        features["score_gap_to_top"] = float(top_score - score)

        vector, names = _features_to_vector(features)

        # Label: 1 if title matches an expected doc with relevance >= 2
        if title in expected_titles:
            relevance = expected_relevance.get(title, 3)
            label = 1 if relevance >= 2 else 0
        else:
            label = 0

        rows.append({
            "vector": vector,
            "feature_names": names,
            "label": label,
            "query_id": query_entry.get("id", ""),
            "domain": domain,
            "title": title,
            "score": float(score),
        })

    return rows


def generate_all_features(
    queries: List[dict],
    env: str = "production",
) -> List[Dict[str, Any]]:
    """Generate features for all benchmark queries using live MongoDB.

    Returns a flat list of labelled feature rows.
    """
    from config import COLLECTION, load_environment
    load_environment(env)
    from RAG_interface import RAG

    all_rows: List[Dict[str, Any]] = []
    by_domain: Dict[str, List[dict]] = {}
    for q in queries:
        by_domain.setdefault(q["domain"], []).append(q)

    for domain, domain_queries in by_domain.items():
        collection_key = DOMAIN_TO_COLLECTION.get(domain)
        if not collection_key or collection_key not in COLLECTION:
            logger.warning("Skipping domain %s: collection not configured", domain)
            continue

        logger.info("Generating features for %s (%d queries)...", domain, len(domain_queries))
        rag = RAG(COLLECTION[collection_key], debug_mode=False)

        for q in domain_queries:
            rows = generate_features_for_query(q, rag, domain)
            all_rows.extend(rows)
            pos = sum(1 for r in rows if r["label"] == 1)
            logger.info("  %s: %d candidates (%d positive)", q["id"], len(rows), pos)

    logger.info("Total feature rows: %d (positive: %d)", len(all_rows), sum(r["label"] for r in all_rows))
    return all_rows


def save_features_cache(rows: List[Dict[str, Any]], path: str) -> None:
    """Save generated features to JSON for reuse."""
    # Convert numpy types
    serialisable = []
    for row in rows:
        r = dict(row)
        r["vector"] = [float(v) for v in r["vector"]]
        serialisable.append(r)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved %d feature rows to %s", len(serialisable), path)


def load_features_cache(path: str) -> List[Dict[str, Any]]:
    """Load pre-computed features from JSON."""
    with open(path) as f:
        rows = json.load(f)
    logger.info("Loaded %d feature rows from %s", len(rows), path)
    return rows


# ======================================================================
# Model Comparison
# ======================================================================

def _build_model(config: dict, random_state: int = 42):
    """Build a sklearn classifier from a model config dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if config["type"] == "logistic":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif config["type"] == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model type: {config['type']}")


def cross_validate_model(
    model_name: str,
    model_config: dict,
    X: np.ndarray,
    y: np.ndarray,
    domain_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run stratified K-fold CV and report metrics.

    Uses StratifiedKFold on the label to ensure balanced splits.
    Reports per-domain breakdowns when multiple domains are present.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    logger.info("Cross-validating %s (samples=%d, features=%d)...", model_name, X.shape[0], X.shape[1])

    # Ensure we have enough samples for CV
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    actual_splits = min(n_splits, max(2, int(min_class_count)))
    if actual_splits < n_splits:
        logger.warning("Reduced CV folds from %d to %d (min class count=%d)", n_splits, actual_splits, min_class_count)

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = _build_model(model_config, random_state=random_state + fold_idx)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        fm = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }
        if len(np.unique(y_val)) > 1:
            fm["auc_roc"] = roc_auc_score(y_val, y_proba)
        fold_metrics.append(fm)

    # Aggregate
    agg = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        vals = [fm[key] for fm in fold_metrics if key in fm]
        if vals:
            agg[key] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)

    # Per-domain metrics
    unique_domains = np.unique(domain_labels)
    per_domain = {}
    if len(unique_domains) > 1:
        for domain in unique_domains:
            mask = domain_labels == domain
            if np.sum(mask) < 5:
                continue
            domain_y = y[mask]
            if len(np.unique(domain_y)) < 2:
                continue
            # Quick train/test on domain subset
            try:
                from sklearn.model_selection import cross_val_predict
                scaler = StandardScaler()
                X_d = scaler.fit_transform(X[mask])
                model = _build_model(model_config, random_state=random_state)
                dm_splits = min(3, max(2, int(min(np.sum(domain_y == 0), np.sum(domain_y == 1)))))
                dm_cv = StratifiedKFold(n_splits=dm_splits, shuffle=True, random_state=random_state)
                y_pred_d = cross_val_predict(model, X_d, domain_y, cv=dm_cv)
                per_domain[str(domain)] = {
                    "samples": int(np.sum(mask)),
                    "positive_rate": round(float(np.mean(domain_y)), 3),
                    "f1": round(f1_score(domain_y, y_pred_d, zero_division=0), 4),
                    "accuracy": round(accuracy_score(domain_y, y_pred_d), 4),
                }
            except Exception as e:
                per_domain[str(domain)] = {"error": str(e)}

    result = {
        "model_name": model_name,
        "model_config": {k: list(v) if isinstance(v, tuple) else v for k, v in model_config.items()},
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_folds": actual_splits,
        "aggregate_metrics": agg,
        "fold_metrics": [{k: round(v, 4) if isinstance(v, float) else v for k, v in fm.items()} for fm in fold_metrics],
    }
    if per_domain:
        result["per_domain_metrics"] = per_domain

    logger.info("  %s => F1=%.4f  AUC=%.4f  Prec=%.4f  Rec=%.4f",
                model_name, agg.get("f1", 0), agg.get("auc_roc", 0),
                agg.get("precision", 0), agg.get("recall", 0))
    return result


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    domain_labels: np.ndarray,
    n_splits: int = 5,
) -> Tuple[List[Dict[str, Any]], str]:
    """Compare all model architectures and return results + best model name."""
    results = []
    for name, config in MODEL_CONFIGS.items():
        try:
            result = cross_validate_model(name, config, X, y, domain_labels, n_splits=n_splits)
            results.append(result)
        except Exception as e:
            logger.error("Model %s failed: %s", name, e)
            results.append({"model_name": name, "error": str(e)})

    # Select best by F1 (prefer AUC-ROC as tiebreaker)
    best_name = "mlp_3layer_128_64_32"  # default
    best_f1 = -1.0
    for r in results:
        agg = r.get("aggregate_metrics", {})
        f1 = agg.get("f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = r["model_name"]

    logger.info("Best model: %s (F1=%.4f)", best_name, best_f1)
    return results, best_name


# ======================================================================
# Feature Importance
# ======================================================================

def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Estimate feature importance via permutation importance."""
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, verbose=False)
        model.fit(X_scaled, y)

        result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42, scoring="f1")
        importances = []
        for i, name in enumerate(feature_names):
            importances.append({
                "feature": name,
                "importance_mean": round(float(result.importances_mean[i]), 4),
                "importance_std": round(float(result.importances_std[i]), 4),
            })
        importances.sort(key=lambda x: x["importance_mean"], reverse=True)
        return importances
    except Exception as e:
        logger.warning("Feature importance computation failed: %s", e)
        return []


# ======================================================================
# Training Report
# ======================================================================

def generate_training_report(
    comparison_results: List[Dict[str, Any]],
    best_model_name: str,
    final_metrics: Dict[str, Any],
    feature_importance: List[Dict[str, Any]],
    feature_names: List[str],
    n_total_samples: int,
    n_positive: int,
    calibration_quality: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Assemble the full training report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_samples": n_total_samples,
            "positive_samples": n_positive,
            "negative_samples": n_total_samples - n_positive,
            "positive_rate": round(n_positive / max(n_total_samples, 1), 3),
            "n_features": len(feature_names),
            "best_model": best_model_name,
            "best_model_f1": final_metrics.get("f1", 0),
            "best_model_auc": final_metrics.get("auc_roc", 0),
        },
        "model_comparison": comparison_results,
        "final_model": {
            "name": best_model_name,
            "config": MODEL_CONFIGS.get(best_model_name, {}),
            "metrics": final_metrics,
        },
        "feature_names": feature_names,
        "feature_importance": feature_importance[:20],  # top 20
    }
    if calibration_quality:
        report["calibration_quality"] = calibration_quality
    return report


# ======================================================================
# Main Pipeline
# ======================================================================

def load_queries(dataset_path: Optional[str] = None) -> List[dict]:
    """Load queries from eval_dataset.json or benchmark_queries.json."""
    if dataset_path and Path(dataset_path).is_file():
        with open(dataset_path) as f:
            data = json.load(f)
        # Support both formats:
        # eval_dataset.json: {"queries": [...]} or list directly
        # benchmark_queries.json: {"queries": [...]}
        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        else:
            raise ValueError(f"Unrecognised dataset format in {dataset_path}")
        logger.info("Loaded %d queries from %s", len(queries), dataset_path)
        return queries

    # Fallback to benchmark_queries.json
    if BENCHMARK_FILE.is_file():
        with open(BENCHMARK_FILE) as f:
            data = json.load(f)
        queries = data.get("queries", [])
        # Filter to queries with expected_titles for supervised training
        queries = [q for q in queries if q.get("expected_titles")]
        logger.info("Loaded %d queries with expected titles from %s", len(queries), BENCHMARK_FILE)
        return queries

    raise FileNotFoundError("No dataset file found. Provide --dataset or ensure benchmark_queries.json exists.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train MLP Reranker for ARF")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to eval_dataset.json or benchmark_queries.json")
    parser.add_argument("--features-cache", type=str, default=None,
                        help="Path to load/save pre-computed features")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain from cached features (no MongoDB needed)")
    parser.add_argument("--model-output", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Where to save the trained model")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    parser.add_argument("--n-splits", type=int, default=5, help="CV fold count")
    args = parser.parse_args()

    env = args.env or "production"

    print(f"\n{'='*60}")
    print("ARF MLP Reranker Training Pipeline")
    print(f"{'='*60}\n")

    # ----- Step 1: Load or generate features -----
    rows: List[Dict[str, Any]] = []

    if args.retrain and args.features_cache:
        print(f"  Loading cached features from {args.features_cache}")
        rows = load_features_cache(args.features_cache)
    elif args.features_cache and Path(args.features_cache).is_file():
        print(f"  Loading cached features from {args.features_cache}")
        rows = load_features_cache(args.features_cache)
    else:
        print(f"  Generating features from evaluation queries (env={env})...")
        queries = load_queries(args.dataset)
        rows = generate_all_features(queries, env=env)

        if args.features_cache:
            save_features_cache(rows, args.features_cache)

    if not rows:
        print("\n  ERROR: No training data generated. Check your dataset and MongoDB connection.\n")
        sys.exit(1)

    # ----- Step 2: Prepare matrices -----
    # Ensure consistent feature ordering
    feature_names = rows[0]["feature_names"]
    n_features = len(feature_names)

    X = np.array([r["vector"] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int32)
    domains = np.array([r.get("domain", "unknown") for r in rows])

    print(f"\n  Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive: {np.sum(y == 1)} ({np.mean(y) * 100:.1f}%)")
    print(f"  Negative: {np.sum(y == 0)} ({(1 - np.mean(y)) * 100:.1f}%)")
    print(f"  Domains:  {dict(zip(*np.unique(domains, return_counts=True)))}")

    if np.sum(y == 1) < 2 or np.sum(y == 0) < 2:
        print("\n  ERROR: Need at least 2 positive and 2 negative samples for training.\n")
        sys.exit(1)

    # ----- Step 3: Model comparison -----
    print(f"\n  {'='*50}")
    print("  Model Comparison (Stratified {}-Fold CV)".format(args.n_splits))
    print(f"  {'='*50}")

    comparison_results, best_model_name = compare_models(X, y, domains, n_splits=args.n_splits)

    print(f"\n  Model Comparison Results:")
    print(f"  {'Model':<30} {'F1':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    print(f"  {'-'*78}")
    for r in comparison_results:
        agg = r.get("aggregate_metrics", {})
        marker = " <-- BEST" if r["model_name"] == best_model_name else ""
        print(f"  {r['model_name']:<30} {agg.get('f1', 0):>8.4f} {agg.get('auc_roc', 0):>8.4f} "
              f"{agg.get('precision', 0):>8.4f} {agg.get('recall', 0):>8.4f} {agg.get('accuracy', 0):>8.4f}{marker}")

    # ----- Step 4: Train final model -----
    print(f"\n  Training final model ({best_model_name}) on full dataset with calibration...")

    from rag_dependencies.mlp_reranker import MLPReranker

    reranker = MLPReranker()
    best_config = MODEL_CONFIGS[best_model_name]

    if best_config["type"] == "mlp":
        final_metrics = reranker.train(
            X, y,
            hidden_layer_sizes=best_config["hidden_layer_sizes"],
            calibrate=True,
            feature_names=feature_names,
        )
    else:
        # For logistic regression, train via MLPReranker with a single wide layer
        final_metrics = reranker.train(
            X, y,
            hidden_layer_sizes=(max(n_features, 16),),
            max_iter=1000,
            calibrate=True,
            feature_names=feature_names,
        )

    print(f"  Final model metrics: {final_metrics}")

    # ----- Step 5: Calibration quality check -----
    calibration_quality = None
    try:
        probas = reranker.predict(X.tolist())
        probas_arr = np.array(probas)
        # Brier score (lower is better)
        brier = float(np.mean((probas_arr - y) ** 2))
        # Expected calibration error (simple binning)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probas_arr >= bin_edges[i]) & (probas_arr < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(y[mask])
                bin_conf = np.mean(probas_arr[mask])
                ece += np.sum(mask) * abs(bin_acc - bin_conf)
        ece /= len(y)
        calibration_quality = {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
        }
        print(f"  Calibration: Brier={brier:.4f}, ECE={ece:.4f}")
    except Exception as e:
        logger.warning("Calibration quality check failed: %s", e)

    # ----- Step 6: Feature importance -----
    print("  Computing feature importance...")
    feature_importance = compute_feature_importance(X, y, feature_names)
    if feature_importance:
        print(f"  Top 5 features:")
        for fi in feature_importance[:5]:
            print(f"    {fi['feature']:<30} importance={fi['importance_mean']:.4f}")

    # ----- Step 7: Save model -----
    model_path = args.model_output
    reranker.save(model_path)
    print(f"\n  Model saved to: {model_path}")

    # ----- Step 8: Generate and save training report -----
    report = generate_training_report(
        comparison_results=comparison_results,
        best_model_name=best_model_name,
        final_metrics=final_metrics,
        feature_importance=feature_importance,
        feature_names=feature_names,
        n_total_samples=int(X.shape[0]),
        n_positive=int(np.sum(y == 1)),
        calibration_quality=calibration_quality,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Training report saved to: {report_path}")

    # ----- Summary -----
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best model:     {best_model_name}")
    print(f"  F1 score:       {final_metrics.get('f1', 'N/A')}")
    print(f"  AUC-ROC:        {final_metrics.get('auc_roc', 'N/A')}")
    print(f"  Calibrated:     {final_metrics.get('calibrated', False)}")
    print(f"  Total samples:  {X.shape[0]}")
    print(f"  Model path:     {model_path}")
    print(f"  Report path:    {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
