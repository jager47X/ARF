"""
MLP-based relevance reranker for the ARF legal RAG pipeline.

Sits between threshold filtering and LLM verification:
    Vector Search -> Threshold Filter -> MLP Reranker -> LLM Fallback (only for uncertain predictions)

The reranker uses a lightweight scikit-learn MLPClassifier to predict
whether a candidate document is relevant to a query.  Candidates in the
model's "grey zone" (uncertainty_threshold) are forwarded to the
expensive LLM verifier; the rest are accepted or rejected directly,
cutting LLM costs significantly.

Usage:
    reranker = MLPReranker(model_path="models/mlp_reranker.joblib")
    predictions = reranker.predict_with_confidence(feature_vectors)
    # predictions: [{"probability": 0.92, "confident": True, "needs_llm": False}, ...]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default model location (relative to project root)
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "mlp_reranker.joblib")


class MLPReranker:
    """MLP-based relevance reranker for legal RAG pipeline.

    Sits between threshold filtering and LLM verification:
    Vector Search -> Threshold Filter -> MLP Reranker -> LLM Fallback (only for uncertain predictions)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        uncertainty_threshold: Tuple[float, float] = (0.4, 0.6),
    ):
        """Load trained model or initialise empty.

        Args:
            model_path: Path to a joblib-serialised model bundle.  If the
                file exists it is loaded immediately; otherwise the reranker
                starts in an *unloaded* state (``is_loaded == False``).
            uncertainty_threshold: (low, high) probability bounds that define
                the "grey zone".  Predictions inside this range are flagged
                ``needs_llm=True``; predictions outside are considered
                *confident*.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self._model = None  # sklearn MLPClassifier (or CalibratedClassifierCV wrapper)
        self._scaler = None  # sklearn StandardScaler
        self._feature_names: Optional[List[str]] = None
        self._metadata: Dict[str, Any] = {}

        if model_path and Path(model_path).is_file():
            self._load_bundle(model_path)
        elif model_path:
            logger.info("MLPReranker: model file not found at %s — starting unloaded", model_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether a trained model is loaded and ready for inference."""
        return self._model is not None and self._scaler is not None

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Feature names the model was trained on (if available)."""
        return self._feature_names

    @property
    def metadata(self) -> Dict[str, Any]:
        """Training metadata (architecture, metrics, timestamp, etc.)."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, features: List[List[float]]) -> List[float]:
        """Predict relevance probability for feature vectors.

        Args:
            features: List of feature vectors, one per candidate.

        Returns:
            List of probabilities in [0, 1].  Higher means more relevant.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("MLPReranker: no model loaded — call load() or train first")

        start = time.perf_counter()
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)

        # predict_proba returns [[p_class0, p_class1], ...] — we want p(relevant)
        probas = self._model.predict_proba(X_scaled)[:, 1].tolist()

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("MLPReranker.predict: %d candidates in %.1f ms", len(probas), elapsed_ms)
        return probas

    def predict_with_confidence(self, features: List[List[float]]) -> List[Dict[str, Any]]:
        """Predict relevance with confidence flags.

        Returns:
            List of dicts, each containing:
                probability (float): predicted relevance probability
                confident (bool): True when probability is outside the uncertainty zone
                needs_llm (bool): True when probability is inside the uncertainty zone
        """
        probas = self.predict(features)
        low, high = self.uncertainty_threshold
        results = []
        for p in probas:
            confident = p <= low or p >= high
            results.append({
                "probability": round(p, 6),
                "confident": confident,
                "needs_llm": not confident,
            })
        return results

    def score_candidates(self, feature_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score candidates from feature dicts (output of FeatureExtractor).

        Each dict in *feature_dicts* must contain at least a ``"vector"``
        key holding the numeric feature vector.  All other keys are passed
        through to the output.

        Returns:
            Sorted list (highest probability first) of dicts with added keys:
                mlp_score, mlp_confident, mlp_needs_llm
        """
        if not feature_dicts:
            return []

        vectors = [d["vector"] for d in feature_dicts]
        predictions = self.predict_with_confidence(vectors)

        scored = []
        for feat_dict, pred in zip(feature_dicts, predictions):
            entry = dict(feat_dict)
            entry["mlp_score"] = pred["probability"]
            entry["mlp_confident"] = pred["confident"]
            entry["mlp_needs_llm"] = pred["needs_llm"]
            scored.append(entry)

        scored.sort(key=lambda d: d["mlp_score"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Batch training helper (used by benchmarks/train_reranker.py)
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32, 16),
        max_iter: int = 500,
        early_stopping: bool = True,
        calibrate: bool = True,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        activation: str = "relu",
    ) -> Dict[str, Any]:
        """Train a new MLP model in-place.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (0 = not relevant, 1 = relevant).
            hidden_layer_sizes: MLP architecture.
            max_iter: Maximum training iterations.
            early_stopping: Use validation-based early stopping.
            calibrate: Apply isotonic calibration after training.
            feature_names: Optional names for each feature column.
            random_state: Random seed for reproducibility.
            alpha: L2 regularization strength.
            learning_rate_init: Initial learning rate for Adam optimizer.
            activation: Activation function ('relu', 'tanh', 'logistic').

        Returns:
            Dict with training metrics (accuracy, f1, auc, etc.).
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        logger.info(
            "Training MLP: samples=%d, features=%d, arch=%s, alpha=%s, lr=%s, activation=%s, calibrate=%s",
            X.shape[0], X.shape[1], hidden_layer_sizes, alpha, learning_rate_init, activation, calibrate,
        )

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Base MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.1 if early_stopping else 0.0,
            random_state=random_state,
            verbose=False,
        )

        # Cross-validated predictions for metrics
        cv = StratifiedKFold(n_splits=min(5, max(2, int(np.sum(y == 1)))), shuffle=True, random_state=random_state)
        try:
            y_pred_cv = cross_val_predict(mlp, X_scaled, y, cv=cv, method="predict")
            y_proba_cv = cross_val_predict(mlp, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        except ValueError:
            # Fallback if not enough samples per class for CV
            y_pred_cv = y
            y_proba_cv = y.astype(float)

        metrics = {
            "accuracy": round(accuracy_score(y, y_pred_cv), 4),
            "precision": round(precision_score(y, y_pred_cv, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred_cv, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred_cv, zero_division=0), 4),
        }
        if len(np.unique(y)) > 1:
            metrics["auc_roc"] = round(roc_auc_score(y, y_proba_cv), 4)

        # Train final model on full data
        mlp.fit(X_scaled, y)

        # Calibration
        if calibrate and len(np.unique(y)) > 1 and X.shape[0] >= 10:
            logger.info("Applying isotonic calibration...")
            cal_cv = min(3, max(2, int(np.sum(y == 1))))
            calibrated = CalibratedClassifierCV(mlp, method="isotonic", cv=cal_cv)
            calibrated.fit(X_scaled, y)
            model = calibrated
            metrics["calibrated"] = True
        else:
            model = mlp
            metrics["calibrated"] = False

        self._model = model
        self._scaler = scaler
        self._feature_names = feature_names
        self._metadata = {
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "activation": activation,
            "max_iter": max_iter,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "metrics": metrics,
        }

        logger.info("Training complete: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model, scaler, and metadata to a single joblib file."""
        import joblib

        if not self.is_loaded:
            raise RuntimeError("MLPReranker: nothing to save — train or load a model first")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "metadata": self._metadata,
            "uncertainty_threshold": self.uncertainty_threshold,
            "version": 1,
        }
        joblib.dump(bundle, path, compress=3)
        logger.info("MLPReranker saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MLPReranker":
        """Load model from disk.

        Args:
            path: Path to the joblib bundle saved by :meth:`save`.

        Returns:
            A ready-to-use MLPReranker instance.
        """
        instance = cls.__new__(cls)
        instance._load_bundle(path)
        return instance

    def _load_bundle(self, path: str) -> None:
        """Internal: deserialise a saved bundle."""
        import joblib

        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._feature_names = bundle.get("feature_names")
        self._metadata = bundle.get("metadata", {})
        self.uncertainty_threshold = bundle.get("uncertainty_threshold", (0.4, 0.6))
        logger.info(
            "MLPReranker loaded from %s  (features=%s, arch=%s)",
            path,
            self._metadata.get("n_features"),
            self._metadata.get("hidden_layer_sizes"),
        )
