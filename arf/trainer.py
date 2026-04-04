"""MLP reranker training and loading utilities.

Provides functions to train a lightweight MLP classifier for relevance
scoring and to load a trained model as a ``predict_fn`` compatible with
:class:`~arf.pipeline.Pipeline`.

Requires *numpy* and *scikit-learn* — both are optional dependencies of
the ``arf`` package.  Install them with::

    pip install arf[ml]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SKLEARN_ERROR = (
    "scikit-learn is required for MLP training.  "
    "Install it with:  pip install arf[ml]"
)


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(_SKLEARN_ERROR) from exc


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for MLP training.  Install it with:  pip install arf[ml]"
        ) from exc


def train_reranker(
    X: Any,
    y: Any,
    *,
    architecture: Tuple[int, ...] = (64, 32, 16),
    max_iter: int = 500,
    early_stopping: bool = True,
    calibrate: bool = True,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train an MLP relevance classifier.

    Args:
        X: Feature matrix (n_samples, n_features).  Accepts a numpy array
            or a list of lists.
        y: Binary labels (0 = not relevant, 1 = relevant).
        architecture: Hidden layer sizes for the MLP.
        max_iter: Maximum training iterations.
        early_stopping: Use validation-based early stopping.
        calibrate: Apply isotonic calibration after training.
        feature_names: Optional ordered list of feature names.
        random_state: Random seed for reproducibility.
        save_path: If provided, save the model bundle to this path.

    Returns:
        Dict with keys ``"metrics"`` (accuracy, precision, recall, f1,
        auc_roc), ``"model"``, ``"scaler"``, and ``"metadata"``.
    """
    np = _require_numpy()
    _require_sklearn()

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

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    logger.info(
        "Training MLP: samples=%d, features=%d, arch=%s, calibrate=%s",
        X.shape[0],
        X.shape[1],
        architecture,
        calibrate,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.1 if early_stopping else 0.0,
        random_state=random_state,
        verbose=False,
    )

    # Cross-validated predictions for metrics
    n_splits = min(5, max(2, int(np.sum(y == 1))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        y_pred_cv = cross_val_predict(mlp, X_scaled, y, cv=cv, method="predict")
        y_proba_cv = cross_val_predict(mlp, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    except ValueError:
        y_pred_cv = y
        y_proba_cv = y.astype(float)

    metrics: Dict[str, Any] = {
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
        cal_cv = min(3, max(2, int(np.sum(y == 1))))
        calibrated = CalibratedClassifierCV(mlp, method="isotonic", cv=cal_cv)
        calibrated.fit(X_scaled, y)
        model = calibrated
        metrics["calibrated"] = True
    else:
        model = mlp
        metrics["calibrated"] = False

    metadata = {
        "hidden_layer_sizes": list(architecture),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "metrics": metrics,
    }

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metadata": metadata,
        "version": 1,
    }

    if save_path:
        import joblib

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, save_path, compress=3)
        logger.info("Model saved to %s", save_path)

    logger.info("Training complete: %s", metrics)
    return {**bundle, "metrics": metrics}


def load_reranker(
    path: str,
    *,
    uncertainty_threshold: Tuple[float, float] = (0.4, 0.6),
) -> Callable[[List[List[float]]], List[float]]:
    """Load a trained model and return a ``predict_fn``.

    The returned callable accepts a list of feature vectors and returns
    a list of relevance probabilities.  It is directly compatible with
    :class:`~arf.pipeline.Pipeline`'s *predict_fn* parameter.

    Args:
        path: Path to a joblib bundle saved by :func:`train_reranker`.
        uncertainty_threshold: Not used by the returned function but
            stored for reference.

    Returns:
        ``(feature_vectors) -> [float, ...]``
    """
    np = _require_numpy()
    import joblib

    bundle = joblib.load(path)
    model = bundle["model"]
    scaler = bundle["scaler"]

    def predict_fn(features: List[List[float]]) -> List[float]:
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = scaler.transform(X)
        return model.predict_proba(X_scaled)[:, 1].tolist()

    return predict_fn
