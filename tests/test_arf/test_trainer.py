"""Tests for arf.trainer — train_reranker and load_reranker.

These tests require numpy and scikit-learn.  They are skipped if those
packages are not installed.
"""

import pytest

try:
    import numpy as np
    import sklearn  # noqa: F401

    HAS_ML = True
except ImportError:
    HAS_ML = False

pytestmark = pytest.mark.skipif(not HAS_ML, reason="numpy/sklearn not installed")


def test_train_basic():
    from arf.trainer import train_reranker

    rng = np.random.RandomState(42)
    # 20 samples so stratified CV works
    X_pos = rng.uniform(0.6, 1.0, size=(10, 5))
    X_neg = rng.uniform(0.0, 0.4, size=(10, 5))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 10 + [0] * 10)

    result = train_reranker(X, y, architecture=(8, 4), max_iter=50, calibrate=False)
    assert "metrics" in result
    assert "model" in result
    assert "scaler" in result
    assert result["metrics"]["accuracy"] >= 0.0


def test_train_and_save(tmp_path):
    from arf.trainer import load_reranker, train_reranker

    rng = np.random.RandomState(42)
    X_pos = rng.uniform(0.6, 1.0, size=(10, 2))
    X_neg = rng.uniform(0.0, 0.4, size=(10, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 10 + [0] * 10)
    path = str(tmp_path / "model.joblib")

    train_reranker(X, y, architecture=(4,), max_iter=50, calibrate=False, save_path=path)

    predict_fn = load_reranker(path)
    probs = predict_fn([[0.9, 0.8], [0.1, 0.2]])
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
