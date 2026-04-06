"""Tests for MLP reranker training, hyperparameter optimization, and threshold tuning."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag_dependencies.mlp_reranker import MLPReranker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Generate reproducible binary classification data for MLP tests."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 10
    X = rng.randn(n_samples, n_features)
    # Create linearly separable-ish labels
    weights = rng.randn(n_features)
    y = (X @ weights > 0).astype(int)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


@pytest.fixture
def trained_reranker(synthetic_data):
    """Return a trained MLPReranker instance."""
    X, y, feature_names = synthetic_data
    reranker = MLPReranker()
    reranker.train(X, y, feature_names=feature_names, max_iter=100)
    return reranker


# ---------------------------------------------------------------------------
# MLPReranker: init, train, save/load
# ---------------------------------------------------------------------------


class TestMLPRerankerInit:
    def test_unloaded_by_default(self):
        reranker = MLPReranker()
        assert not reranker.is_loaded

    def test_missing_model_path_stays_unloaded(self, tmp_path):
        reranker = MLPReranker(model_path=str(tmp_path / "nonexistent.joblib"))
        assert not reranker.is_loaded

    def test_predict_raises_when_unloaded(self):
        reranker = MLPReranker()
        with pytest.raises(RuntimeError, match="no model loaded"):
            reranker.predict([[1.0, 2.0]])


class TestMLPRerankerTrain:
    def test_train_returns_metrics(self, synthetic_data):
        X, y, names = synthetic_data
        reranker = MLPReranker()
        metrics = reranker.train(X, y, feature_names=names, max_iter=100)
        assert reranker.is_loaded
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_train_with_custom_hyperparams(self, synthetic_data):
        X, y, names = synthetic_data
        reranker = MLPReranker()
        metrics = reranker.train(
            X, y,
            feature_names=names,
            hidden_layer_sizes=(32, 16),
            alpha=1e-3,
            learning_rate_init=5e-3,
            activation="tanh",
            max_iter=100,
        )
        assert reranker.is_loaded
        assert reranker.metadata["alpha"] == 1e-3
        assert reranker.metadata["learning_rate_init"] == 5e-3
        assert reranker.metadata["activation"] == "tanh"
        assert reranker.metadata["hidden_layer_sizes"] == [32, 16]

    def test_train_stores_metadata(self, synthetic_data):
        X, y, names = synthetic_data
        reranker = MLPReranker()
        reranker.train(X, y, feature_names=names, max_iter=100)
        meta = reranker.metadata
        assert meta["n_samples"] == X.shape[0]
        assert meta["n_features"] == X.shape[1]
        assert "metrics" in meta


class TestMLPRerankerInference:
    def test_predict_returns_probabilities(self, trained_reranker, synthetic_data):
        X, _, _ = synthetic_data
        probas = trained_reranker.predict(X[:5].tolist())
        assert len(probas) == 5
        for p in probas:
            assert 0.0 <= p <= 1.0

    def test_predict_with_confidence(self, trained_reranker, synthetic_data):
        X, _, _ = synthetic_data
        results = trained_reranker.predict_with_confidence(X[:5].tolist())
        assert len(results) == 5
        for r in results:
            assert "probability" in r
            assert "confident" in r
            assert "needs_llm" in r
            assert r["confident"] != r["needs_llm"]

    def test_uncertainty_thresholds_respected(self, synthetic_data):
        X, y, names = synthetic_data
        reranker = MLPReranker(uncertainty_threshold=(0.3, 0.7))
        reranker.train(X, y, feature_names=names, max_iter=100)
        results = reranker.predict_with_confidence(X[:20].tolist())
        for r in results:
            p = r["probability"]
            if p <= 0.3 or p >= 0.7:
                assert r["confident"] is True
            else:
                assert r["needs_llm"] is True

    def test_score_candidates(self, trained_reranker):
        features = [
            {"vector": [0.1] * 10, "title": "doc_a"},
            {"vector": [0.5] * 10, "title": "doc_b"},
        ]
        scored = trained_reranker.score_candidates(features)
        assert len(scored) == 2
        assert "mlp_score" in scored[0]
        assert "mlp_confident" in scored[0]
        # Should be sorted by score descending
        assert scored[0]["mlp_score"] >= scored[1]["mlp_score"]

    def test_score_candidates_empty(self, trained_reranker):
        assert trained_reranker.score_candidates([]) == []


class TestMLPRerankerSaveLoad:
    def test_save_and_load(self, trained_reranker, synthetic_data):
        X, _, _ = synthetic_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.joblib")
            trained_reranker.save(path)
            assert Path(path).is_file()

            loaded = MLPReranker.load(path)
            assert loaded.is_loaded
            assert loaded.feature_names == trained_reranker.feature_names

            # Predictions should match
            orig = trained_reranker.predict(X[:3].tolist())
            new = loaded.predict(X[:3].tolist())
            np.testing.assert_allclose(orig, new, atol=1e-6)

    def test_save_raises_when_unloaded(self):
        reranker = MLPReranker()
        with pytest.raises(RuntimeError, match="nothing to save"):
            reranker.save("/tmp/fail.joblib")


# ---------------------------------------------------------------------------
# Hyperparameter optimization (from benchmarks/train_reranker.py)
# ---------------------------------------------------------------------------


class TestOptimizeHyperparameters:
    def test_returns_best_params(self, synthetic_data):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from train_reranker import optimize_hyperparameters

        X, y, _ = synthetic_data
        result = optimize_hyperparameters(X, y, n_iter=5, n_splits=3)

        assert "best_params" in result
        assert "best_score" in result
        assert "top_results" in result
        assert result["best_score"] > 0.0
        assert len(result["top_results"]) <= 10

        # best_params should contain expected keys
        for key in ("hidden_layer_sizes", "activation", "alpha", "learning_rate_init", "max_iter"):
            assert key in result["best_params"], f"Missing key: {key}"

    def test_different_scoring_metrics(self, synthetic_data):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from train_reranker import optimize_hyperparameters

        X, y, _ = synthetic_data
        for metric in ("f1", "roc_auc"):
            result = optimize_hyperparameters(X, y, n_iter=3, n_splits=2, scoring=metric)
            assert result["scoring_metric"] == metric
            assert result["best_score"] > 0.0


# ---------------------------------------------------------------------------
# Threshold optimization (from benchmarks/train_reranker.py)
# ---------------------------------------------------------------------------


class TestOptimizeThresholds:
    def test_returns_per_domain_thresholds(self, synthetic_data):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from train_reranker import optimize_thresholds

        X, y, _ = synthetic_data
        # Create domain labels
        rng = np.random.RandomState(42)
        domains = np.array(["domain_a"] * 100 + ["domain_b"] * 100)

        # Train a model to get model + scaler
        reranker = MLPReranker()
        reranker.train(X, y, max_iter=100)

        result = optimize_thresholds(X, y, domains, reranker._model, reranker._scaler)

        assert "overall" in result
        for domain in ("domain_a", "domain_b", "overall"):
            assert domain in result
            entry = result[domain]
            assert "mlp_uncertainty_low" in entry
            assert "mlp_uncertainty_high" in entry
            assert entry["mlp_uncertainty_low"] < entry["mlp_uncertainty_high"]
            assert 0.0 <= entry["confident_accuracy"] <= 1.0
            assert 0.0 <= entry["coverage"] <= 1.0

    def test_skips_small_domains(self, synthetic_data):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from train_reranker import optimize_thresholds

        X, y, _ = synthetic_data
        # domain_c has only 5 samples — should be skipped
        domains = np.array(["domain_a"] * 100 + ["domain_b"] * 95 + ["domain_c"] * 5)

        reranker = MLPReranker()
        reranker.train(X, y, max_iter=100)

        result = optimize_thresholds(X, y, domains, reranker._model, reranker._scaler)
        assert "domain_c" not in result


# ---------------------------------------------------------------------------
# Config schema validation for MLP fields
# ---------------------------------------------------------------------------


class TestConfigSchemaMLP:
    def test_mlp_fields_validated(self):
        from config_schema import validate_thresholds

        thresholds = validate_thresholds()
        for domain, t in thresholds.items():
            if t.mlp_uncertainty_low is not None and t.mlp_uncertainty_high is not None:
                assert t.mlp_uncertainty_low < t.mlp_uncertainty_high

    def test_mlp_hidden_layer_sizes_is_string(self):
        from config_schema import DomainThresholds

        # Minimal valid thresholds + MLP string field
        t = DomainThresholds(
            query_search=0.5, alias_search=0.5, RAG_SEARCH_min=0.3,
            LLM_VERIFication=0.4, RAG_SEARCH=0.7, confident=0.8,
            FILTER_GAP=0.1, LLM_SCORE=0.5,
            mlp_hidden_layer_sizes="128,64,32",
            mlp_activation="relu",
        )
        assert t.mlp_hidden_layer_sizes == "128,64,32"
        assert t.mlp_activation == "relu"
