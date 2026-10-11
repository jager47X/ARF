"""Tests for arf.ui — the FastAPI app, the admin protection and the demo pipeline."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from arf.ui import create_app  # noqa: E402
from arf.ui.demo import DOCS, build_demo_pipeline, hash_embed  # noqa: E402
from arf.ui.server import apply_config, pipeline_config  # noqa: E402

TOKEN = "s3cret"


@pytest.fixture
def pipeline():
    return build_demo_pipeline()


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline, admin_token=TOKEN))


def admin(path, **kw):
    kw.setdefault("headers", {})["X-ARF-Admin-Token"] = TOKEN
    return kw


# -- demo pipeline ------------------------------------------------------------


def test_demo_pipeline_answers_without_any_keys(pipeline):
    results = pipeline.run("freedom of speech", top_k=5)
    assert results and results[0]["document"].id == "us-const-1a"
    assert results[0]["decided_by"] in {"threshold", "mlp", "rerank", "llm", "unreviewed", "llm_error"}
    assert pipeline.last_stats.fetched >= 1
    second = pipeline.run("freedom of speech", top_k=5)
    assert pipeline.last_stats.cache_hit is True
    assert [r["document"].id for r in second] == [r["document"].id for r in results]


def test_demo_embeddings_are_deterministic_and_normalised():
    a, b = hash_embed("due process of law"), hash_embed("due process of law")
    assert a == b and abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert len({d["_id"] for d in DOCS}) == len(DOCS)


# -- user surface -------------------------------------------------------------


def test_pages_render(client):
    assert "<title>ARF Search</title>" in client.get("/").text
    assert "<title>ARF Admin</title>" in client.get("/admin").text
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["slots"]["llm_fn"] is True and health["slots"]["rerank_fn"] is False


def test_query_endpoint_returns_serialised_results_and_stats(client):
    data = client.post("/api/query", json={"query": "unreasonable searches and seizures", "top_k": 3}).json()
    assert data["query"] == "unreasonable searches and seizures"
    assert 1 <= len(data["results"]) <= 3
    first = data["results"][0]
    assert first["id"] == "us-const-4a" and first["title"] == "Fourth Amendment"
    assert set(first) >= {
        "id",
        "title",
        "path",
        "snippet",
        "score",
        "llm_score",
        "rerank_score",
        "decided_by",
        "summary",
        "context",
    }
    assert "embedding" not in first["metadata"]
    assert data["stats"]["fetched"] >= 1 and "total" in data["stats"]["timings_ms"]


def test_query_validation(client):
    assert client.post("/api/query", json={"query": "", "top_k": 3}).status_code == 422
    assert client.post("/api/query", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/query", json={"query": "x", "top_k": 1000}).status_code == 422


# -- admin protection ---------------------------------------------------------


def test_admin_api_requires_the_token(client):
    assert client.get("/admin/api/stats").status_code == 401
    assert client.get("/admin/api/stats", headers={"X-ARF-Admin-Token": "wrong"}).status_code == 401
    ok = client.get("/admin/api/stats", **admin("/admin/api/stats"))
    assert ok.status_code == 200 and {"summary", "recent", "config"} <= set(ok.json())


def test_admin_api_without_token_is_loopback_only(pipeline, monkeypatch):
    monkeypatch.delenv("ARF_ADMIN_TOKEN", raising=False)
    app = create_app(pipeline)
    remote = TestClient(app)  # client host "testclient"
    assert remote.get("/admin/api/stats").status_code == 403
    local = TestClient(app, client=("127.0.0.1", 50000))
    assert local.get("/admin/api/stats").status_code == 200


def test_admin_token_can_come_from_the_environment(pipeline, monkeypatch):
    monkeypatch.setenv("ARF_ADMIN_TOKEN", "from-env")
    c = TestClient(create_app(pipeline))
    assert c.get("/admin/api/config").status_code == 401
    assert c.get("/admin/api/config", headers={"X-ARF-Admin-Token": "from-env"}).status_code == 200


# -- admin features -----------------------------------------------------------


def test_stats_runs_and_reset(client):
    for query in ("freedom of speech", "right to vote"):
        client.post("/api/query", json={"query": query})
    data = client.get("/admin/api/stats", **admin("")).json()
    assert data["summary"]["runs"] == 2 and data["summary"]["funnel"]["fetched"] > 0
    runs = client.get("/admin/api/runs?limit=10", **admin("")).json()["runs"]
    assert [r["query"] for r in runs] == ["freedom of speech", "right to vote"]
    assert client.post("/admin/api/stats/reset", **admin("")).json() == {"ok": True}
    assert client.get("/admin/api/stats", **admin("")).json()["summary"]["runs"] == 0


def test_config_roundtrip_and_validation(client, pipeline):
    cfg = client.get("/admin/api/config", **admin("")).json()
    assert cfg["triage"]["accept_threshold"] == pipeline.triage.accept_threshold
    assert cfg["slots"]["predict_fn"] is True and cfg["llm_mode"] == "gate"

    updated = client.put(
        "/admin/api/config",
        json={
            "triage": {"accept_threshold": 0.5, "verify_threshold": 0.2},
            "llm_pass_score": 7,
            "predict_zones": [0.3, 0.8],
            "cache_version": "v2",
            "max_candidates": None,
        },
        **admin(""),
    )
    assert updated.status_code == 200, updated.text
    assert pipeline.triage.accept_threshold == 0.5 and pipeline.triage.verify_threshold == 0.2
    assert pipeline.llm_pass_score == 7 and pipeline.predict_zones == (0.3, 0.8)
    assert pipeline.cache_version == "v2" and pipeline.max_candidates is None

    bad = [
        {"triage": {"accept_threshold": 0.1}},  # below verify_threshold
        {"predict_zones": [0.8, 0.3]},
        {"llm_mode": "vote"},
        {"llm_pass_score": 12},
        {"llm_workers": None},
        {"search_fn": None},
        {"triage": {"nope": 1}},
        {"cache_version": 5},
    ]
    for body in bad:
        r = client.put("/admin/api/config", json=body, **admin(""))
        assert r.status_code == 422, body
    # Nothing from the rejected bodies leaked through
    assert pipeline.triage.accept_threshold == 0.5 and pipeline.llm_pass_score == 7


def test_apply_config_is_atomic(pipeline):
    before = pipeline_config(pipeline)
    with pytest.raises(fastapi.HTTPException):
        apply_config(pipeline, {"llm_pass_score": 3, "llm_mode": "nope"})
    assert pipeline_config(pipeline) == before


def test_evaluate_endpoint(client):
    body = {
        "dataset": [
            {"query": "freedom of speech", "relevant": ["us-const-1a"]},
            {"query": "right to vote", "relevant": {"us-const-15a": 2, "us-const-19a": 2}},
        ],
        "ks": [1, 3],
    }
    r = client.post("/admin/api/evaluate", json=body, **admin(""))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_queries"] == 2 and data["metrics"]["hit_rate@3"] > 0
    assert "| metric | value |" in data["markdown"]
    assert data["stats"]["runs"] == 2

    assert client.post("/admin/api/evaluate", json={"dataset": [{"relevant": []}]}, **admin("")).status_code == 422
    assert client.post("/admin/api/evaluate", json={"dataset": []}, **admin("")).status_code == 422


def test_cache_clear_endpoint(pipeline):
    without = TestClient(create_app(pipeline, admin_token=TOKEN))
    assert without.post("/admin/api/cache/clear", **admin("")).status_code == 501

    store = pipeline.demo_store
    with_fn = TestClient(create_app(pipeline, admin_token=TOKEN, cache_clear_fn=store.clear_cache))
    with_fn.post("/api/query", json={"query": "freedom of speech"})
    assert store.cache
    assert with_fn.post("/admin/api/cache/clear", **admin("")).json() == {"ok": True}
    assert store.cache == {}
