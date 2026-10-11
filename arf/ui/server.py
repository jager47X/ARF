"""FastAPI application exposing a :class:`~arf.pipeline.Pipeline`.

Two surfaces:

* **User** — ``/`` (search page) and ``POST /api/query``: run a query and see
  each result's score, the stage that accepted it, its summary and context.
* **Admin** — ``/admin`` (console) and ``/admin/api/*``: live statistics
  (funnel, LLM share, cache hit rate, stage latency, errors), a live
  threshold / policy editor, recent runs, an evaluation runner and cache tools.

The admin API is protected by a token passed in the ``X-ARF-Admin-Token``
header (``admin_token=`` or the ``ARF_ADMIN_TOKEN`` environment variable).
Without a token it only answers requests from the loopback interface.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "fastapi is required for arf.ui.  Install it with:  pip install advanced-rag-framework[ui]"
    ) from exc

from arf import __version__
from arf.document import Document
from arf.evaluation import evaluate
from arf.pipeline import Pipeline

_STATIC = Path(__file__).parent / "static"
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_SLOTS = (
    "predict_fn",
    "rerank_fn",
    "llm_fn",
    "keyword_fn",
    "alias_fn",
    "expand_fn",
    "rephrase_fn",
    "cache_lookup",
    "cache_store",
    "cache_nearest_fn",
    "summarize_fn",
    "resolve_fn",
    "moderate_fn",
    "preprocess_fn",
)
_POLICY_FIELDS = {
    "llm_mode": Pipeline._LLM_MODES,
    "llm_error_policy": Pipeline._LLM_ERROR_POLICIES,
    "unreviewed_policy": Pipeline._UNREVIEWED_POLICIES,
    "keyword_mode": Pipeline._KEYWORD_MODES,
}
# name -> (type, minimum, maximum, nullable)
_NUMERIC_FIELDS = {
    "llm_pass_score": (int, 0, 9, False),
    "llm_workers": (int, 1, 64, False),
    "keyword_floor": (float, 0.0, 1.0, True),
    "keyword_boost": (float, 0.0, 1.0, False),
    "candidate_multiplier": (float, 1.0, 50.0, False),
    "max_candidates": (int, 1, 10000, True),
    "max_expansions": (int, 0, 20, False),
    "max_rephrase": (int, 0, 10, False),
    "graph_max_hops": (int, 0, 20, False),
    "cache_ttl": (float, 1.0, 10**9, True),
    "cache_similarity_threshold": (float, 0.0, 1.0, False),
}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)


class EvaluateRequest(BaseModel):
    dataset: List[Dict[str, Any]] = Field(min_length=1, max_length=500)
    ks: List[int] = Field(default=[1, 3, 5, 10], min_length=1, max_length=10)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)


def _page(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def _scalar_metadata(doc: Document) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in doc.metadata.items():
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str) and len(value) <= 200:
            out[key] = value
    return out


def serialize_result(result: Dict[str, Any], snippet_chars: int = 320) -> Dict[str, Any]:
    """JSON-friendly view of one ``Pipeline.run`` result."""
    doc = result.get("document")
    if isinstance(doc, Document):
        content = doc.content or ""
        info = {
            "id": doc.id,
            "title": doc.title,
            "path": doc.path,
            "depth": doc.depth,
            "snippet": content[:snippet_chars] + ("…" if len(content) > snippet_chars else ""),
            "metadata": _scalar_metadata(doc),
            "context": [c.title or c.id for c in result.get("context") or [] if isinstance(c, Document)],
        }
    else:
        info = {"id": str(doc), "title": str(doc), "path": "", "depth": 0, "snippet": "", "metadata": {}, "context": []}
    info.update(
        score=round(float(result.get("score", 0.0)), 4),
        llm_score=result.get("llm_score"),
        rerank_score=None if result.get("rerank_score") is None else round(float(result["rerank_score"]), 4),
        decided_by=result.get("decided_by"),
        summary=result.get("summary") or "",
    )
    return info


def pipeline_config(pipeline: Pipeline) -> Dict[str, Any]:
    """The editable settings of *pipeline* plus which optional slots are wired."""
    t = pipeline.triage
    cfg: Dict[str, Any] = {
        "triage": {
            "min_score": t.min_score,
            "verify_threshold": t.verify_threshold,
            "accept_threshold": t.accept_threshold,
            "gap": t.gap,
        },
        "predict_zones": list(pipeline.predict_zones),
        "rerank_zones": list(pipeline.rerank_zones),
    }
    for name in list(_POLICY_FIELDS) + list(_NUMERIC_FIELDS) + ["cache_version"]:
        cfg[name] = getattr(pipeline, name)
    cfg["slots"] = {name: getattr(pipeline, name) is not None for name in _SLOTS}
    cfg["version"] = __version__
    return cfg


def _number(name: str, value: Any, kind: type, lo: float, hi: float, nullable: bool) -> Any:
    if value is None:
        if nullable:
            return None
        raise HTTPException(status_code=422, detail=f"{name} cannot be null")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=422, detail=f"{name} must be a number")
    number = kind(value)
    if not lo <= number <= hi:
        raise HTTPException(status_code=422, detail=f"{name} must be between {lo} and {hi}")
    return number


def _zones(name: str, value: Any) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise HTTPException(status_code=422, detail=f"{name} must be [low, high]")
    low = _number(f"{name}[0]", value[0], float, 0.0, 1.0, False)
    high = _number(f"{name}[1]", value[1], float, 0.0, 1.0, False)
    if not low < high:
        raise HTTPException(status_code=422, detail=f"{name}: low must be below high")
    return (low, high)


def apply_config(pipeline: Pipeline, changes: Dict[str, Any]) -> Dict[str, Any]:
    """Validate *changes* (a partial config dict) and apply them atomically."""
    updates: Dict[str, Any] = {}
    triage_updates: Dict[str, float] = {}
    for name, value in changes.items():
        if name == "triage":
            if not isinstance(value, dict):
                raise HTTPException(status_code=422, detail="triage must be an object")
            for key, raw in value.items():
                if key not in ("min_score", "verify_threshold", "accept_threshold", "gap"):
                    raise HTTPException(status_code=422, detail=f"unknown triage field {key!r}")
                triage_updates[key] = _number(f"triage.{key}", raw, float, 0.0, 1.0, False)
        elif name in ("predict_zones", "rerank_zones"):
            updates[name] = _zones(name, value)
        elif name in _POLICY_FIELDS:
            if value not in _POLICY_FIELDS[name]:
                raise HTTPException(status_code=422, detail=f"{name} must be one of {list(_POLICY_FIELDS[name])}")
            updates[name] = value
        elif name in _NUMERIC_FIELDS:
            kind, lo, hi, nullable = _NUMERIC_FIELDS[name]
            updates[name] = _number(name, value, kind, lo, hi, nullable)
        elif name == "cache_version":
            if value is not None and not isinstance(value, str):
                raise HTTPException(status_code=422, detail="cache_version must be a string or null")
            updates[name] = value or None
        else:
            raise HTTPException(status_code=422, detail=f"unknown or read-only field {name!r}")

    t = pipeline.triage
    new_min = triage_updates.get("min_score", t.min_score)
    new_verify = triage_updates.get("verify_threshold", t.verify_threshold)
    new_accept = triage_updates.get("accept_threshold", t.accept_threshold)
    if not new_min <= new_verify <= new_accept:
        raise HTTPException(
            status_code=422, detail="triage thresholds must satisfy min_score <= verify_threshold <= accept_threshold"
        )

    for key, value in triage_updates.items():
        setattr(t, key, value)
    for name, value in updates.items():
        setattr(pipeline, name, value)
    return pipeline_config(pipeline)


def create_app(
    pipeline: Pipeline,
    *,
    admin_token: Optional[str] = None,
    cache_clear_fn: Optional[Callable[[], Any]] = None,
    title: str = "ARF",
) -> FastAPI:
    """Build the FastAPI app for *pipeline*.

    Args:
        pipeline: The pipeline to serve.
        admin_token: Secret required in the ``X-ARF-Admin-Token`` header for
            every ``/admin/api/*`` call.  Defaults to the ``ARF_ADMIN_TOKEN``
            environment variable; with neither set, the admin API only
            answers loopback clients.
        cache_clear_fn: Optional ``() -> None`` that empties the cache
            backend, exposed as ``POST /admin/api/cache/clear``.
        title: App title.
    """
    token = admin_token if admin_token is not None else (os.environ.get("ARF_ADMIN_TOKEN") or None)
    app = FastAPI(title=title, version=__version__)
    app.state.pipeline = pipeline

    def require_admin(request: Request) -> None:
        if token:
            supplied = request.headers.get("x-arf-admin-token") or ""
            if not secrets.compare_digest(supplied.encode(), token.encode()):
                raise HTTPException(status_code=401, detail="Admin token required (X-ARF-Admin-Token header)")
            return
        host = request.client.host if request.client else ""
        if host not in _LOOPBACK:
            raise HTTPException(
                status_code=403,
                detail="Admin API is limited to localhost unless an admin token is configured (ARF_ADMIN_TOKEN)",
            )

    admin = [Depends(require_admin)]

    # ---- user -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def user_page() -> str:
        return _page("user.html")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "slots": pipeline_config(pipeline)["slots"]}

    @app.post("/api/query")
    def query(req: QueryRequest) -> Dict[str, Any]:
        results = pipeline.run(req.query, top_k=req.top_k)
        stats = pipeline.last_stats
        return {
            "query": req.query,
            "results": [serialize_result(r) for r in results],
            "stats": stats.to_dict() if stats is not None else None,
        }

    # ---- admin ------------------------------------------------------------

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_page() -> str:
        return _page("admin.html")

    @app.get("/admin/api/stats", dependencies=admin)
    def admin_stats(limit: int = 50) -> Dict[str, Any]:
        return {
            "summary": pipeline.stats.summary(),
            "recent": [r.to_dict() for r in pipeline.stats.recent(max(1, min(limit, 500)))],
            "config": pipeline_config(pipeline),
        }

    @app.post("/admin/api/stats/reset", dependencies=admin)
    def admin_reset_stats() -> Dict[str, Any]:
        pipeline.stats.reset()
        return {"ok": True}

    @app.get("/admin/api/runs", dependencies=admin)
    def admin_runs(limit: int = 50) -> Dict[str, Any]:
        return {"runs": [r.to_dict() for r in pipeline.stats.recent(max(1, min(limit, 500)))]}

    @app.get("/admin/api/config", dependencies=admin)
    def admin_config() -> Dict[str, Any]:
        return pipeline_config(pipeline)

    @app.put("/admin/api/config", dependencies=admin)
    def admin_update_config(changes: Dict[str, Any]) -> Dict[str, Any]:
        return apply_config(pipeline, changes)

    @app.post("/admin/api/evaluate", dependencies=admin)
    def admin_evaluate(req: EvaluateRequest) -> Dict[str, Any]:
        for entry in req.dataset:
            if not isinstance(entry.get("query"), str) or not entry["query"].strip():
                raise HTTPException(status_code=422, detail="every dataset entry needs a non-empty 'query'")
        report = evaluate(pipeline, req.dataset, ks=tuple(req.ks), top_k=req.top_k)
        data = report.to_dict()
        data["markdown"] = report.markdown()
        return data

    @app.post("/admin/api/cache/clear", dependencies=admin)
    def admin_cache_clear() -> Dict[str, Any]:
        if cache_clear_fn is None:
            raise HTTPException(
                status_code=501,
                detail="No cache_clear_fn configured; change cache_version to invalidate existing entries",
            )
        cache_clear_fn()
        return {"ok": True}

    return app
