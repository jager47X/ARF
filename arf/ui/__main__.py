"""Serve an ARF pipeline with its user page and admin console.

    python -m arf.ui --demo
    python -m arf.ui --app mypackage.search:build_pipeline --admin-token "$ARF_ADMIN_TOKEN"

``--app`` names a ``module:callable`` that returns a configured
:class:`~arf.pipeline.Pipeline` (or a ``module:attribute`` holding one).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any

from arf.ui.server import create_app


def _load_pipeline(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise SystemExit("--app expects module:callable, e.g. mypackage.search:build_pipeline")
    target = getattr(importlib.import_module(module_name), attr)
    return target() if callable(target) else target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arf.ui", description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="serve the built-in demo pipeline (no keys needed)")
    source.add_argument("--app", metavar="MODULE:CALLABLE", help="factory returning your Pipeline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("ARF_ADMIN_TOKEN"),
        help="token for the admin API (default: $ARF_ADMIN_TOKEN; without one only localhost may use it)",
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is required: pip install advanced-rag-framework[ui]")

    cache_clear_fn = None
    if args.demo:
        from arf.ui.demo import build_demo_pipeline

        pipeline = build_demo_pipeline()
        cache_clear_fn = pipeline.demo_store.clear_cache  # type: ignore[attr-defined]
    else:
        pipeline = _load_pipeline(args.app)

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.admin_token:
        print(
            "WARNING: serving on a non-loopback interface without --admin-token; "
            "the admin API will refuse every remote request.",
            file=sys.stderr,
        )

    app = create_app(pipeline, admin_token=args.admin_token, cache_clear_fn=cache_clear_fn)
    print(f"ARF UI:   http://{args.host}:{args.port}/")
    print(f"Admin:    http://{args.host}:{args.port}/admin")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
