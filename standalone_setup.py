"""
Import shim for running ARF standalone (outside the parent services package).

When ARF lives inside a larger project, modules import from `services.rag.*`.
This shim creates that path alias so ARF can run directly from its own directory.

Usage — import this BEFORE importing any ARF modules:

    import standalone_setup   # patches sys.modules
    from RAG_interface import RAG
    from config import COLLECTION
"""

import sys
import types
from pathlib import Path

_ARF_ROOT = Path(__file__).resolve().parent


def _setup():
    if "services" in sys.modules and "services.rag" in sys.modules:
        return  # already configured (running inside parent project)

    # Ensure ARF root is on sys.path
    root_str = str(_ARF_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Create services package
    if "services" not in sys.modules:
        services = types.ModuleType("services")
        services.__path__ = []
        sys.modules["services"] = services
    else:
        services = sys.modules["services"]

    # Create services.rag pointing to ARF root
    if "services.rag" not in sys.modules:
        rag = types.ModuleType("services.rag")
        rag.__path__ = [root_str]
        rag.__file__ = str(_ARF_ROOT / "__init__.py")
        sys.modules["services.rag"] = rag
        services.rag = rag

    # Create services.rag.rag_dependencies pointing to rag_dependencies/
    deps_dir = str(_ARF_ROOT / "rag_dependencies")
    if "services.rag.rag_dependencies" not in sys.modules:
        deps = types.ModuleType("services.rag.rag_dependencies")
        deps.__path__ = [deps_dir]
        deps.__file__ = str(_ARF_ROOT / "rag_dependencies" / "__init__.py")
        sys.modules["services.rag.rag_dependencies"] = deps

    # Create services.rag.preprocess pointing to preprocess/
    preprocess_dir = str(_ARF_ROOT / "preprocess")
    if "services.rag.preprocess" not in sys.modules:
        preprocess = types.ModuleType("services.rag.preprocess")
        preprocess.__path__ = [preprocess_dir]
        sys.modules["services.rag.preprocess"] = preprocess


_setup()
