"""Optional web UI for an ARF pipeline: a user search page and an admin console.

Requires the ``ui`` extra::

    pip install advanced-rag-framework[ui]      # fastapi + uvicorn

Try it with no API keys or database::

    python -m arf.ui --demo                      # http://127.0.0.1:8000  (admin at /admin)

Serve your own pipeline::

    from arf.ui import create_app
    app = create_app(my_pipeline, admin_token="...")   # then: uvicorn mymodule:app

    # or: python -m arf.ui --app mymodule:build_pipeline --admin-token ...
"""

from arf.ui.server import create_app

__all__ = ["create_app"]
