"""Optional cross-encoder re-rankers for the :class:`~arf.pipeline.Pipeline` *rerank_fn* slot.

A cross-encoder reads the query and the document *together* and is usually a
stronger relevance judge than features + MLP, at a small fraction of an LLM
call.  ARF keeps it optional: this module imports *sentence-transformers*
lazily, so the core package stays dependency-free::

    pip install advanced-rag-framework[rerank]

Usage::

    from arf.rerankers import sentence_transformers_reranker

    pipeline = Pipeline(
        ...,
        rerank_fn=sentence_transformers_reranker(config=my_doc_config),
        rerank_zones=(0.3, 0.7),
    )

Any other scorer works too — wrap it with :func:`cross_encoder_rerank_fn` if
it exposes ``predict(pairs)``, or write a ``(query, items) -> [prob, ...]``
callable directly.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RerankFn = Callable[[str, Sequence[Any]], List[float]]


def document_text_fn(config: Optional[Any] = None, max_chars: int = 2000) -> Callable[[Any], str]:
    """Build an ``(item) -> str`` that joins the configured title and first text field.

    Honours ``DocumentConfig.title_field`` / ``text_fields`` for dict items and
    reads ``.title`` / ``.content`` from :class:`~arf.document.Document`
    objects.  The text is truncated to *max_chars* — cross-encoders have a
    short context window and the opening of a document carries most of its
    signal.
    """
    title_field = getattr(config, "title_field", None) or "title"
    text_fields = list(getattr(config, "text_fields", None) or ["text", "summary", "content", "body"])

    def text_of(item: Any) -> str:
        if isinstance(item, dict):
            title = str(item.get(title_field) or "")
            body = ""
            for fld in text_fields:
                val = item.get(fld)
                if isinstance(val, str) and val.strip():
                    body = val.strip()
                    break
        else:
            title = str(getattr(item, "title", "") or "")
            body = str(getattr(item, "content", "") or "")
        return f"{title}\n{body}".strip()[:max_chars]

    return text_of


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cross_encoder_rerank_fn(model: Any, *, text_fn: Callable[[Any], str], batch_size: int = 32) -> RerankFn:
    """Wrap any object with ``predict([(query, text), ...]) -> scores`` as a *rerank_fn*.

    Scores that fall outside [0, 1] (raw logits) are squashed with a sigmoid,
    so the result always works with ``Pipeline(rerank_zones=...)``.
    """

    def rerank_fn(query: str, items: Sequence[Any]) -> List[float]:
        if not items:
            return []
        pairs = [(query, text_fn(item)) for item in items]
        raw = model.predict(pairs, batch_size=batch_size)
        scores = [float(s) for s in (raw.tolist() if hasattr(raw, "tolist") else raw)]
        if any(s < 0.0 or s > 1.0 for s in scores):
            scores = [_sigmoid(s) for s in scores]
        return scores

    return rerank_fn


def sentence_transformers_reranker(
    model_name: str = DEFAULT_MODEL,
    *,
    config: Optional[Any] = None,
    text_fn: Optional[Callable[[Any], str]] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    max_length: int = 512,
) -> RerankFn:
    """Load a sentence-transformers ``CrossEncoder`` and return it as a *rerank_fn*.

    Args:
        model_name: Hugging Face model id (default: a 22M-parameter MS MARCO
            cross-encoder that scores ~1k pairs/s on CPU).
        config: :class:`~arf.document.DocumentConfig` used to build the
            document text (ignored when *text_fn* is given).
        text_fn: Custom ``(item) -> str``.
        batch_size: Pairs per forward pass.
        device: ``"cpu"``, ``"cuda"``, ``"mps"`` ... (auto when ``None``).
        max_length: Token limit per (query, document) pair.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for sentence_transformers_reranker.  "
            "Install it with:  pip install advanced-rag-framework[rerank]"
        ) from exc

    model = CrossEncoder(model_name, max_length=max_length, device=device)
    return cross_encoder_rerank_fn(model, text_fn=text_fn or document_text_fn(config), batch_size=batch_size)
