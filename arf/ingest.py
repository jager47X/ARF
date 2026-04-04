"""Document ingestion helper.

Validates documents, computes hierarchy metadata (depth, path), generates
embeddings (via the user's *embed_fn*), and stores enriched documents (via
the user's *store_fn*).  Supports parent-level and child-level embeddings
so that vector search works at multiple granularities.

ARF never touches the database directly — all I/O goes through the
user-provided callables.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from arf.document import DocumentConfig

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Summary of an ingestion run.

    Attributes:
        processed: Number of documents successfully ingested.
        skipped: Number of documents skipped (e.g. missing required fields).
        errors: Number of documents that caused an error.
        error_details: List of ``(index, error_message)`` for failed docs.
    """

    processed: int = 0
    skipped: int = 0
    errors: int = 0
    error_details: List[tuple] = field(default_factory=list)


def ingest_documents(
    documents: List[dict],
    *,
    config: Optional[DocumentConfig] = None,
    embed_fn: Callable[[str], List[float]],
    store_fn: Callable[[dict], None],
    batch_size: int = 50,
) -> IngestResult:
    """Ingest a list of raw document dicts.

    For each document:

    1. Validate that at least one text field has content.
    2. Compute hierarchy metadata (``depth``, ``path``).
    3. Generate an embedding for the parent document.
    4. Generate embeddings for each child (clause / section).
    5. Call *store_fn* with the enriched document.

    Args:
        documents: Raw dicts from the user's data source.
        config: Document field mapping.  Defaults to a standard config.
        embed_fn: ``(text) -> [float, ...]``.  Called for every piece of
            text that needs an embedding.
        store_fn: ``(enriched_doc) -> None``.  Called once per document
            after embedding.
        batch_size: Ignored in this implementation (reserved for future
            batched embedding support).

    Returns:
        An :class:`IngestResult` summarising the run.
    """
    cfg = config or DocumentConfig()
    result = IngestResult()

    for idx, raw_doc in enumerate(documents):
        try:
            enriched = _process_one(raw_doc, cfg, embed_fn)
            if enriched is None:
                result.skipped += 1
                continue
            store_fn(enriched)
            result.processed += 1
        except Exception as exc:
            result.errors += 1
            result.error_details.append((idx, str(exc)))
            logger.warning("Ingest error at index %d: %s", idx, exc)

    logger.info(
        "Ingest complete: %d processed, %d skipped, %d errors",
        result.processed,
        result.skipped,
        result.errors,
    )
    return result


def _process_one(
    doc: dict,
    cfg: DocumentConfig,
    embed_fn: Callable[[str], List[float]],
) -> Optional[dict]:
    """Validate, normalise, embed, and return an enriched copy."""
    # Extract primary text
    text = _extract_text(doc, cfg.text_fields)
    title = str(doc.get(cfg.title_field, "") or "")

    if not text and not title:
        logger.debug("Skipping document with no text or title")
        return None

    # Build embed text for parent
    embed_parts = [title]
    for h_field in cfg.hierarchy:
        if h_field == cfg.title_field:
            continue
        val = doc.get(h_field)
        if val:
            embed_parts.append(str(val))
    if text:
        embed_parts.append(text)
    embed_text = " ".join(p for p in embed_parts if p)

    # Make a shallow copy to avoid mutating the original
    enriched: Dict[str, Any] = dict(doc)

    # Compute hierarchy metadata
    enriched["depth"] = sum(1 for f in cfg.hierarchy if doc.get(f))
    path_parts = [str(doc.get(f)) for f in cfg.hierarchy if doc.get(f)]
    enriched["path"] = " / ".join(path_parts)

    # Normalise: if there's flat text but no children, wrap as a single child
    _ensure_children(enriched, cfg)

    # Embed parent
    if embed_text.strip():
        enriched["embedding"] = embed_fn(embed_text)

    # Embed children
    for children_field in cfg.children_fields:
        children = enriched.get(children_field)
        if not children or not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, dict):
                continue
            child_text = _child_embed_text(child, cfg)
            if child_text.strip():
                child["embedding"] = embed_fn(child_text)

    return enriched


def _extract_text(doc: dict, text_fields: List[str]) -> str:
    for f in text_fields:
        val = doc.get(f)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _child_embed_text(child: dict, cfg: DocumentConfig) -> str:
    parts: List[str] = []
    title = str(child.get(cfg.title_field, "") or "")
    if title:
        parts.append(title)
    text = _extract_text(child, cfg.text_fields)
    if text:
        parts.append(text)
    return " ".join(parts)


def _ensure_children(doc: dict, cfg: DocumentConfig) -> None:
    """If the doc has flat text but no children, wrap text as a single child."""
    for children_field in cfg.children_fields:
        if doc.get(children_field):
            return  # Already has children

    # Check for flat text
    text = _extract_text(doc, cfg.text_fields)
    if not text:
        return

    # Use the first children_field name
    if cfg.children_fields:
        field_name = cfg.children_fields[0]
        title = str(doc.get(cfg.title_field, "") or "")
        doc[field_name] = [{"title": title, "text": text}]
