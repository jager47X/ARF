"""DB-agnostic document model with hierarchy support.

Provides a universal Document representation that works with any database
(MongoDB, PostgreSQL, DynamoDB, Pinecone, etc.) by normalizing field names
and computing hierarchy metadata (depth, path) from raw data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default hierarchy fields in order from root to leaf
_DEFAULT_HIERARCHY = ["title", "article", "chapter", "subchapter", "part", "section"]


@dataclass
class DocumentConfig:
    """Maps your DB schema to ARF's document model.

    Args:
        id_field: Field name for the document ID.
        title_field: Field name for the document title.
        parent_field: Field name for the parent document reference.
        text_fields: Ordered list of field names to search for text content.
        children_fields: Field names for nested child arrays (e.g. clauses, sections).
        hierarchy: Ordered list of field names representing the document tree
            from root to leaf.  Used to compute ``depth`` and ``path``.
        domain_id: Integer encoding for this domain (used as a feature).
        bias_map: Optional mapping of title -> score adjustment.
    """

    id_field: str = "_id"
    title_field: str = "title"
    parent_field: str = "parent_id"
    text_fields: List[str] = field(default_factory=lambda: ["text", "summary", "content", "body"])
    children_fields: List[str] = field(default_factory=lambda: ["clauses", "sections"])
    hierarchy: List[str] = field(default_factory=lambda: list(_DEFAULT_HIERARCHY))
    domain_id: int = 0
    bias_map: Dict[str, float] = field(default_factory=dict)

    @property
    def field_mapping(self) -> Dict[str, Any]:
        """Return a field_mapping dict compatible with the feature extractor."""
        mapping: Dict[str, Any] = {
            "title": self.title_field,
            "text": self.text_fields,
            "nested_text": self.children_fields,
        }
        for h in self.hierarchy:
            if h != "title":
                mapping[h] = h
        return mapping


@dataclass
class Document:
    """A DB-agnostic document with hierarchy metadata.

    Can be constructed directly or via :meth:`from_dict` with a
    :class:`DocumentConfig`.
    """

    id: str = ""
    title: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children: Optional[List["Document"]] = None
    depth: int = 0
    path: str = ""
    embedding: Optional[List[float]] = None

    @property
    def full_text(self) -> str:
        """Content of this document plus all children."""
        parts = [self.content]
        if self.children:
            for child in self.children:
                if child.content:
                    parts.append(child.content)
        return " ".join(p for p in parts if p)

    @property
    def has_children(self) -> bool:
        return bool(self.children and len(self.children) > 0)

    def ancestors(self) -> List[str]:
        """Parse path into ancestor segments."""
        if not self.path:
            return []
        return [p.strip() for p in self.path.split("/") if p.strip()]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, config: Optional[DocumentConfig] = None) -> "Document":
        """Build a Document from a raw dict using field mapping from *config*."""
        if config is None:
            config = DocumentConfig()

        doc_id = str(data.get(config.id_field, data.get("_id", data.get("id", ""))))
        title = str(data.get(config.title_field, "") or "")
        content = _extract_text(data, config.text_fields)
        parent_id_raw = data.get(config.parent_field)
        parent_id = str(parent_id_raw) if parent_id_raw is not None else None
        children = _extract_children(data, config)
        depth_val = _compute_depth(data, config.hierarchy)
        path_val = _compute_path(data, config.hierarchy)
        embedding = data.get("embedding")

        metadata = {
            k: v
            for k, v in data.items()
            if k not in (config.id_field, "embedding")
        }

        return cls(
            id=doc_id,
            title=title,
            content=content,
            metadata=metadata,
            parent_id=parent_id,
            children=children,
            depth=depth_val,
            path=path_val,
            embedding=embedding,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _extract_text(data: dict, text_fields: List[str]) -> str:
    for f in text_fields:
        val = data.get(f)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_children(data: dict, config: DocumentConfig) -> Optional[List[Document]]:
    for f in config.children_fields:
        items = data.get(f)
        if not items or not isinstance(items, list):
            continue
        children: List[Document] = []
        for item in items:
            if isinstance(item, dict):
                text = _extract_text(item, config.text_fields)
                child_title = str(item.get(config.title_field, "") or "")
                children.append(Document(
                    id=str(item.get(config.id_field, item.get("_id", ""))),
                    title=child_title,
                    content=text,
                    embedding=item.get("embedding"),
                ))
            elif isinstance(item, str):
                children.append(Document(content=item))
        if children:
            return children
    return None


def _compute_depth(data: dict, hierarchy: List[str]) -> int:
    return sum(1 for f in hierarchy if data.get(f))


def _compute_path(data: dict, hierarchy: List[str]) -> str:
    parts = []
    for f in hierarchy:
        val = data.get(f)
        if val:
            parts.append(str(val))
    return " / ".join(parts)
