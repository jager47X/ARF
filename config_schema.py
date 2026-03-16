"""
Pydantic models for validating ARF configuration at startup.

Usage:
    from config_schema import validate_config, validate_thresholds
    validate_thresholds()       # raises ValidationError on bad thresholds
    validate_config("CFR_SET")  # validates a single collection config
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except ImportError:
    print(
        "ERROR: pydantic is required for config validation. "
        "Install it with: pip install pydantic>=2.0",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Threshold schema
# ---------------------------------------------------------------------------

class DomainThresholds(BaseModel):
    """Thresholds that control the retrieval pipeline for a single domain."""

    query_search: float = Field(ge=0.0, le=1.0, description="Initial semantic search threshold")
    alias_search: float = Field(ge=0.0, le=1.0, description="Alias matching threshold")
    RAG_SEARCH_min: float = Field(ge=0.0, le=1.0, description="Minimum score to continue processing")
    LLM_VERIFication: float = Field(ge=0.0, le=1.0, description="Threshold for LLM reranking")
    RAG_SEARCH: float = Field(ge=0.0, le=1.0, description="High-confidence result threshold")
    confident: float = Field(ge=0.0, le=1.0, description="Threshold for saving summaries")
    FILTER_GAP: float = Field(ge=0.0, le=1.0, description="Maximum score gap between results")
    LLM_SCORE: float = Field(ge=0.0, le=1.0, description="LLM reranking score adjustment")
    HYBRID_SEMANTIC_WEIGHT: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    HYBRID_BM25_WEIGHT: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_threshold_ordering(self) -> "DomainThresholds":
        if self.RAG_SEARCH_min > self.RAG_SEARCH:
            raise ValueError(
                f"RAG_SEARCH_min ({self.RAG_SEARCH_min}) must be <= RAG_SEARCH ({self.RAG_SEARCH})"
            )
        if self.LLM_VERIFication > self.RAG_SEARCH:
            raise ValueError(
                f"LLM_VERIFication ({self.LLM_VERIFication}) must be <= RAG_SEARCH ({self.RAG_SEARCH})"
            )
        return self


# ---------------------------------------------------------------------------
# Field mapping schema
# ---------------------------------------------------------------------------

class FieldMapping(BaseModel):
    title: str = "title"
    article: Optional[str] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    part: Optional[str] = None
    subchapter: Optional[str] = None
    text: List[str] = Field(default_factory=lambda: ["text", "summary", "content", "body"])
    nested_text: Optional[List[str]] = None
    references: Optional[str] = None


# ---------------------------------------------------------------------------
# Collection config schema
# ---------------------------------------------------------------------------

class CollectionConfig(BaseModel):
    db_name: str
    query_collection_name: str
    main_collection_name: str
    document_type: str
    unique_index: str = "title"
    sql_attached: bool = False
    use_keyword_matcher: bool = False
    use_alias_search: bool = False
    disable_hybrid_search: bool = False
    thresholds: DomainThresholds
    field_mapping: FieldMapping

    # Optional fields
    main_vector_index: str = "vector_index"
    query_vector_index: str = "vector_index"
    cases_collection_name: Optional[str] = None
    cases_vector_index: Optional[str] = None
    main_fulltext_index: Optional[str] = None
    patterns: Optional[str] = None
    bias: Optional[Dict[str, float]] = None
    tag: Optional[str] = None
    autoupdate_enabled: Optional[bool] = None
    autoupdate_url: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # allow forward-compat with new keys


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def validate_thresholds() -> Dict[str, DomainThresholds]:
    """Validate all DOMAIN_THRESHOLDS from config.py. Returns parsed models."""
    from config import DOMAIN_THRESHOLDS, CLIENT_CASE_THRESHOLDS

    results: Dict[str, DomainThresholds] = {}
    for domain, thr in DOMAIN_THRESHOLDS.items():
        results[domain] = DomainThresholds(**thr)
    results["client_cases"] = DomainThresholds(**CLIENT_CASE_THRESHOLDS)
    return results


def validate_config(collection_key: str) -> CollectionConfig:
    """Validate a single COLLECTION entry from config.py."""
    from config import COLLECTION

    if collection_key not in COLLECTION:
        raise KeyError(f"Unknown collection key: {collection_key!r}. Available: {list(COLLECTION)}")
    return CollectionConfig(**COLLECTION[collection_key])


def validate_all() -> Dict[str, CollectionConfig]:
    """Validate every COLLECTION entry. Returns parsed models."""
    from config import COLLECTION

    results: Dict[str, CollectionConfig] = {}
    for key in COLLECTION:
        results[key] = validate_config(key)
    return results


if __name__ == "__main__":
    print("Validating DOMAIN_THRESHOLDS...")
    thresholds = validate_thresholds()
    for name, t in thresholds.items():
        print(f"  {name}: OK")

    print("\nValidating COLLECTION configs...")
    configs = validate_all()
    for name, c in configs.items():
        print(f"  {name} ({c.document_type}): OK")

    print("\nAll configurations valid.")
