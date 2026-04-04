"""Feature engineering for query-document relevance scoring.

Extracts 15 numeric features from ``(query, document, score)`` triples for
use in a learned reranking model.  All computation is pure Python with an
optional *numpy* acceleration path.

The features are domain-agnostic — they measure textual overlap, structure
depth, embedding similarity, etc. — and can be used with any document
schema via :class:`~arf.document.DocumentConfig`.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional numpy — fall back to pure Python if unavailable
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# Ordered feature names — must stay in sync with extract_features / to_vector
FEATURE_NAMES: List[str] = [
    "semantic_score",
    "bm25_score",
    "alias_match",
    "keyword_match",
    "domain_type",
    "document_length",
    "query_length",
    "section_depth",
    "embedding_cosine_similarity",
    "match_type",
    "score_gap_from_top",
    "query_term_coverage",
    "title_similarity",
    "has_nested_content",
    "bias_adjustment",
]

# Hierarchy fields in order of depth
_HIERARCHY_FIELDS = ["title", "article", "chapter", "subchapter", "part", "section"]

_TOKEN_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _cosine_similarity_manual(vec_a: list, vec_b: list) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-9
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-9
    sim = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))


def _cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    if _HAS_NUMPY:
        try:
            a = np.asarray(vec_a, dtype=float).ravel()
            b = np.asarray(vec_b, dtype=float).ravel()
            if a.size == 0 or b.size == 0 or a.size != b.size:
                return 0.0
            denom = (float(np.linalg.norm(a)) or 1e-9) * (float(np.linalg.norm(b)) or 1e-9)
            sim = float(np.dot(a, b) / denom)
            return max(-1.0, min(1.0, sim))
        except Exception:
            return 0.0
    else:
        if not isinstance(vec_a, list):
            vec_a = list(vec_a)
        if not isinstance(vec_b, list):
            vec_b = list(vec_b)
        return _cosine_similarity_manual(vec_a, vec_b)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _compute_bm25_score(
    query: str, doc_text: str, k1: float = 1.5, b: float = 0.75, avgdl: float = 2000.0
) -> float:
    query_terms = _tokenize(query)
    if not query_terms:
        return 0.0
    doc_tokens = _tokenize(doc_text)
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0
    tf_counter = Counter(doc_tokens)
    score = 0.0
    for term in query_terms:
        tf = tf_counter.get(term, 0)
        if tf == 0:
            continue
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dl / avgdl))
        score += numerator / denominator
    return score


def _get_document_text(document: dict, field_mapping: Optional[dict] = None) -> str:
    text_fields = ["text", "summary", "content", "body"]
    if field_mapping and "text" in field_mapping:
        mapping = field_mapping["text"]
        if isinstance(mapping, list):
            text_fields = mapping
        elif isinstance(mapping, str):
            text_fields = [mapping]
    for fld in text_fields:
        val = document.get(fld)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _get_nested_text(document: dict, field_mapping: Optional[dict] = None) -> str:
    nested_fields = ["clauses", "sections"]
    if field_mapping and "nested_text" in field_mapping:
        mapping = field_mapping["nested_text"]
        if isinstance(mapping, list):
            nested_fields = mapping
    parts: List[str] = []
    for fld in nested_fields:
        items = document.get(fld)
        if not items or not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                for text_key in ("text", "content", "body", "summary"):
                    val = item.get(text_key)
                    if val and isinstance(val, str):
                        parts.append(val)
                        break
            elif isinstance(item, str):
                parts.append(item)
    return " ".join(parts)


def _has_nested(document: dict, field_mapping: Optional[dict] = None) -> bool:
    nested_fields = ["clauses", "sections"]
    if field_mapping and "nested_text" in field_mapping:
        mapping = field_mapping["nested_text"]
        if isinstance(mapping, list):
            nested_fields = mapping
    for fld in nested_fields:
        items = document.get(fld)
        if items and isinstance(items, list) and len(items) > 0:
            return True
    return False


def _section_depth(document: dict, field_mapping: Optional[dict] = None) -> int:
    depth = 0
    for fld in _HIERARCHY_FIELDS:
        if field_mapping:
            mapped = field_mapping.get(fld)
            if mapped is None:
                continue
            actual_field = mapped if isinstance(mapped, str) else fld
        else:
            actual_field = fld
        val = document.get(actual_field)
        if val is not None and val != "":
            depth += 1
    return depth


def _match_type(
    query: str,
    document: dict,
    keyword_matches: Optional[list],
    alias_matches: Optional[list],
) -> int:
    doc_title = (document.get("title") or "").strip().lower()
    if not doc_title:
        return 0
    if keyword_matches:
        for match in keyword_matches:
            match_lower = (match if isinstance(match, str) else str(match)).lower()
            if match_lower == doc_title:
                return 2
            if match_lower in doc_title or doc_title in match_lower:
                return 1
    if alias_matches:
        for match in alias_matches:
            if isinstance(match, (list, tuple)) and len(match) >= 2:
                match_title = str(match[1]).lower()
            else:
                match_title = str(match).lower()
            if match_title == doc_title:
                return 2
            if match_title in doc_title or doc_title in match_title:
                return 1
    query_tokens = set(_tokenize(query))
    title_tokens = set(_tokenize(doc_title))
    if query_tokens and title_tokens and query_tokens & title_tokens:
        return 1
    return 0


def _bias_for_document(document: dict, bias_map: Optional[dict]) -> float:
    if not bias_map:
        return 0.0
    doc_title = document.get("title")
    if doc_title is None:
        return 0.0
    return float(bias_map.get(doc_title, 0.0))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Extracts 15 numeric features from ``(query, document)`` pairs.

    Args:
        config: A :class:`~arf.document.DocumentConfig` describing the
            document schema.  If ``None`` a default config is used.
    """

    def __init__(self, config: Optional[Any] = None):
        # Accept a DocumentConfig or fall back to sensible defaults
        if config is not None:
            self.domain_id: int = getattr(config, "domain_id", 0)
            self.bias_map: dict = getattr(config, "bias_map", {}) or {}
            self.field_mapping: Optional[dict] = getattr(config, "field_mapping", None)
        else:
            self.domain_id = 0
            self.bias_map = {}
            self.field_mapping = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self,
        query: str,
        document: dict,
        semantic_score: float,
        query_embedding: Optional[list] = None,
        doc_embedding: Optional[list] = None,
        top_score: Optional[float] = None,
        keyword_matches: Optional[list] = None,
        alias_matches: Optional[list] = None,
    ) -> Dict[str, float]:
        """Extract all 15 features for a single query-document pair."""
        query = query or ""
        document = document or {}
        semantic_score = float(semantic_score or 0.0)

        doc_text = _get_document_text(document, self.field_mapping)
        nested_text = _get_nested_text(document, self.field_mapping)
        full_text = f"{doc_text} {nested_text}".strip() if nested_text else doc_text

        feat_bm25 = _compute_bm25_score(query, full_text)
        feat_alias = self._alias_match_flag(document, alias_matches)
        feat_keyword = self._keyword_match_flag(document, keyword_matches)
        raw_len = len(full_text)
        feat_doc_len = math.log1p(raw_len)
        feat_query_len = len(query)
        feat_depth = _section_depth(document, self.field_mapping)
        feat_emb_cos = _cosine_similarity(query_embedding, doc_embedding)
        feat_match_type = _match_type(query, document, keyword_matches, alias_matches)
        feat_score_gap = (float(top_score) - semantic_score) if top_score is not None else 0.0

        query_tokens = set(_tokenize(query))
        if query_tokens and full_text:
            doc_token_set = set(_tokenize(full_text))
            feat_coverage = len(query_tokens & doc_token_set) / len(query_tokens)
        else:
            feat_coverage = 0.0

        doc_title = document.get("title") or ""
        feat_title_sim = _jaccard_similarity(query, doc_title)
        feat_nested = int(_has_nested(document, self.field_mapping))
        feat_bias = _bias_for_document(document, self.bias_map)

        return {
            "semantic_score": semantic_score,
            "bm25_score": feat_bm25,
            "alias_match": feat_alias,
            "keyword_match": feat_keyword,
            "domain_type": self.domain_id,
            "document_length": feat_doc_len,
            "query_length": feat_query_len,
            "section_depth": feat_depth,
            "embedding_cosine_similarity": feat_emb_cos,
            "match_type": feat_match_type,
            "score_gap_from_top": feat_score_gap,
            "query_term_coverage": feat_coverage,
            "title_similarity": feat_title_sim,
            "has_nested_content": feat_nested,
            "bias_adjustment": feat_bias,
        }

    def extract_batch(
        self,
        query: str,
        results: List[Tuple[dict, float]],
        query_embedding: Optional[list] = None,
        keyword_matches: Optional[list] = None,
        alias_matches: Optional[list] = None,
    ) -> List[Dict[str, float]]:
        """Extract features for every result of a single query."""
        if not results:
            return []
        top_score = max((float(score) for _doc, score in results), default=0.0)
        feature_list: List[Dict[str, float]] = []
        for doc, score in results:
            doc_embedding = doc.get("embedding") if isinstance(doc, dict) else None
            features = self.extract_features(
                query=query,
                document=doc,
                semantic_score=float(score),
                query_embedding=query_embedding,
                doc_embedding=doc_embedding,
                top_score=top_score,
                keyword_matches=keyword_matches,
                alias_matches=alias_matches,
            )
            feature_list.append(features)
        return feature_list

    @staticmethod
    def feature_names() -> List[str]:
        """Return the ordered list of feature names."""
        return list(FEATURE_NAMES)

    def to_vector(self, features: dict) -> List[float]:
        """Convert a feature dict to an ordered numeric list."""
        return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_match_flag(document: dict, alias_matches: Optional[list]) -> int:
        if not alias_matches:
            return 0
        doc_title = (document.get("title") or "").lower()
        if not doc_title:
            return 0
        for match in alias_matches:
            if isinstance(match, (list, tuple)) and len(match) >= 2:
                match_title = str(match[1]).lower()
            else:
                match_title = str(match).lower()
            if match_title == doc_title:
                return 1
        return 0

    @staticmethod
    def _keyword_match_flag(document: dict, keyword_matches: Optional[list]) -> int:
        if not keyword_matches:
            return 0
        doc_title = (document.get("title") or "").lower()
        if not doc_title:
            return 0
        for match in keyword_matches:
            if isinstance(match, str) and match.lower() == doc_title:
                return 1
        return 0
