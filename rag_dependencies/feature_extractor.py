# rag_dependencies/feature_extractor.py
"""
Feature engineering module for the MLP reranker.

Extracts numeric features from query-document pairs for use in a learned
reranking model. All features are numeric (float/int) and the extractor
works without a live MongoDB connection — it operates on pre-computed data
passed in from the retrieval pipeline.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Try numpy for fast cosine similarity; fall back to pure Python
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN_ENCODING: Dict[str, int] = {
    "us_constitution": 0,
    "code_of_federal_regulations": 1,
    "us_code": 2,
    "uscis_policy": 3,
}

# Ordered feature names — must stay in sync with extract_features / to_vector
_FEATURE_NAMES: List[str] = [
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

# Hierarchy fields in order of depth (used for section_depth calculation)
_HIERARCHY_FIELDS = ["title", "article", "chapter", "subchapter", "part", "section"]

# Simple tokeniser: word-boundary split, lowercased, non-empty
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word-boundary tokenisation."""
    return _TOKEN_RE.findall((text or "").lower())


def _cosine_similarity_manual(vec_a: list, vec_b: list) -> float:
    """Pure-Python cosine similarity (fallback when numpy unavailable)."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-9
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-9
    sim = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))


def _cosine_similarity(vec_a, vec_b) -> float:
    """Cosine similarity using numpy if available, else manual fallback."""
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
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _compute_bm25_score(query: str, doc_text: str, k1: float = 1.5, b: float = 0.75, avgdl: float = 2000.0) -> float:
    """Simple single-document BM25 approximation.

    Since we don't have corpus-level IDF statistics, we use a term-frequency
    based approximation that still captures the diminishing-returns TF
    saturation and document length normalisation of BM25.  IDF is set to 1.0
    for every query term (equivalent to assuming each term appears in roughly
    half the corpus).
    """
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
        # BM25 TF component (IDF fixed at 1.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dl / avgdl))
        score += numerator / denominator

    return score


def _get_document_text(document: dict, field_mapping: Optional[dict] = None) -> str:
    """Extract the primary text content from a document."""
    text_fields = ["text", "summary", "content", "body"]
    if field_mapping and "text" in field_mapping:
        mapping = field_mapping["text"]
        if isinstance(mapping, list):
            text_fields = mapping
        elif isinstance(mapping, str):
            text_fields = [mapping]

    for field in text_fields:
        val = document.get(field)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _get_nested_text(document: dict, field_mapping: Optional[dict] = None) -> str:
    """Extract text from nested arrays (clauses, sections)."""
    nested_fields = ["clauses", "sections"]
    if field_mapping and "nested_text" in field_mapping:
        mapping = field_mapping["nested_text"]
        if isinstance(mapping, list):
            nested_fields = mapping

    parts: List[str] = []
    for field in nested_fields:
        items = document.get(field)
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
    """Check whether the document has non-empty nested content arrays."""
    nested_fields = ["clauses", "sections"]
    if field_mapping and "nested_text" in field_mapping:
        mapping = field_mapping["nested_text"]
        if isinstance(mapping, list):
            nested_fields = mapping

    for field in nested_fields:
        items = document.get(field)
        if items and isinstance(items, list) and len(items) > 0:
            return True
    return False


def _section_depth(document: dict, field_mapping: Optional[dict] = None) -> int:
    """Count how many hierarchy levels are populated in the document."""
    depth = 0
    for field in _HIERARCHY_FIELDS:
        # Respect field_mapping — if the config says a field is None, skip it
        if field_mapping:
            mapped = field_mapping.get(field)
            if mapped is None:
                continue
            actual_field = mapped if isinstance(mapped, str) else field
        else:
            actual_field = field

        val = document.get(actual_field)
        if val is not None and val != "":
            depth += 1
    return depth


def _match_type(query: str, document: dict, keyword_matches: Optional[list], alias_matches: Optional[list]) -> int:
    """Determine match type: 0=none, 1=partial, 2=exact.

    Exact means the document title appears verbatim in a keyword or alias
    match list. Partial means the query overlaps with the title tokens.
    """
    doc_title = (document.get("title") or "").strip().lower()
    if not doc_title:
        return 0

    # Check keyword matches
    if keyword_matches:
        for match in keyword_matches:
            match_lower = (match if isinstance(match, str) else str(match)).lower()
            if match_lower == doc_title:
                return 2
            if match_lower in doc_title or doc_title in match_lower:
                return 1

    # Check alias matches — items may be (alias, title, score) tuples or strings
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

    # Fallback: token overlap between query and title
    query_tokens = set(_tokenize(query))
    title_tokens = set(_tokenize(doc_title))
    if query_tokens and title_tokens and query_tokens & title_tokens:
        return 1

    return 0


def _bias_for_document(document: dict, bias_map: Optional[dict]) -> float:
    """Look up the bias adjustment for this document from the config bias map."""
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
    """Extracts numeric features from a (query, document) pair for the MLP reranker."""

    def __init__(self, config: dict, domain: str):
        """Initialise with a collection config dict and a domain identifier.

        Args:
            config: One entry from ``COLLECTION`` (e.g. ``COLLECTION["US_CONSTITUTION_SET"]``).
            domain: Domain string such as ``"us_constitution"`` or ``"us_code"``.
        """
        self.config = config or {}
        self.domain = domain
        self.domain_id = DOMAIN_ENCODING.get(domain, -1)
        self.bias_map: dict = self.config.get("bias") or {}
        self.field_mapping: Optional[dict] = self.config.get("field_mapping")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self,
        query: str,
        document: dict,
        semantic_score: float,
        query_embedding: list = None,
        doc_embedding: list = None,
        top_score: float = None,
        keyword_matches: list = None,
        alias_matches: list = None,
    ) -> dict:
        """Extract all features for a single query-document pair.

        Returns:
            Dict mapping feature name to numeric value.
        """
        query = query or ""
        document = document or {}
        semantic_score = float(semantic_score or 0.0)

        # 1. Semantic score (raw cosine from vector search)
        feat_semantic = semantic_score

        # 2. BM25 score
        doc_text = _get_document_text(document, self.field_mapping)
        nested_text = _get_nested_text(document, self.field_mapping)
        full_text = f"{doc_text} {nested_text}".strip() if nested_text else doc_text
        feat_bm25 = _compute_bm25_score(query, full_text)

        # 3. Alias match flag
        feat_alias = self._alias_match_flag(document, alias_matches)

        # 4. Keyword match flag
        feat_keyword = self._keyword_match_flag(document, keyword_matches)

        # 5. Domain type
        feat_domain = self.domain_id

        # 6. Document length (log-scaled character count)
        raw_len = len(full_text)
        feat_doc_len = math.log1p(raw_len)  # log(1 + len) for normalisation

        # 7. Query length (character count)
        feat_query_len = len(query)

        # 8. Section depth
        feat_depth = _section_depth(document, self.field_mapping)

        # 9. Embedding cosine similarity
        feat_emb_cos = _cosine_similarity(query_embedding, doc_embedding)

        # 10. Match type (0=none, 1=partial, 2=exact)
        feat_match_type = _match_type(query, document, keyword_matches, alias_matches)

        # 11. Score gap from top
        if top_score is not None:
            feat_score_gap = float(top_score) - semantic_score
        else:
            feat_score_gap = 0.0

        # 12. Query term coverage
        query_tokens = set(_tokenize(query))
        if query_tokens and full_text:
            doc_token_set = set(_tokenize(full_text))
            feat_coverage = len(query_tokens & doc_token_set) / len(query_tokens)
        else:
            feat_coverage = 0.0

        # 13. Title similarity (Jaccard)
        doc_title = document.get("title") or ""
        feat_title_sim = _jaccard_similarity(query, doc_title)

        # 14. Has nested content
        feat_nested = int(_has_nested(document, self.field_mapping))

        # 15. Bias adjustment
        feat_bias = _bias_for_document(document, self.bias_map)

        return {
            "semantic_score": feat_semantic,
            "bm25_score": feat_bm25,
            "alias_match": feat_alias,
            "keyword_match": feat_keyword,
            "domain_type": feat_domain,
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
        results: list,
        query_embedding: list = None,
        keyword_matches: list = None,
        alias_matches: list = None,
    ) -> list:
        """Extract features for every result of a single query.

        Args:
            query: The user query string.
            results: List of ``(document, score)`` tuples from retrieval.
            query_embedding: Optional pre-computed query embedding.
            keyword_matches: Optional list of keyword match strings.
            alias_matches: Optional list of alias match tuples/strings.

        Returns:
            List of feature dicts (same order as *results*).
        """
        if not results:
            return []

        # Determine the top score across all results
        top_score = max((float(score) for _doc, score in results), default=0.0)

        feature_list: List[dict] = []
        for doc, score in results:
            # Try to get the document embedding from the doc dict itself
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
    def feature_names() -> list:
        """Return the ordered list of feature names (for consistent vector ordering)."""
        return list(_FEATURE_NAMES)

    def to_vector(self, features: dict) -> list:
        """Convert a feature dict to an ordered numeric list for ML model input.

        Missing keys default to ``0.0``.
        """
        return [float(features.get(name, 0.0)) for name in _FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_match_flag(document: dict, alias_matches: Optional[list]) -> int:
        """Return 1 if the document title appears in alias matches, else 0."""
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
        """Return 1 if the document title appears in keyword matches, else 0."""
        if not keyword_matches:
            return 0
        doc_title = (document.get("title") or "").lower()
        if not doc_title:
            return 0
        for match in keyword_matches:
            if isinstance(match, str) and match.lower() == doc_title:
                return 1
        return 0
