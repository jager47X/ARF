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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional numpy — fall back to pure Python if unavailable
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# Bump whenever a feature is added, removed or changes meaning.  The trainer
# records it in every model bundle and load_reranker() refuses (or warns
# about) models trained on another version.
#   1: original 15 features (pre-0.3.0)
#   2: bm25 with IDF, log-scaled token query_length, cosine falls back to
#      semantic_score, + exact_phrase_match
FEATURE_VERSION = 2

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
    "exact_phrase_match",
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


# Public alias — also used by the pipeline to put keyword hits on the semantic scale.
cosine_similarity = _cosine_similarity


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _compute_bm25_score(
    query: str,
    doc_text: str,
    k1: float = 1.5,
    b: float = 0.75,
    avgdl: float = 2000.0,
    idf_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """BM25 of *query* against *doc_text*.

    Without *idf_fn* every term weighs 1.0, which reduces the score to a
    saturated term-frequency overlap; pass :meth:`CorpusStats.idf` for the
    real thing.
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
        idf = idf_fn(term) if idf_fn is not None else 1.0
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (dl / avgdl))
        score += idf * numerator / denominator
    return score


def _exact_phrase_match(query: str, text: str) -> int:
    """1 if the query's tokens occur contiguously, in order, in *text*."""
    phrase = " ".join(_tokenize(query))
    if not phrase:
        return 0
    haystack = " " + " ".join(_tokenize(text)) + " "
    return int(f" {phrase} " in haystack)


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


def _full_text(document: dict, field_mapping: Optional[dict] = None) -> str:
    doc_text = _get_document_text(document, field_mapping)
    nested_text = _get_nested_text(document, field_mapping)
    return f"{doc_text} {nested_text}".strip() if nested_text else doc_text


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


def _section_depth(document: dict, hierarchy: List[str]) -> int:
    return sum(1 for fld in hierarchy if document.get(fld) not in (None, ""))


def _match_type(
    query: str,
    doc_title: str,
    keyword_matches: Optional[list],
    alias_matches: Optional[list],
) -> int:
    doc_title = (doc_title or "").strip().lower()
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


def _bias_for_title(doc_title: str, bias_map: Optional[dict]) -> float:
    if not bias_map or not doc_title:
        return 0.0
    return float(bias_map.get(doc_title, 0.0))


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------


@dataclass
class CorpusStats:
    """Corpus-level statistics that turn the ``bm25_score`` feature into real BM25.

    Without them every query term weighs 1.0 and the average document length
    is assumed, so the feature is only a term-frequency overlap.  Build the
    stats once at ingest time with :meth:`from_documents`, persist them with
    :meth:`to_dict` / :meth:`from_dict`, and hand the *same* stats to the
    :class:`FeatureExtractor` used for training and to
    ``Pipeline(corpus_stats=...)`` at query time.
    """

    n_docs: int = 0
    doc_freq: Dict[str, int] = field(default_factory=dict)
    avgdl: float = 2000.0

    @classmethod
    def from_documents(cls, documents: Iterable[dict], config: Optional[Any] = None) -> "CorpusStats":
        """Count document frequencies and the average length over *documents*."""
        field_mapping = getattr(config, "field_mapping", None)
        doc_freq: Counter = Counter()
        total_len = 0
        n_docs = 0
        for document in documents:
            tokens = _tokenize(_full_text(document or {}, field_mapping))
            n_docs += 1
            total_len += len(tokens)
            doc_freq.update(set(tokens))
        avgdl = (total_len / n_docs) if n_docs else 2000.0
        return cls(n_docs=n_docs, doc_freq=dict(doc_freq), avgdl=avgdl)

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency, ``log(1 + (N - df + 0.5) / (df + 0.5))``."""
        if self.n_docs <= 0:
            return 1.0
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def to_dict(self) -> Dict[str, Any]:
        return {"n_docs": self.n_docs, "avgdl": self.avgdl, "doc_freq": dict(self.doc_freq)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusStats":
        return cls(
            n_docs=int(data.get("n_docs", 0)),
            doc_freq={str(k): int(v) for k, v in dict(data.get("doc_freq") or {}).items()},
            avgdl=float(data.get("avgdl", 2000.0)),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Extracts 16 numeric features from ``(query, document)`` pairs.

    Args:
        config: A :class:`~arf.document.DocumentConfig` describing the
            document schema.  If ``None`` a default config is used.
        corpus_stats: Optional :class:`CorpusStats` giving ``bm25_score``
            real IDF weights and the corpus' average document length.
    """

    def __init__(self, config: Optional[Any] = None, corpus_stats: Optional[CorpusStats] = None):
        self.corpus_stats = corpus_stats
        # Accept a DocumentConfig (or any object with the same attributes) and
        # fall back to DocumentConfig's defaults.  Every title / hierarchy /
        # embedding lookup below goes through these so a schema with e.g.
        # ``title_field="name"`` produces the same features as the default one.
        self.domain_id: int = getattr(config, "domain_id", 0)
        self.bias_map: dict = getattr(config, "bias_map", None) or {}
        self.field_mapping: Optional[dict] = getattr(config, "field_mapping", None)
        self.title_field: str = getattr(config, "title_field", None) or "title"
        self.embedding_field: str = getattr(config, "embedding_field", None) or "embedding"
        self.hierarchy: List[str] = list(getattr(config, "hierarchy", None) or _HIERARCHY_FIELDS)

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
        """Extract all 16 features for a single query-document pair."""
        query = query or ""
        document = document or {}
        semantic_score = float(semantic_score or 0.0)

        full_text = _full_text(document, self.field_mapping)
        doc_title = str(document.get(self.title_field) or "")

        if self.corpus_stats is not None:
            feat_bm25 = _compute_bm25_score(
                query, full_text, avgdl=self.corpus_stats.avgdl, idf_fn=self.corpus_stats.idf
            )
        else:
            feat_bm25 = _compute_bm25_score(query, full_text)
        feat_alias = self._alias_match_flag(doc_title, alias_matches)
        feat_keyword = self._keyword_match_flag(doc_title, keyword_matches)
        raw_len = len(full_text)
        feat_doc_len = math.log1p(raw_len)
        feat_query_len = math.log1p(len(_tokenize(query)))
        feat_depth = _section_depth(document, self.hierarchy)
        if query_embedding is not None and doc_embedding is not None:
            feat_emb_cos = _cosine_similarity(query_embedding, doc_embedding)
        else:
            # No independent signal available — mirror the retrieval score
            # rather than claiming the vectors are orthogonal.
            feat_emb_cos = semantic_score
        feat_match_type = _match_type(query, doc_title, keyword_matches, alias_matches)
        feat_phrase = _exact_phrase_match(query, f"{doc_title} {full_text}")
        feat_score_gap = (float(top_score) - semantic_score) if top_score is not None else 0.0

        query_tokens = set(_tokenize(query))
        if query_tokens and full_text:
            doc_token_set = set(_tokenize(full_text))
            feat_coverage = len(query_tokens & doc_token_set) / len(query_tokens)
        else:
            feat_coverage = 0.0

        feat_title_sim = _jaccard_similarity(query, doc_title)
        feat_nested = int(_has_nested(document, self.field_mapping))
        feat_bias = _bias_for_title(doc_title, self.bias_map)

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
            "exact_phrase_match": feat_phrase,
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
        top_score: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Extract features for every result of a single query.

        Args:
            top_score: Score of the best candidate *for this query*.  Pass it
                when *results* is a subset (e.g. only the "needs review"
                band) so ``score_gap_from_top`` is measured against the real
                top, not the top of the subset.  Defaults to the highest
                score in *results*.
        """
        if not results:
            return []
        if top_score is None:
            top_score = max((float(score) for _doc, score in results), default=0.0)
        feature_list: List[Dict[str, float]] = []
        for doc, score in results:
            doc_embedding = doc.get(self.embedding_field) if isinstance(doc, dict) else None
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

    @staticmethod
    def feature_version() -> int:
        """Return :data:`FEATURE_VERSION` (recorded in trained model bundles)."""
        return FEATURE_VERSION

    def to_vector(self, features: dict) -> List[float]:
        """Convert a feature dict to an ordered numeric list."""
        return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_match_flag(doc_title: str, alias_matches: Optional[list]) -> int:
        if not alias_matches:
            return 0
        doc_title = (doc_title or "").lower()
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
    def _keyword_match_flag(doc_title: str, keyword_matches: Optional[list]) -> int:
        if not keyword_matches:
            return 0
        doc_title = (doc_title or "").lower()
        if not doc_title:
            return 0
        for match in keyword_matches:
            if isinstance(match, str) and match.lower() == doc_title:
                return 1
        return 0
