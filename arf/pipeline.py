"""Cost-optimised retrieval pipeline with classifier-based LLM routing.

Orchestrates the full query-to-answer flow:

    query → preprocess → cache check → embed → search → triage →
    MLP scoring → LLM verification (uncertain only) → rephrase retry →
    cache store → resolve parents → summarise

The pipeline calls **user-provided functions** for all I/O (search,
embed, LLM, storage).  ARF provides the routing logic — threshold gates,
gap filtering, MLP-based triage, score blending, and rephrase-graph
caching.

No external dependencies beyond the other ``arf`` modules.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from arf.document import Document, DocumentConfig
from arf.features import FeatureExtractor
from arf.query_graph import follow_rephrase_chain
from arf.score_parser import adjust_score as _adjust_score
from arf.triage import Triage

logger = logging.getLogger(__name__)

Scored = Tuple[Any, float]


class Pipeline:
    """End-to-end retrieval pipeline.

    Only *search_fn* and *embed_fn* are required.  All other function
    slots are optional — the pipeline gracefully skips steps when the
    corresponding function is ``None``.

    Args:
        doc_config: Document field mapping.
        triage: Triage instance with threshold settings.
        search_fn: ``(embedding, top_k) -> [(dict, float), ...]``
        embed_fn: ``(text) -> [float, ...]``
        predict_fn: ``(feature_vectors) -> [float, ...]`` (probabilities)
        llm_fn: ``(query, document_dict) -> str`` (raw LLM output with score)
        cache_lookup: ``(query) -> {"results": ..., "next": ...} | None``
        cache_store: ``(query, results) -> None``
        link_fn: ``(source_query, target_query) -> None``
        store_query_fn: ``(query, data_dict) -> None``
        preprocess_fn: ``(query) -> str``
        moderate_fn: ``(query) -> bool`` (True = safe)
        rephrase_fn: ``(query, previous_attempts) -> str | None``
        keyword_fn: ``(query) -> [(dict, float), ...]``
        resolve_fn: ``(parent_id) -> dict | None``
        summarize_fn: ``(query, Document, context_list) -> str``
        graph_max_hops: Max rephrase chain depth.
        parser_range: ``(min_mult, max_mult)`` for score blending.
        predict_zones: ``(low, high)`` probability zones for MLP triage.
        max_rephrase: Number of rephrase-and-retry attempts.
    """

    def __init__(
        self,
        *,
        # Core config
        doc_config: Optional[DocumentConfig] = None,
        triage: Optional[Triage] = None,
        # Required functions
        search_fn: Callable,
        embed_fn: Callable,
        # Optional scoring
        predict_fn: Optional[Callable] = None,
        llm_fn: Optional[Callable] = None,
        # Optional cache
        cache_lookup: Optional[Callable] = None,
        cache_store: Optional[Callable] = None,
        link_fn: Optional[Callable] = None,
        store_query_fn: Optional[Callable] = None,
        # Optional preprocessing
        preprocess_fn: Optional[Callable] = None,
        moderate_fn: Optional[Callable] = None,
        rephrase_fn: Optional[Callable] = None,
        # Optional matching
        keyword_fn: Optional[Callable] = None,
        # Optional hierarchy
        resolve_fn: Optional[Callable] = None,
        # Optional answer
        summarize_fn: Optional[Callable] = None,
        # Tuning
        graph_max_hops: int = 3,
        parser_range: Tuple[float, float] = (0.50, 1.50),
        predict_zones: Tuple[float, float] = (0.4, 0.6),
        max_rephrase: int = 2,
    ):
        self.doc_config = doc_config or DocumentConfig()
        self.triage = triage or Triage()
        self.features = FeatureExtractor(self.doc_config)

        self.search_fn = search_fn
        self.embed_fn = embed_fn
        self.predict_fn = predict_fn
        self.llm_fn = llm_fn
        self.cache_lookup = cache_lookup
        self.cache_store = cache_store
        self.link_fn = link_fn
        self.store_query_fn = store_query_fn
        self.preprocess_fn = preprocess_fn
        self.moderate_fn = moderate_fn
        self.rephrase_fn = rephrase_fn
        self.keyword_fn = keyword_fn
        self.resolve_fn = resolve_fn
        self.summarize_fn = summarize_fn

        self.graph_max_hops = graph_max_hops
        self.parser_min, self.parser_max = parser_range
        self.predict_zones = predict_zones
        self.max_rephrase = max_rephrase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, *, top_k: int = 10) -> List[Dict[str, Any]]:
        """Execute the full retrieval pipeline.

        Returns a list of result dicts, each containing:

        * ``"document"`` — the :class:`~arf.document.Document` object
        * ``"score"`` — final relevance score
        * ``"context"`` — list of ancestor :class:`~arf.document.Document`
          objects (if *resolve_fn* is provided)
        * ``"summary"`` — LLM-generated summary string (if *summarize_fn*
          is provided)
        """
        original_query = query

        # 1. Moderation
        if self.moderate_fn is not None:
            safe = self.moderate_fn(query)
            if not safe:
                logger.info("Query blocked by moderation: %r", query[:80])
                return []

        # 2. Preprocessing
        if self.preprocess_fn is not None:
            query = self.preprocess_fn(query)

        # 3. Cache check (rephrase graph walk)
        if self.cache_lookup is not None:
            chain = follow_rephrase_chain(
                query,
                lookup_fn=self.cache_lookup,
                max_hops=self.graph_max_hops,
            )
            if chain.hit:
                logger.info("Cache hit for %r (hops=%d)", query[:60], chain.hops)
                return self._wrap_results(chain.cached_results, query)
            if chain.final_text != query:
                query = chain.final_text

        # 4. Search loop (with rephrase retry)
        previous_attempts: List[str] = []
        for attempt in range(self.max_rephrase + 1):
            results = self._search_and_rank(query, top_k=top_k)
            if results:
                break

            # Rephrase
            if self.rephrase_fn is None or attempt >= self.max_rephrase:
                break
            rephrased = self.rephrase_fn(query, previous_attempts)
            if not rephrased or rephrased == query:
                break

            # Link rephrase in graph
            if self.link_fn is not None:
                try:
                    self.link_fn(query, rephrased)
                except Exception:
                    logger.debug("link_fn failed for %r -> %r", query[:40], rephrased[:40])

            previous_attempts.append(query)
            query = rephrased

        if not results:
            return []

        # 5. Cache store
        if self.cache_store is not None:
            try:
                self.cache_store(query, results)
            except Exception:
                logger.debug("cache_store failed for %r", query[:60])

        # 6. Build output with optional parent context + summaries
        return self._build_output(results, original_query)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_and_rank(self, query: str, *, top_k: int) -> List[Scored]:
        """Embed → search → keyword merge → triage → MLP → LLM verify."""
        # Embed
        embedding = self.embed_fn(query)

        # Store query embedding
        if self.store_query_fn is not None:
            try:
                self.store_query_fn(query, {"embedding": embedding})
            except Exception:
                logger.debug("store_query_fn failed for %r", query[:60])

        # Search
        candidates: List[Scored] = self.search_fn(embedding, top_k) or []
        if not candidates:
            return []

        # Keyword merge
        if self.keyword_fn is not None:
            kw_results = self.keyword_fn(query) or []
            if kw_results:
                candidates = self.triage.dedupe(candidates, kw_results)

        # Triage — split into accept / needs_review / reject
        result = self.triage.classify(candidates)

        # MLP scoring on needs_review candidates
        mlp_accepted: List[Scored] = []
        mlp_uncertain: List[Scored] = result.needs_review

        if self.predict_fn is not None and result.needs_review:
            mlp_accepted, mlp_uncertain = self._mlp_triage(
                query, result.needs_review, embedding
            )

        # LLM verification on uncertain candidates
        llm_verified: List[Scored] = []
        if self.llm_fn is not None and mlp_uncertain:
            llm_verified = self._llm_verify(query, mlp_uncertain)

        # Merge all accepted
        merged = self.triage.dedupe(result.accepted, mlp_accepted, llm_verified)

        # Gap filter
        if merged:
            merged = self.triage.gap_filter(merged)

        return merged[: top_k]

    def _mlp_triage(
        self, query: str, candidates: List[Scored], query_embedding: Any
    ) -> Tuple[List[Scored], List[Scored]]:
        """Run MLP on candidates, return (accepted, uncertain)."""
        try:
            features = self.features.extract_batch(
                query=query,
                results=candidates,
                query_embedding=query_embedding,
            )
            vectors = [self.features.to_vector(f) for f in features]
            probs = self.predict_fn(vectors)

            routed = self.triage.by_zones(candidates, probs, zones=self.predict_zones)
            return routed.accepted, routed.needs_review
        except Exception as exc:
            logger.warning("MLP triage failed: %s — sending all to LLM", exc)
            return [], candidates

    def _llm_verify(self, query: str, candidates: List[Scored]) -> List[Scored]:
        """Run LLM verification, parse scores, blend."""
        verified: List[Scored] = []
        for item, base_score in candidates:
            try:
                raw_output = self.llm_fn(query, item)
                adj = _adjust_score(
                    base_score,
                    raw_output,
                    min_mult=self.parser_min,
                    max_mult=self.parser_max,
                )
                if adj >= self.triage.accept_threshold:
                    verified.append((item, adj))
            except Exception:
                logger.debug("LLM verify failed for item; keeping base score %.3f", base_score)
                if base_score >= self.triage.accept_threshold:
                    verified.append((item, base_score))
        return verified

    def _build_output(
        self, results: List[Scored], query: str
    ) -> List[Dict[str, Any]]:
        """Convert scored tuples to rich output dicts."""
        output: List[Dict[str, Any]] = []
        for item, score in results:
            doc = Document.from_dict(item, self.doc_config) if isinstance(item, dict) else item

            entry: Dict[str, Any] = {"document": doc, "score": score}

            # Resolve parent chain
            if self.resolve_fn is not None and isinstance(doc, Document) and doc.parent_id:
                entry["context"] = self._resolve_ancestors(doc)
            else:
                entry["context"] = []

            # Generate summary
            if self.summarize_fn is not None and isinstance(doc, Document):
                try:
                    entry["summary"] = self.summarize_fn(query, doc, entry["context"])
                except Exception:
                    entry["summary"] = ""
            else:
                entry["summary"] = ""

            output.append(entry)
        return output

    def _resolve_ancestors(self, doc: Document, max_depth: int = 5) -> List[Document]:
        """Walk up the parent chain via resolve_fn."""
        ancestors: List[Document] = []
        current_pid = doc.parent_id
        seen: set = set()

        for _ in range(max_depth):
            if not current_pid or current_pid in seen:
                break
            seen.add(current_pid)
            raw = self.resolve_fn(current_pid)
            if raw is None:
                break
            parent = Document.from_dict(raw, self.doc_config) if isinstance(raw, dict) else raw
            ancestors.append(parent)
            current_pid = getattr(parent, "parent_id", None)

        ancestors.reverse()
        return ancestors

    def _wrap_results(self, cached: list, query: str) -> List[Dict[str, Any]]:
        """Wrap raw cached results as output dicts."""
        results: List[Scored] = []
        for item in cached:
            if isinstance(item, tuple) and len(item) == 2:
                results.append(item)
            elif isinstance(item, dict):
                score = float(item.get("score", 1.0))
                results.append((item, score))
            else:
                results.append((item, 1.0))
        return self._build_output(results, query)
