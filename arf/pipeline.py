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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from arf.document import Document, DocumentConfig
from arf.features import FeatureExtractor, cosine_similarity
from arf.query_graph import follow_rephrase_chain
from arf.score_parser import extract_score, multiplier
from arf.triage import Triage

logger = logging.getLogger(__name__)

Scored = Tuple[Any, float]


@dataclass
class _RunContext:
    """Per-``run()`` scratch state threaded through the stages."""

    # dedupe key -> 0-9 score the LLM gave that candidate
    llm_scores: Dict[Any, int] = field(default_factory=dict)


def _make_key_fn(config: DocumentConfig) -> Callable[[Any], Any]:
    """Build a dedupe key function that honours ``config.id_field``.

    Falls back the same way :meth:`Document.from_dict` does (``_id``, ``id``),
    then to the configured title field, then to object identity.
    """

    def key_fn(item: Any) -> Any:
        if isinstance(item, dict):
            for fld in (config.id_field, "_id", "id"):
                val = item.get(fld)
                if val is not None and val != "":
                    return str(val)
            title = item.get(config.title_field)
            if title:
                return ("title", str(title))
            return id(item)
        item_id = getattr(item, "id", None)
        return str(item_id) if item_id else id(item)

    return key_fn


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
        llm_fn: ``(query, document_dict) -> str`` — raw LLM output containing
            a 0-9 relevance score (see :mod:`arf.score_parser` for accepted
            formats).  How the score is used depends on *llm_mode*.
        cache_lookup: ``(query) -> {"results": ..., "next": ...} | None``
        cache_store: ``(query, results) -> None``
        link_fn: ``(source_query, target_query) -> None``
        store_query_fn: ``(query, data_dict) -> None``
        preprocess_fn: ``(query) -> str``
        moderate_fn: ``(query) -> bool`` (True = safe)
        rephrase_fn: ``(query, previous_attempts) -> str | None``
        keyword_fn: ``(query) -> [(dict, float), ...]`` — lexical / keyword
            hits.  Their raw scores are only used in ``keyword_mode="max"``;
            the hit titles also feed the MLP's ``keyword_match`` feature.
        alias_fn: ``(query) -> [(alias, title), ...]`` — alias / synonym
            hits, fed to the MLP's ``alias_match`` and ``match_type``
            features.  Plain title strings are accepted too.
        resolve_fn: ``(parent_id) -> dict | None``
        summarize_fn: ``(query, Document, context_list) -> str``
        graph_max_hops: Max rephrase chain depth.
        parser_range: ``(min_mult, max_mult)`` for score blending
            (``llm_mode="blend"`` only).
        predict_zones: ``(low, high)`` probability zones for MLP triage.
        max_rephrase: Number of rephrase-and-retry attempts.
        unreviewed_policy: What to do with candidates in the "needs review"
            band when no reviewer is left to resolve them (no *predict_fn*
            / *llm_fn*, or the MLP was uncertain and there is no *llm_fn*).
            ``"keep"`` (default) returns them at their retrieval score;
            ``"drop"`` discards them (pre-0.3.0 behaviour).
        keyword_mode: How *keyword_fn* hits are merged with vector candidates.
            ``"rescore"`` (default) puts them on the semantic scale: a hit
            that is also a vector candidate keeps its cosine score; a
            keyword-only hit is scored by cosine against its stored embedding
            (``doc_config.embedding_field``) or, when it has none, assigned
            *keyword_floor*.  Every hit then receives *keyword_boost*.
            ``"max"`` merges the raw keyword scores as-is (pre-0.3.0
            behaviour — only sensible if they are already cosine-like).
        keyword_floor: Score given to keyword-only hits that have no
            embedding.  Defaults to ``triage.verify_threshold`` so they are
            sent for review rather than accepted or rejected outright.
        keyword_boost: Additive bonus applied to every keyword hit in
            ``"rescore"`` mode (result capped at 1.0).
        llm_mode: How the LLM's 0-9 score is used for uncertain candidates.
            ``"gate"`` (default): pass/fail — a candidate scoring at least
            *llm_pass_score* is accepted and keeps its **retrieval** score,
            so ranking and the gap filter stay on one scale; the LLM score
            is reported as ``"llm_score"`` in the results.
            ``"blend"``: pre-0.3.0 behaviour — the retrieval score is
            multiplied by a factor in *parser_range* and the candidate is
            accepted if the blended score reaches ``accept_threshold``.
        llm_pass_score: Minimum 0-9 LLM score to pass in ``"gate"`` mode.
            The default 6 corresponds to "clearly related" on the usual
            calibration prompts.
    """

    _UNREVIEWED_POLICIES = ("keep", "drop")
    _KEYWORD_MODES = ("rescore", "max")
    _LLM_MODES = ("gate", "blend")

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
        alias_fn: Optional[Callable] = None,
        # Optional hierarchy
        resolve_fn: Optional[Callable] = None,
        # Optional answer
        summarize_fn: Optional[Callable] = None,
        # Tuning
        graph_max_hops: int = 3,
        parser_range: Tuple[float, float] = (0.50, 1.50),
        predict_zones: Tuple[float, float] = (0.4, 0.6),
        max_rephrase: int = 2,
        unreviewed_policy: str = "keep",
        keyword_mode: str = "rescore",
        keyword_floor: Optional[float] = None,
        keyword_boost: float = 0.05,
        llm_mode: str = "gate",
        llm_pass_score: int = 6,
    ):
        if unreviewed_policy not in self._UNREVIEWED_POLICIES:
            raise ValueError(f"unreviewed_policy must be one of {self._UNREVIEWED_POLICIES}, got {unreviewed_policy!r}")
        if keyword_mode not in self._KEYWORD_MODES:
            raise ValueError(f"keyword_mode must be one of {self._KEYWORD_MODES}, got {keyword_mode!r}")
        if llm_mode not in self._LLM_MODES:
            raise ValueError(f"llm_mode must be one of {self._LLM_MODES}, got {llm_mode!r}")
        if not 0 <= llm_pass_score <= 9:
            raise ValueError(f"llm_pass_score must be within 0-9, got {llm_pass_score!r}")

        self.doc_config = doc_config or DocumentConfig()
        self.triage = triage or Triage()
        self.features = FeatureExtractor(self.doc_config)
        self._key_fn = _make_key_fn(self.doc_config)

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
        self.alias_fn = alias_fn
        self.resolve_fn = resolve_fn
        self.summarize_fn = summarize_fn

        self.graph_max_hops = graph_max_hops
        self.parser_min, self.parser_max = parser_range
        self.predict_zones = predict_zones
        self.max_rephrase = max_rephrase
        self.unreviewed_policy = unreviewed_policy
        self.keyword_mode = keyword_mode
        self.keyword_floor = keyword_floor
        self.keyword_boost = keyword_boost
        self.llm_mode = llm_mode
        self.llm_pass_score = llm_pass_score

        # Emitted at most once per instance so a mis-calibrated threshold
        # setup is visible without flooding the logs on every query.
        self._warned_all_rejected = False

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
        * ``"llm_score"`` — the 0-9 score the LLM gave this candidate, or
          ``None`` if it never reached LLM verification
        """
        original_query = query
        ctx = _RunContext()

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
            results = self._search_and_rank(query, top_k=top_k, ctx=ctx)
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
        return self._build_output(results, original_query, ctx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_and_rank(self, query: str, *, top_k: int, ctx: Optional[_RunContext] = None) -> List[Scored]:
        """Embed → search → keyword merge → triage → MLP → LLM verify."""
        ctx = ctx if ctx is not None else _RunContext()
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
        keyword_matches: List[str] = []
        if self.keyword_fn is not None:
            kw_results = self.keyword_fn(query) or []
            if kw_results:
                candidates = self._merge_keyword_hits(candidates, kw_results, embedding)
                keyword_matches = [t for t in (self._title_of(item) for item, _ in kw_results) if t]

        # Alias matches (features only)
        alias_matches: List[Any] = []
        if self.alias_fn is not None:
            try:
                alias_matches = list(self.alias_fn(query) or [])
            except Exception as exc:
                logger.warning("alias_fn failed for %r: %s", query[:60], exc)

        # Triage — split into accept / needs_review / reject
        result = self.triage.classify(candidates)
        if not result.accepted and not result.needs_review:
            self._warn_all_rejected(candidates)

        # MLP scoring on needs_review candidates
        mlp_accepted: List[Scored] = []
        mlp_uncertain: List[Scored] = result.needs_review

        if self.predict_fn is not None and result.needs_review:
            mlp_accepted, mlp_uncertain = self._mlp_triage(
                query,
                result.needs_review,
                embedding,
                top_score=max(float(score) for _item, score in candidates),
                keyword_matches=keyword_matches,
                alias_matches=alias_matches,
            )

        # LLM verification on uncertain candidates
        llm_verified: List[Scored] = []
        unreviewed: List[Scored] = []
        if self.llm_fn is not None and mlp_uncertain:
            llm_verified = self._llm_verify(query, mlp_uncertain, ctx)
        elif mlp_uncertain and self.unreviewed_policy == "keep":
            # Nothing left to review these — keep them at their retrieval
            # score rather than silently discarding the whole grey zone.
            unreviewed = mlp_uncertain
            logger.debug("Keeping %d unreviewed candidate(s) (no reviewer configured)", len(unreviewed))

        # Merge all accepted
        merged = self.triage.dedupe(result.accepted, mlp_accepted, llm_verified, unreviewed, key_fn=self._key_fn)

        # Gap filter — always on retrieval scores, so a blended LLM boost can
        # neither evict the other candidates nor rescue a far-off one.
        if merged:
            retrieval_scores = {self._key_fn(item): float(score) for item, score in candidates}
            merged = self._gap_filter_on_retrieval(merged, retrieval_scores)

        return merged[: top_k]

    def _gap_filter_on_retrieval(self, merged: List[Scored], retrieval_scores: Dict[Any, float]) -> List[Scored]:
        def retrieval(c: Scored) -> float:
            return retrieval_scores.get(self._key_fn(c[0]), float(c[1]))

        top = max(retrieval(c) for c in merged)
        return [c for c in merged if top - retrieval(c) <= self.triage.gap]

    def _merge_keyword_hits(
        self, candidates: List[Scored], kw_results: List[Scored], query_embedding: Any
    ) -> List[Scored]:
        """Merge keyword hits into the vector candidates on one score scale.

        Keyword scores (BM25, text-index ranks, ...) are not comparable with
        cosine similarities, so in ``"rescore"`` mode each hit is given a
        semantic score first (see the *keyword_mode* docs) and only then
        merged.  ``dedupe`` keeps the higher score per document, so a hit
        present in both lists ends up with its (boosted) cosine score.
        """
        if self.keyword_mode == "max":
            return self.triage.dedupe(candidates, kw_results, key_fn=self._key_fn)

        floor = self.keyword_floor if self.keyword_floor is not None else self.triage.verify_threshold
        semantic_by_key = {self._key_fn(item): float(score) for item, score in candidates}

        rescored: List[Scored] = []
        for item, _keyword_score in kw_results:
            key = self._key_fn(item)
            if key in semantic_by_key:
                base = semantic_by_key[key]
            else:
                doc_embedding = self._embedding_of(item)
                base = cosine_similarity(query_embedding, doc_embedding) if doc_embedding is not None else floor
            rescored.append((item, min(1.0, base + self.keyword_boost)))

        return self.triage.dedupe(candidates, rescored, key_fn=self._key_fn)

    def _embedding_of(self, item: Any) -> Optional[Any]:
        if isinstance(item, dict):
            return item.get(self.doc_config.embedding_field)
        return getattr(item, "embedding", None)

    def _title_of(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get(self.doc_config.title_field) or "")
        return str(getattr(item, "title", "") or "")

    def _warn_all_rejected(self, candidates: List[Scored]) -> None:
        """Warn (once) when search returned candidates but triage rejected them all.

        This almost always means the thresholds were tuned for a different
        embedding model — e.g. many models produce cosine similarities in
        the 0.3–0.6 range, well below the 0.65 default ``min_score``.
        """
        if self._warned_all_rejected:
            return
        self._warned_all_rejected = True
        top = max(float(score) for _item, score in candidates)
        logger.warning(
            "Triage rejected all %d candidates (top score %.3f; min_score=%.2f, "
            "verify_threshold=%.2f). If your embedding model yields lower "
            "similarities, lower the Triage thresholds to match its score range.",
            len(candidates),
            top,
            self.triage.min_score,
            self.triage.verify_threshold,
        )

    def _mlp_triage(
        self,
        query: str,
        candidates: List[Scored],
        query_embedding: Any,
        *,
        top_score: Optional[float] = None,
        keyword_matches: Optional[List[str]] = None,
        alias_matches: Optional[List[Any]] = None,
    ) -> Tuple[List[Scored], List[Scored]]:
        """Run MLP on candidates, return (accepted, uncertain).

        The same signals used at training time (keyword / alias hits and the
        gap to the query's *overall* top candidate) are passed through so
        the model sees the feature distribution it was trained on.
        """
        try:
            features = self.features.extract_batch(
                query=query,
                results=candidates,
                query_embedding=query_embedding,
                keyword_matches=keyword_matches,
                alias_matches=alias_matches,
                top_score=top_score,
            )
            vectors = [self.features.to_vector(f) for f in features]
            probs = self.predict_fn(vectors)

            routed = self.triage.by_zones(candidates, probs, zones=self.predict_zones)
            return routed.accepted, routed.needs_review
        except Exception as exc:
            logger.warning("MLP triage failed: %s — sending all to LLM", exc)
            return [], candidates

    def _llm_verify(self, query: str, candidates: List[Scored], ctx: _RunContext) -> List[Scored]:
        """Run LLM verification and apply *llm_mode* (gate or blend)."""
        verified: List[Scored] = []
        for item, base_score in candidates:
            try:
                raw_output = self.llm_fn(query, item)
                llm_score = extract_score(raw_output)
            except Exception:
                logger.debug("LLM verify failed for item; keeping base score %.3f", base_score)
                if base_score >= self.triage.accept_threshold:
                    verified.append((item, base_score))
                continue

            ctx.llm_scores[self._key_fn(item)] = llm_score
            if self.llm_mode == "gate":
                if llm_score >= self.llm_pass_score:
                    verified.append((item, base_score))
            else:
                adj = min(base_score * multiplier(llm_score, min_mult=self.parser_min, max_mult=self.parser_max), 1.0)
                if adj >= self.triage.accept_threshold:
                    verified.append((item, adj))
        return verified

    def _build_output(
        self, results: List[Scored], query: str, ctx: Optional[_RunContext] = None
    ) -> List[Dict[str, Any]]:
        """Convert scored tuples to rich output dicts."""
        llm_scores = ctx.llm_scores if ctx is not None else {}
        output: List[Dict[str, Any]] = []
        for item, score in results:
            doc = Document.from_dict(item, self.doc_config) if isinstance(item, dict) else item

            entry: Dict[str, Any] = {
                "document": doc,
                "score": score,
                "llm_score": llm_scores.get(self._key_fn(item)),
            }

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
