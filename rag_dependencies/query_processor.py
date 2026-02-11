# src/rag/query_processor.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple, Optional, Union, Sequence
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
Doc = Dict[str, Any]
Scored = Tuple[Doc, float]
Results = List[Union[Doc, Scored]]


class QueryProcessor:
    """
    End-to-end pipeline using all managers.

    Key policies:
      - CACHE: If self.db.query has a cached `summaries` map for the normalized query,
        return those (no search, no regeneration).
      - NO `case_reference`: all caching goes through `summaries` only.
      - CASES-first:
          * Accept any case whose semantic score >= self.RAG_SEARCH.
          * LLM-verify only cases with self.LLM_VERIFICATION <= score < self.RAG_SEARCH.
          * Map accepted cases -> MAIN by references; then keep any mapped MAIN docs
            with score >= self.RAG_SEARCH and return immediately (skip MAIN search).
          * If CASES reached self.LLM_VERIFICATION but mapping produced no MAIN docs,
            stop (per policy).
          * If CASES top < self.LLM_VERIFICATION, we still try LLM verify on a few top
            cases; if any pass, map -> MAIN and return.
      - MAIN path:
          * Score each MAIN doc and ACCEPT if score >= self.RAG_SEARCH (no rank/gap filter).
          * If none accepted:
              - If any candidate had score >= self.LLM_VERIFICATION, LLM-verify those;
                accept verified docs with LLM score >= self.RAG_SEARCH.
          * Save summaries only if top accepted doc score >= self.CONFIDENT.
      - SUMMARY RE-USE: If a per-(query, knowledge_id) summary already exists, NEVER
        regenerate (both in generate_summary() and when attaching explanations).
    """
    def __init__(self, rag,debug_mode):
        
        self.rag = rag
        self.sql=self.rag.sql
        self.cfg = self.rag.config
        self.debug_mode = debug_mode
        self.thr = self.cfg.get("thresholds", {})
        if not self.cfg["sql_attached"]:
            logger.info("sql is not attached")
            self.RAG_MIN    = float(self.thr.get('RAG_SEARCH_min'))
            self.RAG_SEARCH = float(self.thr.get('RAG_SEARCH'))
            # FIX: key name; keep your original if config uses that exact string
            self.LLM_VERIF  = float(self.thr.get('LLM_VERIFication', self.thr.get('LLM_VERIF', 0.0)))
            self.ALIAS_THR  = float(self.thr.get('alias_search'))
            self.FILTER_GAP = float(self.thr.get('FILTER_GAP'))
            self.LLM_SCORE  = float(self.thr.get('LLM_SCORE'))
            self.CONFIDENT  = float(self.thr.get('confident'))
            # No hybrid search weights - using ABC gate only (semantic search + ABC gates)
            
            self.REPHRASE_LIMIT = int(self.cfg.get("REPHRASE_LIMIT", 3))
            self.vector_search = self.rag.vector_search
            self.kw     = self.rag.keyword
            self.alias  = self.rag.alias
        else:
            logger.info("sql is not attached")
            self.REPHRASE_LIMIT = 0
            self.RAG_MIN = float(self.thr.get('RAG_SEARCH_min', 0.70))  # Load RAG_SEARCH_min for SQL path
            self.RAG_SEARCH = float(self.thr.get('RAG_SEARCH'))-0.1
            # FIX: key name; keep your original if config uses that exact string
            self.LLM_VERIF  = float(self.thr.get('LLM_VERIFication', self.thr.get('LLM_VERIF', 0.0)))-0.1
            self.LLM_SCORE  = float(self.thr.get('LLM_SCORE'))-0.1
            self.FILTER_GAP = float(self.thr.get('FILTER_GAP', 0.05))  # Load FILTER_GAP for SQL path too
            # Hybrid search weights only for client cases (used in keyword_search_cases, not here)
        self.TOP_K_RET  = int(self.cfg.get("top_k", 20))
        self.DOC_TYPE   = (self.cfg.get("document_type") or "").strip()
        self.openAI = self.rag.query_manager.openAI
        self.db     = self.rag.db
        self.qm     = self.rag.query_manager
        self.llmv   = self.rag.llmv
        self.unique_index = self.cfg.get("unique_index", "title")
        self._original_query = None  # Store original query for return value
        self._language = "en"  # Store language for cache hit tracking
        # Request-level caches to avoid redundant API calls
        self._cached_query_embedding = None  # Cache embedding for current request
        self._request_insight_cache = {}  # Key: (norm_query, knowledge_id, language) -> insight text

    # ----------------- public -----------------
    def process_query(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        language: str = "en",
        skip_pre_checks: bool = False,
        skip_cases_search: bool = False
    ) -> Tuple[List[Tuple[dict, float]], str]:
        """
        Policy:
        - ALWAYS create/get embedding first.
        - If embedding is cached and we have cached results, RETURN THEM and
            SKIP Omni + classifier + any new OpenAI calls.
        - Only if there is no cached result path do we:
            * run Omni moderation
            * run US-Const relevance classifier
            * run fix_query + RAG search
        - BILINGUAL: Always search in English. Translate Spanish queries to English.
        """
        # Store original query for display purposes
        original_query = query
        self._language = language  # Store language for cache hit tracking
        
        # Translate Spanish queries to English for search
        # Always search in English, but preserve original for response
        search_query = query
        if language == "es":
            # Import here to avoid circular dependencies
            from services.rag.rag_dependencies.ai_service import translate_query
            translated = translate_query(query, source_lang="es", target_lang="en")
            if translated and translated != query:
                logger.info(f"[RAG][TRANSLATE] Spanish query translated: {query[:50]}... -> {translated[:50]}...")
                search_query = translated
            else:
                logger.info(f"[RAG][TRANSLATE] Query translation skipped or failed, using original")
        
        # Use translated query for search, but return original in response
        current_text = search_query
        root_original_text = search_query
        seen_norm = {self.db.normalize_query(search_query)}
        previous_rephrases: List[str] = []
        current_decay = 0.0
        rephrase_count = 0  # Track actual number of rephrases performed (enforce REPHRASE_LIMIT)
        
        # Store original query for return value
        self._original_query = original_query
        
        # Reset request-level caches for this new query processing
        self._cached_query_embedding = None
        self._request_insight_cache = {}

        logger.info(
            "[RAG][LIFE][BEGIN] query=%r (search=%r) norm=%r limit=%d",
            original_query,
            search_query,
            self.db.normalize_query(search_query),
            self.REPHRASE_LIMIT,
        )

        for attempt in range(self.REPHRASE_LIMIT + 1):
            logger.info(
                "[RAG][REPHRASE][ATTEMPT] #%d current=%r norm=%r",
                attempt,
                current_text,
                self.db.normalize_query(current_text),
            )

            # ---------- 1) EMBEDDING FIRST ----------
            embed_text = self.db.normalize_query(current_text)
            logger.info("[RAG][EMB][GET] text=%r", current_text)
            emb, cached = self.qm.get_or_create_query_embedding(
                embed_text, self.db, previous_rephrases
            )
            logger.info(
                "[RAG][EMB][RES] cached=%s have_emb=%s",
                cached,
                bool(emb is not None),
            )

            cache_used = False

            # ---------- 2) CACHE FAST-PATH ----------
            if cached:
                logger.info("[RAG][CHAIN][START] seed=%r", current_text)
            state = self._follow_rephrases_or_cached(
                current_text, max_hops=self.REPHRASE_LIMIT, top_k=self.TOP_K_RET
            )

            logger.info(
                "[RAG][CHAIN][STATE] hops=%d loop=%s capped=%s final=%r chain=%r cached_results=%s",
                state["hops"],
                state["loop_detected"],
                state["hit_max_hops"],
                state["final_text"],
                state["chain"],
                bool(state["cached_results"]),
            )

            if state["cached_results"] is not None:
                # IMPORTANT: cached path -> DO NOT RUN Omni / classifier
                cache_used = True
                logger.info(
                    "[RAG][CACHE][HIT] final=%r count=%d -> returning cached (skip Omni + classifier)",
                    state["final_text"],
                    len(state["cached_results"]),
                )
                
                # Track query cache hit
                # For Spanish queries, track on the original Spanish query document, not the translated English one
                try:
                    if self._language == "es" and self._original_query:
                        # Track cache hit on the Spanish query document (what user actually searched)
                        norm_original = self.db.normalize_query(self._original_query)
                        self.db.track_query_cache_hit(norm_original, cache_type="query", language="es")
                        logger.debug(f"[RAG][TRACK] Tracked cache hit on Spanish query: {norm_original[:50]}... (language=es)")
                    else:
                        # Track cache hit on the English query (or original if not Spanish)
                        norm_query = self.db.normalize_query(state["final_text"])
                        self.db.track_query_cache_hit(norm_query, cache_type="query", language="en")
                except Exception as e:
                    logger.debug(f"[RAG][TRACK] Failed to track query cache hit: {e}")

                cached_with_scores: List[Tuple[dict, float]] = []
                for res in state["cached_results"]:
                    enriched_doc = res
                    score = 1.0

                    if isinstance(res, dict):
                        # Use stored score if present
                        if "score" in res:
                            try:
                                score = float(res["score"])
                            except Exception:
                                score = 1.0

                        # Prefer knowledge_id (pointer to MAIN collection)
                        kid = res.get("knowledge_id")
                        has_content = bool(res.get("text") or res.get("summary"))

                        if kid and not has_content:
                            # Fetch from main by ObjectId
                            try:
                                full_doc = self.db.main.find_one(
                                    {"_id": ObjectId(kid)},
                                    {"text_embedding": 0, "summary_embedding": 0},
                                )
                                if full_doc:
                                    enriched_doc = full_doc
                                    logger.info(
                                        "[RAG][CACHE][ENRICH] fetched full doc for kid=%s",
                                        str(kid)[-6:],
                                    )
                            except Exception as e:
                                logger.warning(
                                    "[RAG][CACHE][ENRICH] failed to fetch kid=%s: %s",
                                    kid,
                                    e,
                                )
                        elif not kid and not has_content:
                            # Fallback by title
                            title = res.get("title")
                            if title:
                                try:
                                    key = self.cfg.get("unique_index", "title")
                                    full_doc = self.db.main.find_one(
                                        {key: title},
                                        {"text_embedding": 0, "summary_embedding": 0},
                                    )
                                    if full_doc:
                                        enriched_doc = full_doc
                                        logger.info(
                                            "[RAG][CACHE][ENRICH] fetched full doc for title=%r",
                                            title[:30],
                                        )
                                except Exception as e:
                                    logger.warning(
                                        "[RAG][CACHE][ENRICH] failed to fetch title=%r: %s",
                                        title,
                                        e,
                                    )

                    cached_with_scores.append((enriched_doc, score))

                # IMPORTANT: do not trigger any AI insight generation here.
                # Summaries / insights are already stored and will be reused
                # by get_or_create_insight_by_index / get_summary.
                return (cached_with_scores[: self.TOP_K_RET], self._original_query or current_text)

            # Follow rephrase chain even if no cached results
            if state["final_text"] != current_text:
                logger.info(
                    "[RAG][CHAIN][ADVANCE] %r -> %r",
                    current_text,
                    state["final_text"],
                )
            current_text = state["final_text"]

            if state["chain"]:
                before = len(previous_rephrases)
                previous_rephrases.extend(
                    [t for t in state["chain"] if t not in previous_rephrases]
                )
                logger.info(
                    "[RAG][CHAIN][HISTORY] added=%d total=%d",
                    len(previous_rephrases) - before,
                    len(previous_rephrases),
                )

            # ---------- 3) SAFETY LAYER (Omni) ----------
            # Only on first attempt AND only if we did not short-circuit on cache.
            # Skip if skip_pre_checks is True (for parallel domain workers)
            if attempt == 0 and not cache_used and not skip_pre_checks:
                try:
                    import datetime

                    logger.info(
                        "[RAG][SAFETY][CHECK] Running OpenAI Omni moderation check on query (FIRST, no-cache)"
                    )
                    moderation_result = self.openAI.check_moderation(current_text)

                    if moderation_result["flagged"]:
                        flagged_categories = moderation_result["categories"]
                        score_dict = moderation_result["scores"]

                        logger.warning(
                            "[RAG][SAFETY][REJECT] Query flagged by OpenAI Omni moderation: %s - storing in MongoDB and stopping",
                            flagged_categories,
                        )

                        if score_dict:
                            logger.info(
                                "[RAG][SAFETY][SCORES] Category scores: %s", score_dict
                            )

                        # Store moderation result for the normalized query
                        try:
                            norm_query = self.db.normalize_query(current_text)
                            now = datetime.datetime.utcnow()
                            self.db.query.update_one(
                                {"query": norm_query},
                                {
                                    "$set": {
                                        "query": norm_query,
                                        "rejected_by_moderation": True,
                                        "moderation_flagged_categories": flagged_categories,
                                        "moderation_scores": score_dict,
                                        "rejected_at": now,
                                        "updated_at": now,
                                    },
                                    "$setOnInsert": {"created_at": now},
                                },
                                upsert=True,
                            )
                            logger.info(
                                "[RAG][SAFETY][STORE] Stored rejected query in MongoDB: %r",
                                norm_query,
                            )
                        except Exception as e:
                            logger.exception(
                                "[RAG][SAFETY][STORE][ERROR] Failed to store rejected query: %s",
                                e,
                            )

                        return ([], self._original_query or current_text)
                    else:
                        logger.info("[RAG][SAFETY][PASS] Query passed OpenAI Omni check")
                except Exception as e:
                    logger.exception(
                        "[RAG][SAFETY][ERROR] Failed to run moderation check: %s", e
                    )
                    logger.info(
                        "[RAG][SAFETY][FALLBACK] Continuing with RAG processing despite moderation check failure"
                    )

            # ---------- 4) US CONSTITUTION TOPIC CHECK + fix_query ----------
            # Only on first attempt AND only if no cache-shortcut was used.
            # Skip if skip_pre_checks is True (for parallel domain workers - domain already selected)
            if attempt == 0 and not cache_used and not skip_pre_checks:
                try:
                    logger.info(
                        "[RAG][TOPIC][CHECK] Checking query against US Constitution using LLM"
                    )
                    is_relevant = self.openAI.check_us_constitution_relevance(current_text)

                    if not is_relevant:
                        logger.info(
                            "[RAG][TOPIC][REJECT] Query is not relevant to US Constitution, checking for jurisdiction"
                        )

                        # If no jurisdiction provided, ask for it
                        if not jurisdiction or not jurisdiction.strip():
                            logger.info(
                                "[RAG][TOPIC][JURISDICTION_REQUEST] No jurisdiction provided, requesting jurisdiction"
                            )
                            jurisdiction_message = (
                                "Para brindarle la información más relevante, por favor especifique en qué jurisdicción o ciudad está buscando."
                                if language == "es"
                                else
                                "To provide you with the most relevant information, please specify which jurisdiction or city you are searching for."
                            )
                            jurisdiction_title = "Jurisdicción Requerida" if language == "es" else "Jurisdiction Required"
                            jurisdiction_request_result = {
                                "_general_info": True,
                                "_request_jurisdiction": True,
                                "text": jurisdiction_message,
                                "title": jurisdiction_title,
                            }
                            return ([(jurisdiction_request_result, 0.0)], self._original_query or current_text)

                        # Generate general information response using LLM with jurisdiction context
                        try:
                            general_info = self.openAI.generate_general_info(
                                current_text, jurisdiction=jurisdiction, language=language
                            )

                            logger.info(
                                "[RAG][TOPIC][GENERAL_INFO] Generated general info response with jurisdiction %s: %s",
                                jurisdiction, general_info[:100]
                            )

                            # Return special marker result that indicates general info response
                            # The API will detect this and format it appropriately
                            action_button_text = "Iniciar Caso" if language == "es" else "Start Case"
                            action_button_message = "Podemos evaluar su caso" if language == "es" else "We can evaluate your case"
                            general_info_title = "Información General" if language == "es" else "General Information"

                            general_info_result = {
                                "_general_info": True,
                                "text": general_info.strip(),
                                "title": general_info_title,
                                "action_button": {
                                    "text": action_button_text,
                                    "message": action_button_message
                                }
                            }
                            return ([(general_info_result, 0.0)], current_text)
                        except Exception as e:
                            logger.exception(
                                "[RAG][TOPIC][GENERAL_INFO][ERROR] Failed to generate general info: %s", e
                            )
                            # Fallback: return empty but log the attempt
                            return ([], current_text)
                    else:
                        logger.info(
                            "[RAG][TOPIC][PASS] Query is relevant to US Constitution, proceeding with fix_query and RAG"
                        )
                        pi_removed = self.openAI.remove_personal_info(current_text)
                        fixed = self.openAI.fix_query(pi_removed)
                        logger.info(
                            "[RAG][FIX] before=%r after=%r | norm_before=%r norm_after=%r",
                            current_text,
                            fixed,
                            self.db.normalize_query(current_text),
                            self.db.normalize_query(fixed),
                        )

                        if (
                            self.db.normalize_query(fixed)
                            != self.db.normalize_query(current_text)
                            and not self.rag.debug_mode
                        ):
                            logger.info(
                                "[RAG][LINK][TRY] orig=%r -> rephrased=%r",
                                current_text,
                                fixed,
                            )
                            self.qm.update_query_rephrased_ref(
                                self.db, current_text, fixed
                            )
                            current_text = fixed
                            has_link = self.qm.check_query_has_update_reference(
                                self.db, current_text
                            )
                            nxt = self.qm.get_query_with_rephrase(self.db, current_text)
                            logger.info(
                                "[RAG][LINK][POST] has_link=%s next=%r", has_link, nxt
                            )

                        current_text = fixed
                        seen_norm.add(self.db.normalize_query(fixed))
                        logger.info(
                            "[RAG][FIX][APPLY] current_text=%r", current_text
                        )
                except Exception as e:
                    logger.exception(
                        "[RAG][TOPIC][ERROR] Failed to check US Constitution relevance: %s", e
                    )
                    logger.info(
                        "[RAG][TOPIC][FALLBACK] Continuing with RAG processing despite topic check failure"
                    )


            # ==================== 1) CASES path ====================
            # Use same thresholds as MAIN path (cases are already filtered by relevance)
            # Cap thresholds at 1.0 to ensure they're achievable
            CASES_LLM_VERIF = min(self.LLM_VERIF, 1.0)
            CASES_RAG_SEARCH = min(self.RAG_SEARCH, 1.0)
            CASES_LLM_SCORE = min(self.LLM_SCORE + 0.05, 1.0)  # Small boost for verified cases
            
            # Initialize case-mapped results list to merge with MAIN results later
            case_mapped_results = []
            
            # Try cases search with error handling - if it fails, continue to MAIN path
            # Skip cases search if skip_cases_search flag is set (e.g., for general questions)
            sem_cases = None
            top_cases = 0.0
            try:
                if not skip_cases_search and hasattr(self.vector_search, 'search_cases') and self.vector_search.search_cases:
                    sem_cases = self.vector_search.search_cases.search_similar(emb)
                    if sem_cases:
                        logger.info("[RAG][CASES] candidates \n%s", self._fmt_pairs(sem_cases))
                        top_cases = sem_cases[0][1] if sem_cases else 0.0
                    else:
                        logger.info("[RAG][CASES] No cases found or search returned None")
                else:
                    logger.info("[RAG][CASES] search_cases not available, skipping cases path")
            except Exception as e:
                logger.warning("[RAG][CASES] Error during cases search: %s - continuing to MAIN path", e)
                sem_cases = None
                top_cases = 0.0

            # Use RAG_MIN for entry condition to allow cases path to run more often
            # Cases will still be filtered by CASES_RAG_SEARCH and CASES_LLM_VERIF thresholds
            if sem_cases and top_cases >= self.RAG_MIN:
                logger.info(f"[RAG][CASES] Processing cases path: top_cases={top_cases:.4f} >= RAG_MIN={self.RAG_MIN:.4f}")
                logger.info(f"[RAG][CASES] Thresholds: CASES_RAG_SEARCH={CASES_RAG_SEARCH:.4f}, CASES_LLM_VERIF={CASES_LLM_VERIF:.4f}")
                
                direct_cases = [(c, s) for (c, s) in sem_cases if s >= CASES_RAG_SEARCH]
                verify_cases = [(c, s) for (c, s) in sem_cases if CASES_LLM_VERIF <= s < CASES_RAG_SEARCH]
                
                logger.info(f"[RAG][CASES] direct_cases: {len(direct_cases)} (score >= {CASES_RAG_SEARCH:.4f})")
                logger.info(f"[RAG][CASES] verify_cases: {len(verify_cases)} ({CASES_LLM_VERIF:.4f} <= score < {CASES_RAG_SEARCH:.4f})")

                verified_cases: List[Tuple[dict, float]] = []
                if verify_cases:
                    try:
                        # Use parallel verification for faster processing
                        vm = self.llmv.verify_many_parallel(current_text, verify_cases, item_type="case", max_workers=5) or []
                        verified_cases = [(cd, sc) for (cd, sc) in vm if (sc + CASES_LLM_SCORE) >= CASES_RAG_SEARCH]
                        if verified_cases:
                            logger.info("[RAG][CASES] verified_accept +bump\n%s", self._fmt_pairs(verified_cases))
                    except Exception as e:
                        logger.info("[RAG][CASES] verify_many_parallel failed: %s", e)

                accepted_cases = direct_cases + verified_cases
                logger.info(f"[RAG][CASES] accepted_cases: {len(accepted_cases)} total (direct: {len(direct_cases)}, verified: {len(verified_cases)})")
                
                if accepted_cases:
                    logger.info(f"[RAG][CASES] Mapping {len(accepted_cases)} cases to main documents via references...")
                    mapped = self._cases_to_main_by_references(accepted_cases)
                    logger.info(f"[RAG][CASES] Mapped to {len(mapped)} main documents from case references")
                    
                    if mapped:
                        logger.info(f"[RAG][CASES] Mapped documents:\n%s", self._fmt_pairs(mapped[:5]))  # Show top 5
                    
                    accepted_main, verify_need_main = self._filter_kw_alias(
                        current_text=current_text,
                        sem_main=mapped,
                        current_decay=current_decay,
                        emb=emb,
                    )
                    logger.info(f"[RAG][CASES] After filter_kw_alias: accepted={len(accepted_main)}, verify_need={len(verify_need_main)}")
                    
                    decision = self._apply_main_abc_gates(
                        current_text=current_text,
                        accepted=accepted_main,
                        need_verify=verify_need_main,
                        apply_gap=True
                    )
                    if decision is not None:
                        logger.info("[RAG][CASES] Found %d results from case mapping (will merge with MAIN results)", len(decision))
                        case_mapped_results = decision
                    else:
                        logger.info("[RAG][CASES] ABC gates returned None - no results passed gates, continuing to MAIN path")
                else:
                    logger.info("[RAG][CASES] No accepted cases (direct or verified) - continuing to MAIN path")
            else:
                if not sem_cases:
                    logger.info(f"[RAG][CASES] No cases found")
                elif top_cases < self.RAG_MIN:
                    logger.info(f"[RAG][CASES] Top case score {top_cases:.4f} < RAG_MIN {self.RAG_MIN:.4f} - skipping cases path")

            # ==================== 2) MAIN path =====================
            # Always use combined search (document + clause level)
            # Try MAIN search with error handling - if it fails, still return case results if available
            sem_main = None
            top_sem = 0.0
            try:
                if hasattr(self.vector_search, 'search_main_with_clauses'):
                    sem_main = self.vector_search.search_main_with_clauses(emb, k=20, k_clauses=10)
                    logger.info("[RAG][MAIN] Using combined search (document + clause level)")
                elif hasattr(self.vector_search, 'search_main') and self.vector_search.search_main:
                    sem_main = self.vector_search.search_main.search_similar(emb)
                    logger.info("[RAG][MAIN] Using document-level search only (fallback)")
                else:
                    logger.warning("[RAG][MAIN] search_main not available, skipping MAIN path")
            except Exception as e:
                logger.warning("[RAG][MAIN] Error during MAIN search: %s - will return case results if available", e)
                sem_main = None
            
            logger.info(
                "[RAG][MAIN] thresholds MIN=%.3f SEARCH=%.3f LLM_VERIF=%.3f δ=%.3f ALIAS=%.3f GAP=%.2f LLM_SCORE=%.2f",
                self.RAG_MIN, self.RAG_SEARCH, self.LLM_VERIF, current_decay, self.ALIAS_THR, self.FILTER_GAP, self.LLM_SCORE
            )
            if sem_main:
                logger.info("[RAG][MAIN] candidates\n%s", self._fmt_pairs(sem_main))
                # Log detailed score breakdown for top candidates
                logger.info("[RAG][MAIN][DETAIL] Top 10 candidates with scores:")
                for i, (doc, score) in enumerate(sem_main[:10], 1):
                    title = doc.get("title", "N/A")
                    article = doc.get("article", "")
                    section = doc.get("section", "")
                    logger.info("[RAG][MAIN][DETAIL] #%d: score=%.4f title=%r article=%r section=%r", 
                               i, score, title, article, section)
                top_sem = sem_main[0][1] if sem_main else 0.0
            else:
                logger.info("[RAG][MAIN] no semantic candidates")
                top_sem = 0.0

            # Ensure sem_main is a list (not None) for _filter_kw_alias
            sem_main = sem_main or []
            
            accepted_main, verify_need_main = self._filter_kw_alias(
                current_text=current_text,
                sem_main=sem_main,
                current_decay=current_decay,
                emb=emb,
            )

            decision = self._apply_main_abc_gates(
                current_text=current_text,
                accepted=accepted_main,
                need_verify=verify_need_main,
                        apply_gap=True
            )
            
            # Merge case-mapped results with MAIN results
            combined_results = []
            seen_doc_ids = set()
            
            # Add case-mapped results first (they have priority)
            if case_mapped_results:
                for doc, score in case_mapped_results:
                    doc_id = doc.get("_id") or doc.get("knowledge_id")
                    if doc_id:
                        doc_id_str = str(doc_id)
                        if doc_id_str not in seen_doc_ids:
                            combined_results.append((doc, score))
                            seen_doc_ids.add(doc_id_str)
                logger.info("[RAG][MERGE] Added %d case-mapped results", len(combined_results))
            
            # Add MAIN results (avoid duplicates)
            if decision is not None:
                main_count = 0
                for doc, score in decision:
                    doc_id = doc.get("_id") or doc.get("knowledge_id")
                    if doc_id:
                        doc_id_str = str(doc_id)
                        if doc_id_str not in seen_doc_ids:
                            combined_results.append((doc, score))
                            seen_doc_ids.add(doc_id_str)
                            main_count += 1
                logger.info("[RAG][MERGE] Added %d MAIN results (deduplicated)", main_count)
            
            # Sort combined results by score (descending)
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            if combined_results:
                # Validate all results meet threshold
                invalid = [(d, s) for d, s in combined_results if s < self.RAG_SEARCH]
                if invalid:
                    logger.warning("[RAG][VALIDATE] Found %d results below RAG_SEARCH=%.3f, filtering out", 
                                  len(invalid), self.RAG_SEARCH)
                    combined_results = [(d, s) for d, s in combined_results if s >= self.RAG_SEARCH]
                
                logger.info("[RAG][PATH] Combined results: %d total (all >= %.3f) (case-mapped: %d, MAIN: %d)\n%s", 
                           len(combined_results), self.RAG_SEARCH, len(case_mapped_results), 
                           len(decision) if decision else 0,
                           self._fmt_pairs(combined_results[:5]))
                return (combined_results[: self.TOP_K_RET], current_text)
            elif case_mapped_results:
                # Fallback: return case results if MAIN search failed or returned nothing
                # Validate all results meet threshold
                invalid = [(d, s) for d, s in case_mapped_results if s < self.RAG_SEARCH]
                if invalid:
                    logger.warning("[RAG][VALIDATE] Found %d case-mapped results below RAG_SEARCH=%.3f, filtering out", 
                                  len(invalid), self.RAG_SEARCH)
                    case_mapped_results = [(d, s) for d, s in case_mapped_results if s >= self.RAG_SEARCH]
                
                logger.info("[RAG][PATH] Returning case-mapped results only (all >= %.3f) (MAIN search had no results)\n%s", 
                           self.RAG_SEARCH, self._fmt_pairs(case_mapped_results[:5]))
                return (case_mapped_results[: self.TOP_K_RET], current_text)
            elif decision is not None:
                # Validate all results meet threshold
                invalid = [(d, s) for d, s in decision if s < self.RAG_SEARCH]
                if invalid:
                    logger.warning("[RAG][VALIDATE] Found %d MAIN results below RAG_SEARCH=%.3f, filtering out", 
                                  len(invalid), self.RAG_SEARCH)
                    decision = [(d, s) for d, s in decision if s >= self.RAG_SEARCH]
                
                logger.info("[RAG][PATH] winner=MAIN (all >= %.3f)\n%s", self.RAG_SEARCH, self._fmt_pairs(decision))
                return (decision[: self.TOP_K_RET], current_text)

            # --- rephrase decision ---
            # Enforce REPHRASE_LIMIT: don't rephrase more than the configured limit
            should_rephrase = (rephrase_count < self.REPHRASE_LIMIT) and (attempt < self.REPHRASE_LIMIT) and (top_sem >= self.RAG_MIN)
            logger.info("[RAG][REPHRASE] attempt=%d/%d rephrase_count=%d/%d should_rephrase=%s top_sem=%.3f thr_min=%.3f",
                        attempt, self.REPHRASE_LIMIT, rephrase_count, self.REPHRASE_LIMIT, bool(should_rephrase), top_sem, self.RAG_MIN)
            if not should_rephrase:
                if rephrase_count >= self.REPHRASE_LIMIT:
                    logger.info("[RAG][REPHRASE] stopping: reached REPHRASE_LIMIT=%d", self.REPHRASE_LIMIT)
                break

            # Use domain-specific document_type for rephrasing
            # This ensures each domain rephrases queries using its own terminology
            logger.info(f"[RAG][REPHRASE] Rephrasing query for document_type='{self.DOC_TYPE}' (domain-specific)")
            new_text = self.openAI.rephrase_query(current_text, self.DOC_TYPE, previous_rephrases)
            new_norm = self.db.normalize_query(new_text or "")
            if (not new_text) or (new_norm in seen_norm):
                logger.info("[RAG][REPHRASE] stopping: empty_or_seen=%s", not bool(new_text))
                break

            self.qm.update_query_rephrased_ref(self.db, current_text, new_text)
            previous_rephrases.append(new_text)
            seen_norm.add(new_norm)
            current_text = new_text
            rephrase_count += 1  # Increment actual rephrase count
            current_decay = min(current_decay + 0.005, 0.05)

        logger.info("[RAG][END] no results for query=%r after %d attempts", self._original_query or query, self.REPHRASE_LIMIT + 1)
        return ([], self._original_query or current_text)
    def process_query_light(self, filtered_cases: List[Any], query: str) -> List[Tuple[dict, float]]:
        """
        Light path: search ONLY within the filtered cases provided by the caller.
        `filtered_cases` may be a list of titles (str) and/or docs (dict).
        Uses QueryManager.search_similar(...) to compute similarities and then
        resolves *only those matched titles* back to docs for the return list.

        Returns: List[(doc_without_embedding, score_float)]
        
        NOTE: Initialization and ingestion must be done BEFORE calling this function.
        """
        if not filtered_cases:
            logger.info("[LIGHT] No cases provided -> []")
            return []

        # 1) sanitize / normalize
        norm = self.db.normalize_query(query)

        # 2) NEW: Incremental search with ObjectId-based range tracking
        cached_visible_results = []
        cached_titles_set = set()
        cases_to_search = []
        current_case_ids = []
        
        # Extract ObjectIds and titles from filtered_cases
        def _norm_title(s: str) -> str:
            return " ".join((s or "").split()).lower()
        
        case_id_to_title = {}  # Map ObjectId -> title for quick lookup
        title_to_case = {}     # Map normalized title -> full case dict
        
        for fc in filtered_cases:
            if isinstance(fc, dict):
                case_id = str(fc.get("_id", ""))
                title = fc.get("title", "")
                if case_id:
                    current_case_ids.append(case_id)
                    case_id_to_title[case_id] = title
                    title_to_case[_norm_title(title)] = fc
            else:
                # String title only - no ObjectId available
                title_to_case[_norm_title(str(fc))] = {"title": str(fc)}
        
        logger.info("[LIGHT][RANGE] Current search range: %d cases with ObjectIds", len(current_case_ids))
        
        try:
            # Find similar query in cache (semantic search on query)
            cached_query_doc = self.qm.find_cached_similar_query(self.db, query, self.db.query)
            
            if cached_query_doc:
                logger.info("[LIGHT][CACHE] Found similar cached query (semantic match)")
                
                # Get cached search range (ObjectIds)
                cached_case_ids = self.qm.get_cached_search_range(cached_query_doc)
                
                if cached_case_ids and current_case_ids:
                    # Identify new, removed, and existing cases
                    new_ids, removed_ids, existing_ids = self.qm.identify_new_cases(
                        current_case_ids, 
                        cached_case_ids
                    )
                    
                    # Get cached results for existing cases (that are still visible)
                    cached_case_titles = self.qm.get_cached_case_titles(cached_query_doc)
                    cached_docs = self.db.get_cases_by_titles(cached_case_titles) or []
                    
                    for doc in cached_docs:
                        doc_id = str(doc.get("_id", ""))
                        doc_title = doc.get("title", "")
                        doc_title_norm = _norm_title(doc_title)
                        
                        # Check if this cached case still exists in current range
                        if doc_id in existing_ids or doc_title_norm in title_to_case:
                            view = dict(doc)
                            view.pop("embedding", None)
                            # Use cached score (preserve original score)
                            # Try to get original score from cached results
                            cached_score = 0.95  # Default high score
                            for result in cached_query_doc.get("results", []):
                                if isinstance(result, dict) and result.get("title") == doc_title:
                                    cached_score = result.get("score", 0.95)
                                    break
                            
                            cached_visible_results.append((view, cached_score))
                            cached_titles_set.add(doc_title_norm)
                            logger.info("[LIGHT][CACHE] ✓ Reusing cached result: %r (score: %.3f)", doc_title, cached_score)
                    
                    # Build cases_to_search: only NEW cases (NOT in existing_ids)
                    existing_ids_set = set(existing_ids)  # For fast lookup
                    
                    for case_id in new_ids:
                        # Double-check: ensure this case is NOT in existing_ids
                        if case_id in existing_ids_set:
                            logger.warning("[LIGHT][INCREMENTAL] Skipping case %s (already in existing_ids)", case_id)
                            continue
                        
                        title = case_id_to_title.get(case_id)
                        if title:
                            title_norm = _norm_title(title)
                            case_dict = title_to_case.get(title_norm)
                            if case_dict:
                                cases_to_search.append(case_dict)
                    
                    logger.info("[LIGHT][INCREMENTAL] Cached: %d, New to search: %d, Removed: %d, Existing (skipped): %d",
                               len(cached_visible_results), len(cases_to_search), len(removed_ids), len(existing_ids))
                    
                    if removed_ids:
                        logger.info("[LIGHT][INCREMENTAL] %d cases were archived/removed since last search", len(removed_ids))
                    
                    # CRITICAL: If no new cases to search, return immediately
                    # Don't re-search cases already in the cached range
                    if not cases_to_search:
                        if cached_visible_results:
                            logger.info("[LIGHT][INCREMENTAL] ✓ No new cases, returning %d cached results", 
                                       len(cached_visible_results))
                            return cached_visible_results[:self.TOP_K_RET]
                        else:
                            logger.info("[LIGHT][INCREMENTAL] ✓ Search range covers all cases, no results found previously, returning empty")
                            return []
                else:
                    # No cached range or no current IDs - search all
                    logger.info("[LIGHT][CACHE] No search range tracking, searching all cases")
                    cases_to_search = filtered_cases
            else:
                # No cached query found - search all cases
                logger.info("[LIGHT] No similar cached query found, searching all cases")
                cases_to_search = filtered_cases
                
        except Exception as e:
            logger.exception("[LIGHT][CACHE] Failed to retrieve cached results: %s", e)
            # Fallback: search all cases
            cases_to_search = filtered_cases

        # 3) old cache fast-path (keep for backward compatibility)
        probe = self._follow_rephrases_or_cached(query, max_hops=self.REPHRASE_LIMIT, top_k=self.TOP_K_RET)
        if probe.get("cached_results"):
            logger.info("[LIGHT] Using old-style cached results for norm=%r", norm)
            # Convert cached dicts to (doc, score) tuples
            cached_with_scores = []
            for res in probe["cached_results"]:
                if isinstance(res, dict):
                    # Extract score if stored, otherwise default to 1.0
                    score = float(res.get("score", 1.0))
                    cached_with_scores.append((res, score))
                elif isinstance(res, tuple) and len(res) == 2:
                    # Already a tuple, use as-is
                    cached_with_scores.append(res)
                else:
                    # Fallback
                    cached_with_scores.append((res, 1.0))
            logger.info("[LIGHT] Returning %d old-style cached results as tuples", len(cached_with_scores))
            return cached_with_scores

        # 4) query embedding
        emb, _cached = self.qm.get_or_create_query_embedding(norm, self.db, previous_rephrases=[])
        if emb is None:
            logger.info("[LIGHT] Missing/invalid query embedding -> []")
            # If we have cached results but no embedding, return cached results
            if cached_visible_results:
                logger.info("[LIGHT] Returning %d cached results (no embedding)", len(cached_visible_results))
                return cached_visible_results
            return []

        # 5) Search logic has been moved up to section 2 (incremental search)
        # cases_to_search is now populated based on cache range comparison
        
        logger.info("[LIGHT] Searching %d cases (cached: %d, incremental: %s)", 
                   len(cases_to_search), len(cached_visible_results), 
                   "yes" if current_case_ids else "no")
        
        # If no cases to search and we have cached results, return them
        if not cases_to_search:
            if cached_visible_results:
                logger.info("[LIGHT] All results from cache, no new search needed")
                return cached_visible_results[:self.TOP_K_RET]
            logger.info("[LIGHT] No cases to search -> []")
            return []
        
        #    This uses the function you supplied and returns e.g. [ ({"CASE-2025-0006": 0.812},), ... ]
        raw_sims: List[tuple[Dict[str, float]]] = self.qm.search_similar(
            db=self.db,
            cleaned_query=query,
            vec=emb,
            titles=cases_to_search,
        )
        if not raw_sims:
            logger.info("[LIGHT] No matched cases with embeddings -> []")
            # Return cached results if available
            if cached_visible_results:
                logger.info("[LIGHT] No new matches, returning %d cached results", len(cached_visible_results))
                return cached_visible_results[:self.TOP_K_RET]
            return []

        # 6) extract matched titles from results (preserve order)
        matched_titles: List[str] = []
        matched_scores: Dict[str, float] = {}
        for item in raw_sims:
            if isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], dict) and item[0]:
                t, s = next(iter(item[0].items()))
                try:
                    t_str = str(t)
                    matched_titles.append(t_str)
                    matched_scores[t_str] = float(s)
                except Exception:
                    continue

        if not matched_titles:
            logger.info("[LIGHT] No titles resolved from sims -> []")
            # Return cached results if available
            if cached_visible_results:
                logger.info("[LIGHT] No new titles, returning %d cached results", len(cached_visible_results))
                return cached_visible_results[:self.TOP_K_RET]
            return []

        # 7) resolve ONLY the matched titles back to docs
        #    Note: get_cases_by_titles itself de-dupes and preserves input order.
        docs = self.db.get_cases_by_titles(matched_titles) or []
        def _norm(s: str) -> str:
            return " ".join((s or "").split()).lower()
        by_title = {_norm(d.get("title", "")): d for d in docs}

        # 8) Build (doc, score) with semantic scoring only (ABC gate), keep order of matched_titles
        sims: List[Tuple[Dict[str, Any], float]] = []
        for t in matched_titles:
            d = by_title.get(_norm(t))
            if d is not None:
                view = dict(d)
                view.pop("embedding", None)  # don't leak embeddings
                
                # Get semantic score only (no BM25/Atlas search)
                sem_score = matched_scores.get(t, 1.0)
                
                logger.debug(
                    "[LIGHT][SEMANTIC] doc=%r sem=%.4f (using semantic-only with ABC gate)",
                    t, sem_score
                )
                
                sims.append((view, sem_score))
            else:
                logger.info("[LIGHT] matched title not found in docs: %r", t)

        if not sims:
            logger.info("[LIGHT] Could not resolve any sims to docs -> []")
            # Return cached results if available
            if cached_visible_results:
                logger.info("[LIGHT] No sims resolved, returning %d cached results", len(cached_visible_results))
                return cached_visible_results[:self.TOP_K_RET]
            return []

        # 10) gates — accepted vs verify
        # Include results >= RAG_SEARCH_min (0.80) for verification, not just >= LLM_VERIF (0.85)
        accepted = [(d, min(s, 1.0)) for (d, s) in sims if s >= self.RAG_SEARCH]
        need_verify = [(d, min(s, 1.0)) for (d, s) in sims if self.RAG_SEARCH > s >= self.RAG_MIN]

        decision = self._apply_main_abc_gates(
            current_text=query,
            accepted=accepted,
            need_verify=need_verify,
            apply_gap=True  # Apply gap filter to remove results with gap > FILTER_GAP from top result
        )

        new_ranked = [p for p in (decision or []) if p[1] >= self.RAG_SEARCH]
        
        # 11) Merge cached results with new results
        # Cached results come first (they were high-confidence originally)
        all_results = cached_visible_results + new_ranked
        
        # Deduplicate by title (keep first occurrence)
        seen_titles = set()
        deduplicated = []
        for doc, score in all_results:
            doc_title_norm = _norm(doc.get("title", ""))
            if doc_title_norm not in seen_titles:
                seen_titles.add(doc_title_norm)
                deduplicated.append((doc, score))
        
        # Apply gap filter to final merged results to remove results with gap > FILTER_GAP from top
        if hasattr(self, 'FILTER_GAP') and deduplicated:
            before_gap = len(deduplicated)
            deduplicated = self._gap_filter(deduplicated)
            logger.info("[LIGHT][GAP] Applied gap filter to merged results: kept=%d dropped=%d (FILTER_GAP=%.3f)", 
                       len(deduplicated), before_gap - len(deduplicated), self.FILTER_GAP)
        
        # Final threshold enforcement - ensure ALL results >= RAG_SEARCH
        before_threshold = len(deduplicated)
        deduplicated = [(d, s) for (d, s) in deduplicated if s >= self.RAG_SEARCH]
        if before_threshold != len(deduplicated):
            logger.warning("[LIGHT][THRESHOLD] Filtered %d results below RAG_SEARCH=%.3f (kept %d)", 
                          before_threshold - len(deduplicated), self.RAG_SEARCH, len(deduplicated))
        else:
            logger.info("[LIGHT][THRESHOLD] All %d results meet RAG_SEARCH=%.3f", len(deduplicated), self.RAG_SEARCH)
        
        # Apply TOP_K limit
        final_ranked = deduplicated[:self.TOP_K_RET]
        
        # 12) Store ALL results >= RAG_SEARCH threshold + search range for future caching
        # This includes both direct accepts AND LLM-verified results that got bumped up
        cacheable_results = [(d, s) for (d, s) in new_ranked if s >= self.RAG_SEARCH]
        confident_threshold = float(self.thr.get('confident', 0.90))  # Get from thresholds dict
        
        if cacheable_results or current_case_ids:
            try:
                # Store ALL results >= RAG_SEARCH, not just high-confidence ones
                # This allows better cache reuse for similar queries
                self.qm.store_case_query_pairs(
                    self.db, 
                    query, 
                    cacheable_results,  # Store all results >= RAG_SEARCH
                    searched_case_ids=current_case_ids if current_case_ids else None,
                    min_score=self.RAG_SEARCH  # Use RAG_SEARCH threshold, not confident
                )
                high_conf_count = len([s for (_, s) in cacheable_results if s >= confident_threshold])
                logger.info("[LIGHT][STORE] Cached %d results (including %d high-confidence >= %.2f) + search range (%d case IDs)",
                           len(cacheable_results), high_conf_count, confident_threshold, 
                           len(current_case_ids) if current_case_ids else 0)
            except Exception as e:
                logger.exception("[LIGHT][STORE] Failed to cache results: %s", e)
        
        logger.info(
            "[LIGHT][RET] cached=%d new_accepted=%d new_verify=%d -> final=%d (min>=%.3f, top_k=%d)",
            len(cached_visible_results), len(accepted), len(need_verify), len(final_ranked), self.RAG_SEARCH, self.TOP_K_RET
        )
        return final_ranked


    def _follow_rephrases_or_cached(self, seed_text: str, *, max_hops: Optional[int] = None, top_k: Optional[int] = None):
        # Default to REPHRASE_LIMIT if not specified, but allow override for special cases
        if max_hops is None:
            max_hops = self.REPHRASE_LIMIT
        if top_k is None:
            top_k = self.TOP_K_RET

        final = seed_text
        chain: List[str] = []
        hops = 0
        loop_detected = False
        hit_max_hops = False
        seen_norm = { self.db.normalize_query(seed_text) }

        while True:
            norm_final = self.db.normalize_query(final)

            # cache hit?
            if self.qm.check_query_has_results(self.db, norm_final):
                # Get collection_key from config to filter cached results by collection
                collection_key = self.cfg.get("collection_key")
                cached_results = self.qm.get_query_with_results(self.db, norm_final, limit=top_k, collection_key=collection_key)
                if cached_results:
                    logger.info("[RAG][CACHE] hit for %r (hop=%d) -> %d", final, hops, len(cached_results))
                    return {
                        "final_text": final,
                        "chain": chain,
                        "hops": hops,
                        "loop_detected": loop_detected,
                        "hit_max_hops": hit_max_hops,
                        "cached_results": cached_results,
                    }

            # follow rephrase edges using the same normalized key
            if not self.qm.check_query_has_update_reference(self.db, norm_final):
                break

            nxt = self.qm.get_query_with_rephrase(self.db, norm_final)
            if not nxt:
                break

            nxt_norm = self.db.normalize_query(nxt)
            if nxt_norm in seen_norm:
                loop_detected = True
                logger.info("[RAG][REPHRASE] loop detected: %r -> %r; breaking.", final, nxt)
                break

            if hops >= max_hops:
                hit_max_hops = True
                logger.info("[RAG][REPHRASE] reached MAX_HOPS=%d; breaking.", max_hops)
                break

            seen_norm.add(nxt_norm)
            chain.append(nxt)
            final = nxt
            hops += 1

        return {
            "final_text": final,
            "chain": chain,
            "hops": hops,
            "loop_detected": loop_detected,
            "hit_max_hops": hit_max_hops,
            "cached_results": None,
        }

    # ---------- pretty log helpers ----------
    def _name(self, doc: dict) -> str:
        return (doc.get("case") or doc.get("title") or doc.get("article") or str(doc.get("_id")) or "").strip()

    def _mid(self, doc: dict) -> str:
        try:
            return str(doc.get("_id"))
        except Exception:
            return "<no-id>"

    def _short(self, text: str, limit: int = 64) -> str:
        text = text or ""
        return text if len(text) <= limit else (text[:limit - 1] + "…")

    def _fmt_pairs(self, pairs, n: int = 100) -> str:
        lines = []
        for d, s in pairs[:n]:
            nm = self._short(self._name(d))
            id6 = self._mid(d)[-6:]
            lines.append(f"  • {nm} [{id6}]  score={s:0.3f}")
        return "\n".join(lines) if lines else "  (none)"

    def _fmt_case_refs(self, cases: list[tuple[dict, float]] | list[dict], n: int = 5) -> str:
        lines = []
        seq: list[tuple[dict, float]] = []
        if cases and isinstance(cases[0], tuple):
            seq = cases  # type: ignore
        elif cases and isinstance(cases[0], dict):
            seq = [(c, float(c.get("score", 0.0))) for c in cases]  # type: ignore
        for case, score in seq[:n]:
            case_name = self._short(self._name(case))
            cid = self._mid(case)[-6:]
            refs = case.get("references") or []
            ref_txt = ", ".join(refs) if refs else "(no refs)"
            lines.append(f"  • {case_name} [{cid}]  score={score:0.3f}\n      ↳ refs: {ref_txt}")
        return "\n".join(lines) if lines else "  (none)"

    ############### INSIGHT / EXPLANATION HANDLING ###############
    def get_or_create_insight_by_index(
        self,
        *,
        query: str,
        result_list: list,
        index: int,
        language: str = "en",
        query_en: Optional[str] = None,  # Pre-translated English query to avoid redundant translation
    ) -> str:
        """
        Policy:
        - NEVER regenerate insight if a cached insight exists for this (normalized_query, index[, kid]).
        - Cached queries (served from summaries / results) should always reuse this.
        - All OpenAI calls go through self.openAI.insight_explain (service layer).
        """
        logger.info(
            "[INSIGHTIDX][IN] query=%r | q_len=%d | results=%d | index=%d",
            query[:100] if query else "",
            len(query or ""),
            len(result_list or []),
            index,
        )

        if not result_list:
            logger.warning("[INSIGHTIDX] empty result_list")
            return ""
        if not (0 <= index < len(result_list)):
            logger.warning(
                "[INSIGHTIDX] index out of range (0..%d)", len(result_list) - 1
            )
            return ""

        item = result_list[index]

        def _to_oid(x):
            if isinstance(x, ObjectId):
                return x
            try:
                return ObjectId(str(x)) if x is not None else None
            except Exception:
                return None

        def _fetch_by_id(oid):
            try:
                return self.db.main.find_one(
                    {"_id": oid},
                    {"text_embedding": 0, "summary_embedding": 0},
                )
            except Exception:
                logger.exception(
                    "[INSIGHTIDX] fetch main by id failed: %s", oid
                )
                return None

        def _fetch_by_title(title: str):
            if not title:
                return None
            key = self.cfg.get("unique_index", "title")
            try:
                doc = self.db.main.find_one(
                    {key: title},
                    {"text_embedding": 0, "summary_embedding": 0},
                )
                if not doc:
                    logger.warning(
                        "[INSIGHTIDX] main lookup by %s=%r returned None",
                        key,
                        title,
                    )
                return doc
            except Exception:
                logger.exception("[INSIGHTIDX] fetch main by %s failed", key)
                return None

        # ---- unpack item (supports (doc,score), dict, ObjectId) ----
        doc, score, kid = None, None, None
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            d, s = item[0], item[1]
            score = float(s) if isinstance(s, (int, float)) else None
            if isinstance(d, dict):
                doc = d
                kid = _to_oid(d.get("_id"))
            elif isinstance(d, ObjectId):
                kid = d
                doc = _fetch_by_id(kid)
            elif isinstance(d, list) and d and isinstance(d[0], dict):
                doc = d[0]
                kid = _to_oid(doc.get("_id"))
        elif isinstance(item, dict):
            doc = item
            kid = _to_oid(item.get("_id")) or _to_oid(item.get("knowledge_id"))
            if "score" in item and item["score"] is not None:
                score = float(item["score"])
        elif isinstance(item, ObjectId):
            kid = item
            doc = _fetch_by_id(kid)
        else:
            logger.warning(
                "[INSIGHTIDX] unsupported result type=%s", type(item).__name__
            )

        # ---- ENRICH missing doc/body via kid or title ----
        if kid and (not doc or not (doc.get("summary") or doc.get("text"))):
            full = _fetch_by_id(kid)
            if full:
                doc = full
                logger.info(
                    "[INSIGHTIDX] enriched cached result by knowledge_id -> kid=%s",
                    str(kid)[-6:],
                )

        if doc and not kid and not (doc.get("summary") or doc.get("text")):
            title = (
                doc.get("title")
                or doc.get("article")
                or doc.get("section")
                or ""
            ).strip()
            full = _fetch_by_title(title)
            if full:
                doc = full
                kid = _to_oid(full.get("_id"))
                logger.info(
                    "[INSIGHTIDX] enriched cached result by title %r -> kid=%s",
                    title,
                    (str(kid)[-6:] if kid else None),
                )

        logger.info(
            "[INSIGHTIDX] kid=%s | has_doc=%s | score=%s",
            (str(kid)[-6:] if kid else None),
            bool(doc),
            score,
        )

        # ---- Extract document identifiers for matching when knowledge_id is missing ----
        def _normalize_identifier(text: str) -> str:
            """Normalize identifier for comparison (lowercase, strip whitespace)"""
            return " ".join((text or "").split()).lower()
        
        # Extract and normalize document identifiers from current doc
        doc_title = _normalize_identifier(doc.get("title") if doc else None)
        doc_article = _normalize_identifier(doc.get("article") if doc else None)
        doc_section = _normalize_identifier(doc.get("section") if doc else None)
        
        def _doc_identifiers_match(cached_doc_identifiers: Dict[str, str]) -> bool:
            """
            Compare document identifiers to verify document match.
            Returns True only if all available identifiers match.
            """
            if not cached_doc_identifiers:
                return False
            
            cached_title = _normalize_identifier(cached_doc_identifiers.get("title"))
            cached_article = _normalize_identifier(cached_doc_identifiers.get("article"))
            cached_section = _normalize_identifier(cached_doc_identifiers.get("section"))
            
            # If we have identifiers in current doc, they must all match
            matches = []
            if doc_title:
                matches.append(cached_title == doc_title)
            if doc_article:
                matches.append(cached_article == doc_article)
            if doc_section:
                matches.append(cached_section == doc_section)
            
            # If we have no identifiers in current doc, cannot verify match
            if not matches:
                return False
            
            # All available identifiers must match
            return all(matches)
        
        # ---- 1) CACHE CHECK: Check exact query AND semantically similar queries ----
        # Policy: 
        # - Check if query (exact or similar) has results AND insights
        # - If insights exist for this specific document (matching knowledge_id or identifiers), REUSE them
        # - If results exist but insights don't exist for this document, GENERATE them
        # - Only reuse insights when there's an exact match (same knowledge_id or matching document identifiers)
        # Always use English query for cache lookup and semantic search
        # Use pre-translated query if provided to avoid redundant API calls
        from services.rag.rag_dependencies.ai_service import translate_query
        
        # For cache lookup, always use English query
        query_for_cache = query_en if query_en else query
        if language == "es" and not query_en:
            # Only translate if not already provided (avoid redundant translation)
            translated = translate_query(query, source_lang="es", target_lang="en")
            if translated and translated != query:
                query_for_cache = translated
                logger.info(f"[INSIGHTIDX][CACHE] Translated query for cache lookup: {translated[:50]}...")
        elif query_en:
            logger.info(f"[INSIGHTIDX][CACHE] Using pre-translated English query for cache lookup: {query_en[:50]}...")
        
        norm_query = self.db.normalize_query(query_for_cache or "")
        queries_to_check = [norm_query]  # Start with exact query
        
        logger.info(
            "[INSIGHTIDX][CACHE][START] query=%r (cache=%r) norm_query=%r index=%d kid=%s lang=%s doc_title=%r",
            query[:100] if query else "",
            query_for_cache[:100] if query_for_cache else "",
            norm_query,
            index,
            str(kid)[-6:] if kid else "None",
            language,
            doc_title[:50] if doc_title else "None"
        )
        
        # Fix 4: Check request-level insight cache first (before any API calls)
        cache_key = (norm_query, str(kid) if kid else None, language)
        if cache_key in self._request_insight_cache:
            logger.debug(
                "[INSIGHTIDX][REQUEST_CACHE] Hit for norm_query=%r kid=%s lang=%s",
                norm_query,
                str(kid)[-6:] if kid else "None",
                language
            )
            return self._request_insight_cache[cache_key]
        
        # Add semantically similar queries using query embedding search
        try:
            # Fix 1: Check if we already have embedding from main query processing
            # Avoid redundant embedding generation when embedding is already cached
            if self._cached_query_embedding is not None:
                query_emb = self._cached_query_embedding
                logger.debug("[INSIGHTIDX][CACHE][EMB] Reusing cached embedding from main query processing")
            else:
                # Get query embedding for semantic search
                query_emb, _ = self.qm.get_or_create_query_embedding(query_for_cache, self.db, previous_rephrases=[])
                # Cache it for this request to avoid redundant calls
                self._cached_query_embedding = query_emb
            
            if query_emb is not None:
                # Find semantically similar queries
                similar_query_doc = self.qm.find_cached_similar_query(self.db, query_for_cache, self.db.query)
                if similar_query_doc:
                    # Try both 'query' and 'query_norm' fields
                    similar_query_text = similar_query_doc.get("query") or similar_query_doc.get("query_norm") or ""
                    similar_norm = self.db.normalize_query(similar_query_text)
                    if similar_norm and similar_norm not in queries_to_check:
                        queries_to_check.append(similar_norm)
                        logger.info(
                            "[INSIGHTIDX][CACHE][SEMANTIC] Found semantically similar query: %r -> %r (score=%.3f)",
                            query_for_cache[:50],
                            similar_query_text[:50] if similar_query_text else "unknown",
                            similar_query_doc.get("score", 0.0)
                        )
                    elif not similar_norm:
                        logger.warning(
                            "[INSIGHTIDX][CACHE][SEMANTIC] Similar query document found but no query text available (fields: %s)",
                            list(similar_query_doc.keys())
                        )
        except Exception as e:
            logger.debug("[INSIGHTIDX][CACHE][SEMANTIC] Semantic query matching failed: %s", e)
        
        # Track why cached insights were skipped (for better error messages)
        skipped_reasons = {
            "no_text": 0,
            "wrong_language": 0,
            "index_mismatch": 0
        }
        
        # Check each query (exact + semantically similar) for cached insights
        for check_query in queries_to_check:
            # Determine if this is the exact query or a semantically similar one
            is_exact_query = (check_query == norm_query)
            
            try:
                rows = self.qm.get_query_with_insights(
                    self.db, check_query, limit=200, language=language
                ) or []
                
                if not rows:
                    continue
                    
                logger.info(
                    "[INSIGHTIDX][CACHE] rows=%d lang=%s for query=%r (exact=%s) - matching by index=%d only",
                    len(rows),
                    language,
                    check_query,
                    is_exact_query,
                    index
                )
                
                # Match insights by index only (not knowledge_id)
                # Insights are aligned with results array by index position
                for r in rows:
                    # Match by index only
                    if int(r.get("index", -1)) != int(index):
                        continue  # Check next row
                    
                    # Get text with language preference
                    txt = None
                    if language == "es":
                        # For Spanish, ONLY return if text_es exists - don't fallback to English
                        if "text_es" in r and r["text_es"]:
                            txt = r["text_es"].strip()
                        else:
                            skipped_reasons["wrong_language"] += 1
                            logger.debug(
                                "[INSIGHTIDX][CACHE] Skipping cached insight - no text_es available for Spanish request"
                            )
                            continue  # Check next row
                    elif language == "en":
                        # For English, prefer text_en, fallback to text
                        if "text_en" in r and r["text_en"]:
                            txt = r["text_en"].strip()
                        else:
                            txt = (r.get("text") or r.get("summary") or "").strip()
                    else:
                        txt = (r.get("text") or r.get("summary") or "").strip()
                    
                    if not txt:
                        skipped_reasons["no_text"] += 1
                        continue  # Check next row
                    
                    # Found matching cached insight - return it immediately
                    logger.info(
                        "[INSIGHTIDX][CACHE] HIT index=%d kid=%s lang=%s (query=%r, %s)",
                        index,
                        str(kid)[-6:] if kid else "None",
                        language,
                        check_query,
                        "exact match" if check_query == norm_query else "semantic match"
                    )
                    # Track insight cache hit
                    try:
                        self.db.track_query_cache_hit(check_query, cache_type="insight")
                    except Exception as e:
                        logger.debug(f"[INSIGHTIDX][TRACK] Failed to track insight cache hit: {e}")
                    # Fix 4: Also cache in request-level cache for consistency
                    self._request_insight_cache[cache_key] = txt
                    return txt  # Return immediately when match found
            except Exception as e:
                logger.debug("[INSIGHTIDX][CACHE] read failed for query=%r: %s", check_query, e)
                continue

        # If we reach here, there is NO cached insight for this exact (norm_query, index, knowledge_id) combination.
        # We do NOT search across other queries - insights are only reused for the SAME QUERY (SAME OBJECT).
        
        if not doc:
            logger.warning("[INSIGHTIDX] no doc to summarize at index=%d", index)
            return ""

        # ---- 2) BUILD INPUTS FOR SERVICE-LEVEL OPENAI CALL ----
        cases = doc.get("cases") if isinstance(doc.get("cases"), list) else None
        if cases:
            cases = [c for c in cases if isinstance(c, dict)][:3]
        main_doc = {
            k: v
            for k, v in doc.items()
            if k in ("title", "article", "section", "summary", "text", "sections", "clauses", "document_type")
        }
        logger.info(
            "[INSIGHTIDX][DOC] title=%r | has_text=%s | cases_len=%d",
            (
                main_doc.get("title")
                or main_doc.get("article")
                or main_doc.get("section")
            ),
            bool(main_doc.get("summary") or main_doc.get("text")),
            len(cases or []),
        )

        # ---- 3) CALL OPENAI SERVICE (ONLY HERE) ----
        # WARNING: This should only be called if no cached insight was found after ALL checks
        # Double-check: if we have results cached, we should NOT be here
        total_skipped = sum(skipped_reasons.values())
        if self.qm.check_query_has_results(self.db, norm_query):
            # Check if we skipped insights for legitimate reasons (different documents, etc.)
            if total_skipped > 0:
                # This is expected - cached insights exist but are for different documents/knowledge_ids
                logger.info(
                    "[INSIGHTIDX][OPENAI] Results exist for query=%r but no matching insight found (expected: skipped %d insights: %s). Generating new insight for kid=%s.",
                    norm_query,
                    total_skipped,
                    ", ".join(f"{k}={v}" for k, v in skipped_reasons.items() if v > 0),
                    str(kid)[-6:] if kid else "None"
                )
            else:
                # This is unexpected - results exist but no insights were found at all
                logger.warning(
                    "[INSIGHTIDX][OPENAI][WARNING] Results exist for query=%r but no insights found in cache. This may indicate a cache inconsistency.",
                    norm_query
                )
        
        # Note: We should generate insights if they don't exist, even if query was recently processed.
        # The cache check above already handles reusing existing insights correctly.
        # Only generate if no matching cached insight was found (which is the case if we reached here).
        
        logger.warning(
            "[INSIGHTIDX][OPENAI] No cached insight found after ALL checks - generating new insight for query=%r index=%d kid=%s lang=%s",
            query[:100] if query else "",
            index,
            str(kid)[-6:] if kid else None,
            language
        )
        
        # Generate insights based on language
        # If language is "es", generate both English and Spanish in parallel
        # If language is "en", only generate English
        insight_en = ""
        insight_es = None
        
        def generate_english_insight():
            """Generate English insight"""
            try:
                if self.openAI and hasattr(self.openAI, "insight_explain"):
                    return self.openAI.insight_explain(
                        main_doc=main_doc,
                        query=query_for_cache,  # Use English query for insight generation
                        case_doc=cases,
                        qm=self.qm,
                        db=self.db,
                        knowledge_id=kid,
                        max_tokens=1000,
                        language="en",
                    ) or ""
                else:
                    svc = getattr(self.qm, "openAI", None)
                    if svc and hasattr(svc, "insight_explain"):
                        return svc.insight_explain(
                            main_doc=main_doc,
                            query=query_for_cache,
                            case_doc=cases,
                            qm=self.qm,
                            db=self.db,
                            knowledge_id=str(kid) if kid else None,
                            max_tokens=1000,
                            language="en",
                        ) or ""
                    else:
                        t = (
                            main_doc.get("title")
                            or main_doc.get("article")
                            or main_doc.get("section")
                            or "This provision"
                        ).strip()
                        b = (main_doc.get("summary") or main_doc.get("text") or "").strip()[:400]
                        return (f"{t} is related because it frames the key authority at issue. " + b).strip()
            except Exception as e:
                logger.exception("[INSIGHTIDX] LLM insight_explain (English) failed: %s", e)
                return ""
        
        def generate_spanish_insight():
            """Generate Spanish insight"""
            try:
                if self.openAI and hasattr(self.openAI, "insight_explain"):
                    return self.openAI.insight_explain(
                        main_doc=main_doc,
                        query=query_for_cache,  # Use English query for insight generation
                        case_doc=cases,
                        qm=self.qm,
                        db=self.db,
                        knowledge_id=kid,
                        max_tokens=1000,
                        language="es",
                    ) or ""
                else:
                    svc = getattr(self.qm, "openAI", None)
                    if svc and hasattr(svc, "insight_explain"):
                        return svc.insight_explain(
                            main_doc=main_doc,
                            query=query_for_cache,
                            case_doc=cases,
                            qm=self.qm,
                            db=self.db,
                            knowledge_id=str(kid) if kid else None,
                            max_tokens=1000,
                            language="es",
                        ) or ""
                    else:
                        # Fallback: translate English insight
                        from services.rag.rag_dependencies.ai_service import translate_insight
                        return translate_insight(insight_en, "en", "es") or ""
            except Exception as e:
                logger.exception("[INSIGHTIDX] LLM insight_explain (Spanish) failed: %s", e)
                return ""
        
        # Generate insights
        if language == "es":
            # Generate both English and Spanish insights in parallel
            logger.info("[INSIGHTIDX][PARALLEL] Generating both English and Spanish insights in parallel")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_en = executor.submit(generate_english_insight)
                future_es = executor.submit(generate_spanish_insight)
                
                insight_en = (future_en.result() or "").strip()
                insight_es = (future_es.result() or "").strip()
                
                logger.info("[INSIGHTIDX][PARALLEL] English insight len=%d, Spanish insight len=%d", 
                           len(insight_en), len(insight_es) if insight_es else 0)
        else:
            # Only generate English insight
            logger.info("[INSIGHTIDX] Generating English insight only")
            insight_en = generate_english_insight()
            insight_en = (insight_en or "").strip()
        
        # Use appropriate insight based on language for return value
        insight = insight_es if (language == "es" and insight_es) else insight_en
        
        logger.info("[INSIGHTIDX][LLM] English len=%d, Spanish len=%d, returning len=%d", 
                   len(insight_en), len(insight_es) if insight_es else 0, len(insight))
        
        if not insight_en:
            logger.warning("[INSIGHTIDX] No English insight generated, returning empty")
            return ""

        # ---- 3) SAVE UNDER NORMALIZED QUERY KEY (ONE PLACE) ----
        # Save insight under the exact normalized query (not rephrased or similar queries)
        try:
            if not self.debug_mode:
                # Save under the exact normalized query only
                save_query = norm_query
                
                if save_query:
                    # Store both English and Spanish insights when available
                    # Insights are stored by index only (aligned with results array)
                    self.qm.update_query_with_insight(
                        self.db,
                        save_query,
                        text=insight,  # Legacy field - current language
                        index=index,
                        language=language,
                        insight_en=insight_en if insight_en else None,
                        insight_es=insight_es if insight_es else None,
                    )
                    logger.info(
                        "[INSIGHTIDX][SAVE] ok index=%d lang=%s (en_len=%d, es_len=%d) (saved under query=%r, original=%r)",
                        index,
                        language,
                        len(insight_en),
                        len(insight_es) if insight_es else 0,
                        save_query,
                        norm_query,
                    )
                else:
                    logger.info(
                        "[INSIGHTIDX][SAVE] No results found for query=%r or rephrase chain; not saving insight",
                        norm_query,
                    )
        except Exception:
            logger.exception("[INSIGHTIDX][SAVE] failed")

        # Fix 4: Cache the generated insight in request-level cache
        self._request_insight_cache[cache_key] = insight

        return insight

    def get_summary(
    self,
    *,
    query: str,
    result_list: List[Tuple[dict, float]],
    index: int,
    ) -> str:
        """
        Client-case only: return the summary 'as-is' from Mongo/document.
        - No LLM calls, no generation.
        - First tries the in-memory doc at result_list[index].
        - If missing, attempts a direct Mongo lookup to fetch the 'summary' field.
        Returns empty string if unavailable.
        """
        try:
            # ---- validate inputs ----
            if not result_list:
                logger.warning("[SUMMARY][LIGHT] empty result_list")
                return ""
            if index < 0 or index >= len(result_list):
                logger.warning("[SUMMARY][LIGHT] index out of range: %d (len=%d)", index, len(result_list))
                return ""

            # ---- extract the doc from (doc, score) ----
            item = result_list[index]
            doc = item[0] if isinstance(item, (list, tuple)) and item and isinstance(item[0], dict) else (
                item if isinstance(item, dict) else None
            )
            if not isinstance(doc, dict):
                logger.warning("[SUMMARY][LIGHT] item at idx=%d does not contain a dict", index)
                return ""

            # ---- use in-memory summary if present ----
            s = doc.get("summary")
            if isinstance(s, str) and s.strip():
                return s.strip()

            # ---- fallback: fetch from Mongo by id / unique key / title ----
            # Use self.cfg (set in __init__) instead of self.config
            unique_key = self.cfg.get("unique_index", "title")
            coll_name = self.cfg.get("cases_collection_name") or self.cfg.get("main_collection_name")
            
            # Access database through MongoManager: self.db is MongoManager, self.db.db is the MongoDB database
            database = getattr(self.db, "db", None) if self.db else None

            if database is None or not coll_name:
                logger.info("[SUMMARY][LIGHT] no database handle or collection name; returning empty summary")
                logger.debug("[SUMMARY][LIGHT] database=%s, coll_name=%s, db=%s", database, coll_name, self.db)
                return ""

            coll = database[coll_name]

            # Build a sane lookup query priority: _id > unique_key > title
            query_filter = None
            if doc.get("_id") is not None:
                query_filter = {"_id": doc["_id"]}
            elif doc.get(unique_key):
                query_filter = {unique_key: doc[unique_key]}
            elif doc.get("title"):
                query_filter = {"title": doc["title"]}

            if not query_filter:
                logger.info("[SUMMARY][LIGHT] no lookup key on doc; returning empty summary")
                return ""

            found = coll.find_one(query_filter, {"summary": 1})
            if found and isinstance(found.get("summary"), str) and found["summary"].strip():
                return found["summary"].strip()

            # Nothing available
            return ""
        except Exception as e:
            logger.exception("[SUMMARY][LIGHT] get_summary failed at idx=%d: %s", index, e)
            return ""

    # ----------------- internal helpers ----------------
    def _saves_confident_results(self, current_text: str, documents: Sequence[Any]) -> None:
        """
        Accepts items like:
        - (doc: dict, score: float)
        - dict with a numeric 'score' (or 'relevance'/'rank_score')
        - bare ObjectId (we'll fetch minimal doc; score falls back to 0.0)
        Saves only items with score >= self.CONFIDENT.
        """
        try:
            if self.qm.check_query_has_results(self.db, current_text):
                logging.info("[RAG][SUMMARY] results already exist for %r — skipping save.", current_text)
                return
        except Exception:
            logging.exception("[RAG][SUMMARY] check_query_has_results failed; proceeding cautiously.")

        def _extract_pair(item: Any) -> Tuple[dict | None, float | None]:
            if isinstance(item, (tuple, list)) and len(item) >= 2 and isinstance(item[1], (int, float)):
                return item[0], float(item[1])

            if isinstance(item, dict):
                for key in ("score", "relevance", "rank_score"):
                    v = item.get(key)
                    if isinstance(v, (int, float)):
                        return item, float(v)
                return None, None

            if isinstance(item, ObjectId):
                try:
                    # FIX: fetch via db.main instead of rag.collection
                    doc = self.db.main.find_one({"_id": item}, {"_id": 1, "title": 1})
                    if doc:
                        return doc, 0.0
                except Exception:
                    logging.exception("[RAG][SUMMARY] failed to fetch doc for ObjectId %s", str(item))
                return None, None

            return None, None

        normalized: List[Tuple[dict, float]] = []
        dropped = 0
        for it in (documents or []):
            doc, sc = _extract_pair(it)
            if doc is None or sc is None:
                dropped += 1
                logging.debug("[RAG][SUMMARY] dropped item type=%s", type(it).__name__)
                continue
            normalized.append((doc, sc))

        confident_results: List[Tuple[dict, float]] = [(d, s) for (d, s) in normalized if s >= self.CONFIDENT]

        logging.info(
            "[RAG][SUMMARY] normalized=%d, confident_results=%d (>= %.3f), dropped=%d",
            len(normalized), len(confident_results), self.CONFIDENT, dropped
        )

        if not confident_results:
            logging.info("[RAG][SUMMARY] no confident_results; nothing to save.")
            return

        if self.rag.debug_mode:
            logging.info("[RAG][SUMMARY] debug_mode=True — not saving %d confident_results.", len(confident_results))
            return

        try:
            payload = confident_results
            # Get collection_key from config to identify which collection these results belong to
            collection_key = self.cfg.get("collection_key")
            self.qm.update_query_with_results(self.db, current_text, payload, collection_key=collection_key)
            logging.info("[RAG][SUMMARY] saved %d confident_results for %r.", len(confident_results), current_text)
        except Exception:
            logging.exception("[RAG][SUMMARY] failed to save confident_results.")

    def _cases_to_main_by_references(self, case_pairs: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
        """
        Resolve accepted cases to main docs by title only.
        Drops article/section parsing entirely.
        """
        logger.info(f"[RAG][CASES][MAP] Starting mapping: {len(case_pairs)} cases to process")
        title_refs: set[str] = set()
        ref_sources: Dict[str, List[Tuple[dict, float]]] = {}

        def add_source(title_key: str, case_doc: dict, score: float) -> None:
            ref_sources.setdefault(title_key, []).append((case_doc, score))

        total_refs = 0
        for cdoc, sc in (case_pairs or []):
            case_name = cdoc.get("case", "Unknown")
            refs = cdoc.get("references") or []
            logger.info(f"[RAG][CASES][MAP] Case: {case_name}, score: {sc:.4f}, references: {len(refs)}")
            if not refs:
                logger.warning(f"[RAG][CASES][MAP] Case '{case_name}' has no references field or empty references")
                continue
            
            for raw in refs:
                ref = (raw or "").strip()
                if not ref:
                    continue
                total_refs += 1
                
                # Try to normalize the reference (use static method if alias is available)
                if self.alias is not None:
                    normalized = self.alias.normalize_amendment_title(ref)
                    if normalized:
                        # Use normalized title
                        t = normalized
                        logger.debug(f"[RAG][CASES][MAP] Reference: '{raw}' -> normalized: '{t}'")
                    else:
                        # Normalization failed, use raw reference (might still match)
                        t = ref
                        logger.debug(f"[RAG][CASES][MAP] Reference: '{raw}' -> using raw (normalization failed)")
                else:
                    # Alias not available, use raw reference
                    t = ref
                    logger.debug(f"[RAG][CASES][MAP] Reference: '{raw}' -> using raw (alias search disabled)")
                
                title_refs.add(t)
                add_source(t, cdoc, float(sc))

        logger.info(f"[RAG][CASES][MAP] Total references extracted: {total_refs}, unique titles: {len(title_refs)}")
        if title_refs:
            logger.info(f"[RAG][CASES][MAP] Looking up main documents for titles: {list(title_refs)[:10]}...")  # Show first 10

        if not title_refs:
            logger.warning("[RAG][CASES][MAP] No title references found - cannot map cases to main documents")
            return []

        aggregated: Dict[Any, Dict[str, Any]] = {}
        main_docs_found = 0
        
        # Convert title_refs to list for MongoDB query
        title_refs_list = list(title_refs)
        
        # Try exact match first
        for mdoc in self.db.main.find({self.unique_index: {"$in": title_refs_list}}):
            main_docs_found += 1
            mid = mdoc["_id"]
            mtitle = mdoc.get(self.unique_index)
            
            if not mtitle:
                logger.warning(f"[RAG][CASES][MAP] Main document {mid} has no {self.unique_index} field")
                continue
            
            # Check if this title matches any of our references (exact or normalized)
            matched_refs = []
            for ref_title in title_refs_list:
                # Exact match
                if mtitle == ref_title:
                    matched_refs.append(ref_title)
                # Case-insensitive match
                elif mtitle.lower() == ref_title.lower():
                    matched_refs.append(ref_title)
                    logger.debug(f"[RAG][CASES][MAP] Case-insensitive match: '{mtitle}' == '{ref_title}'")
            
            if not matched_refs:
                continue
            
            entry = aggregated.setdefault(
                mid, {"doc": {**mdoc, "cases": []}, "score": 0.0, "case_ids": set()}
            )
            
            # Add cases from all matched references
            for matched_ref in matched_refs:
                for cdoc, sc in ref_sources.get(matched_ref, []):
                    cid = cdoc.get("_id")
                    if cid in entry["case_ids"]:
                        continue
                    entry["case_ids"].add(cid)
                    entry["doc"]["cases"].append({
                        "_id": cid,
                        "case": cdoc.get("case"),
                        "score": float(sc),
                        "from_ref": matched_ref
                    })
                    entry["score"] = max(entry["score"], float(sc))
                    logger.debug(f"[RAG][CASES][MAP] Added case '{cdoc.get('case')}' (score: {sc:.4f}) to main doc '{mtitle}'")

        logger.info(f"[RAG][CASES][MAP] Found {main_docs_found} main documents matching {len(title_refs)} unique references")
        logger.info(f"[RAG][CASES][MAP] Aggregated {len(aggregated)} unique main documents with case mappings")
        
        out: List[Tuple[dict, float]] = [(v["doc"], min(v["score"], 1.0)) for v in aggregated.values()]
        out.sort(key=lambda x: x[1], reverse=True)
        
        if out:
            logger.info(f"[RAG][CASES][MAP] Returning {len(out)} mapped main documents (top score: {out[0][1]:.4f})")
        else:
            logger.warning("[RAG][CASES][MAP] No main documents found matching case references")
        
        return out

    def _filter_kw_alias(
        self,
        *,
        current_text: str,
        sem_main: List[Tuple[dict, float]],
        current_decay: float,
        emb: Any,
    ) -> Tuple[List[Tuple[dict, float]], List[Tuple[dict, float]]]:

        # Keywords / aliases
        # AliasManager is only available if use_alias_search=True in config
        text_terms = []
        alias_pairs = []
        cleaned = current_text  # Default to original text if alias is not available
        
        if self.alias is not None:
            # Ensure alias cache
            if not getattr(self.alias, "alias_cache", None):
                _loader = (getattr(self.alias, "ensure_cache", None)
                           or getattr(self.alias, "load_cache", None)
                           or getattr(self.alias, "refresh_cache", None))
                if callable(_loader):
                    try:
                        _loader(self.db)
                        # FIX: self.alias reference
                        logger.info("[ALIAS] cache loaded: %d entries", len(getattr(self.alias, "alias_cache", [])))
                    except Exception as e:
                        logger.warning("[ALIAS] cache load failed: %s", e)
                else:
                    logger.warning("[ALIAS] no cache loader found; alias search may be empty")
            
            # Clean query using alias manager
            cleaned = self.alias.clean_query(current_text)
            
            # KeywordMatcher is only available for US Constitution
            if self.kw is not None:
                text_terms = self.kw.find_textual(current_text) or []
                if not text_terms and cleaned != current_text:
                    more_terms = self.kw.find_textual(cleaned) or []
                    if more_terms:
                        text_terms = more_terms
            else:
                # If no keyword matcher, still use alias for query cleaning
                pass
            
            # Try to pass the precomputed embedding
            try:
                alias_hits = self.alias.find_semantic_aliases(cleaned, embedding=emb) or []
            except TypeError:
                try:
                    alias_hits = self.alias.find_semantic_aliases(cleaned, emb=emb) or []
                except TypeError:
                    from types import SimpleNamespace
                    alias_hits = self.alias.find_semantic_aliases(cleaned, SimpleNamespace(get_embedding=lambda _t: emb)) or []

            alias_pairs = [(c, float(s)) for (_a, c, s) in alias_hits if float(s) >= self.ALIAS_THR]
            if alias_hits:
                preview = [(a, c, round(float(s), 3)) for (a, c, s) in alias_hits[:100]]
                logger.info("[RAG][ALIAS] hits=%d keep>=%.2f top=%s", len(alias_hits), self.ALIAS_THR, preview[:8])
                # Log all alias matches above threshold with details
                logger.info("[RAG][ALIAS][DETAIL] All alias matches >= threshold (%.2f):", self.ALIAS_THR)
                for alias_text, canonical, score in alias_hits:
                    if float(score) >= self.ALIAS_THR:
                        logger.info("[RAG][ALIAS][DETAIL]   alias=%r -> canonical=%r score=%.4f", 
                                   alias_text, canonical, float(score))
            else:
                logger.info("[RAG][ALIAS] no hits")
        else:
            # Alias search disabled - use keyword matcher if available
            if self.kw is not None:
                text_terms = self.kw.find_textual(current_text) or []
            logger.debug("[RAG][ALIAS] alias search disabled for this collection")
        logger.info("[RAG][KW] terms=%s", text_terms[:8])

        # ========== SEMANTIC SEARCH ONLY (ABC gate) ==========
        # Build semantic score map by document ID
        sem_scores: Dict[Any, float] = {}
        sem_docs: Dict[Any, dict] = {}
        for doc, sem_score in (sem_main or []):
            doc_id = doc.get("_id")
            if doc_id:
                sem_scores[doc_id] = float(sem_score)
                sem_docs[doc_id] = doc

        # Use only semantic search results (no BM25/Atlas search)
        working: List[Tuple[dict, float]] = []  # (doc, sem_score)
        
        # Add all semantic search documents
        for doc_id, sem_score in sem_scores.items():
            doc = sem_docs[doc_id]
            working.append((doc, sem_score))

        # Build alias boost map (for title/article/section matching)
        # Fixed score when keyword/alias/semantic/exact match is found
        # Default to 0.70 to match config.py (was 0.8, causing incorrect scores)
        KEYWORD_MATCH_SCORE = float(self.cfg.get("KEYWORD_MATCH_SCORE", 0.70))
        term_sim_map = {t.lower(): KEYWORD_MATCH_SCORE for t in (text_terms or [])}
        for c, s in (alias_pairs or []):
            lc = c.lower()
            if s > term_sim_map.get(lc, 0.0):
                term_sim_map[lc] = s

        # ========== ABC GATE SCORING: Set fixed score when match found ==========
        accepted_dict: Dict[Any, Tuple[dict, float]] = {}
        verify_pool: List[Tuple[dict, float]] = []

        for doc, sem_score in working:
            bias = float(self.cfg.get("bias", {}).get((doc.get("title") or "").strip(), 0.0))
            title   = (doc.get("title") or "").strip()
            article = (doc.get("article") or "").strip()
            section = (doc.get("section") or "").strip()

            # Check for keyword/alias matches
            has_keyword_match = False
            if title and title.lower() in term_sim_map:
                has_keyword_match = True
            elif article and article.lower() in term_sim_map:
                has_keyword_match = True
            elif section and section.lower() in term_sim_map:
                has_keyword_match = True

            # Use semantic score only (no BM25/hybrid search)
            sem_score_float = float(sem_score)
            
            # FIX: Preserve semantic scores instead of replacing with KEYWORD_MATCH_SCORE
            # Only boost with KEYWORD_MATCH_SCORE when semantic score is low but keyword match exists
            # This prevents all documents from getting the same score (0.8)
            if has_keyword_match:
                # If keyword match exists, use the higher of semantic score or KEYWORD_MATCH_SCORE
                # This ensures keyword matches get a minimum boost but preserves high semantic scores
                total = max(sem_score_float, KEYWORD_MATCH_SCORE) + bias
            elif sem_score_float >= self.ALIAS_THR:
                # High semantic score without keyword match - use semantic score directly
                total = sem_score_float + bias
            else:
                # No match found, use original semantic score
                total = sem_score_float + bias

            # Log detailed score breakdown for debugging
            # Log when: DEBUG mode, low semantic score, or when match is found
            should_log_detail = (
                logger.isEnabledFor(logging.DEBUG) or 
                sem_score_float < 0.65 or 
                has_keyword_match or
                sem_score_float >= self.ALIAS_THR
            )
            if should_log_detail:
                logger.info(
                    "[RAG][ABC_GATE][SCORE] doc=%r sem=%.4f has_keyword_match=%s bias=%.4f total=%.4f "
                    "(RAG_SEARCH=%.3f LLM_VERIF=%.3f ALIAS_THR=%.3f KEYWORD_MATCH_SCORE=%.2f)", 
                    title, sem_score_float, has_keyword_match, bias, total,
                    self.RAG_SEARCH, self.LLM_VERIF, self.ALIAS_THR, KEYWORD_MATCH_SCORE
                )

            # Accept or verify based on total score threshold
            # Require minimum semantic score (RAG_SEARCH_min) to prevent unrelated documents from passing
            # Even if total score passes threshold, documents with very low semantic similarity should be rejected
            # This prevents keyword/alias matches from boosting semantically unrelated documents
            # Use original sem_score_float (not capped) for threshold checks to allow high semantic scores to pass
            if total >= (self.RAG_SEARCH - current_decay) and sem_score_float >= self.RAG_MIN:
                accepted_dict[doc["_id"]] = (doc, min(total, 1.0))
                logger.debug("[RAG][ACCEPT] doc=%r total=%.4f >= RAG_SEARCH=%.3f sem=%.4f >= RAG_MIN=%.3f", 
                           title, total, self.RAG_SEARCH - current_decay, sem_score_float, self.RAG_MIN)
            elif total >= (self.RAG_SEARCH - current_decay) and sem_score_float < self.RAG_MIN:
                # Total score passes but semantic score is too low - reject or send to verification
                logger.info("[RAG][REJECT_SEMANTIC_FLOOR] doc=%r total=%.4f >= RAG_SEARCH=%.3f but sem=%.4f < RAG_MIN=%.3f - rejecting", 
                           title, total, self.RAG_SEARCH - current_decay, sem_score_float, self.RAG_MIN)
                # Optionally send to verification if it's close to the floor
                if sem_score_float >= (self.RAG_MIN - 0.10) and total >= (self.LLM_VERIF - current_decay):
                    verify_pool.append((doc, total))
                    logger.debug("[RAG][VERIFY_SEMANTIC_FLOOR] doc=%r sent to verification (sem=%.4f close to RAG_MIN=%.3f)", 
                               title, sem_score_float, self.RAG_MIN)
            elif total >= (self.LLM_VERIF - current_decay):
                verify_pool.append((doc, total))
                logger.debug("[RAG][VERIFY] doc=%r total=%.4f >= LLM_VERIF=%.3f", title, total, self.LLM_VERIF - current_decay)
            else:
                logger.debug("[RAG][REJECT] doc=%r total=%.4f < LLM_VERIF=%.3f", title, total, self.LLM_VERIF - current_decay)

        accepted = sorted(accepted_dict.values(), key=lambda x: x[1], reverse=True)
        verify_pool_sorted = sorted(verify_pool, key=lambda x: x[1], reverse=True)

        logger.info("[RAG][MAIN] accepted=%d verify=%d", len(accepted), len(verify_pool_sorted))
        if accepted:
            logger.info("[RAG][MAIN][ACCEPTED] (up to %d)\n%s", self.TOP_K_RET, self._fmt_pairs(accepted, n=self.TOP_K_RET))
        if verify_pool_sorted:
            logger.info("[RAG][MAIN][VERIFY_POOL] (up to %d)\n%s", self.TOP_K_RET, self._fmt_pairs(verify_pool_sorted, n=self.TOP_K_RET))

        return accepted, verify_pool_sorted

    # ---------- local helpers (no side-effects) ----------
    def _gap_filter(self, items: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
        if not items:
            return items
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        top = items_sorted[0][1]
        return [(d, s) for (d, s) in items_sorted if (top - s) <= self.FILTER_GAP]

    def _apply_main_abc_gates(
        self,
        *,
        current_text: str,
        accepted: Optional[List[Tuple[dict, float]]],
        need_verify: Optional[List[Tuple[dict, float]]],
        apply_gap: bool  # keep all ranked results; don’t crush to 1
    ) -> Optional[List[Tuple[dict, float]]]:

        accepted = accepted or []
        need_verify = need_verify or []

        if not accepted and not need_verify:
            return None

        # --- robust dedupe key: works even when _id is missing (light path) ---
        def _dedupe_key(d: dict):
            if not isinstance(d, dict):
                return d
            k = (
                d.get("_id")
                or d.get("id")
                or d.get(getattr(self, "unique_index", "title"))
                or d.get("title")
                or d.get("case")
            )
            if k is None:
                art = d.get("article"); sec = d.get("section")
                if art or sec:
                    k = (art, sec)
            # last resort: object identity so distinct dicts never merge
            return k if k is not None else id(d)

        def _merge_keep_best(a: List[Tuple[dict, float]], b: List[Tuple[dict, float]]):
            best: Dict[Any, Tuple[dict, float]] = {}
            for d, s in (a or []) + (b or []):
                k = _dedupe_key(d)
                prev = best.get(k)
                if (prev is None) or (s > prev[1]):
                    best[k] = (d, s)
            return sorted(best.values(), key=lambda x: x[1], reverse=True)

        def _uniq_top(pairs: List[Tuple[dict, float]]) -> List[Tuple[dict, float]]:
            seen: set = set()
            out: List[Tuple[dict, float]] = []
            for d, s in sorted(pairs, key=lambda x: x[1], reverse=True):
                k = _dedupe_key(d)
                if k in seen:
                    continue
                seen.add(k)
                out.append((d, s))
            return out

        # --- split above/below threshold; LLM-verify grey zone if enabled ---
        base_results: List[Tuple[dict, float]] = [(d, min(s, 1.0)) for (d, s) in accepted if s >= self.RAG_SEARCH]

        accepted_below = [(d, s) for (d, s) in accepted if s < self.RAG_SEARCH]
        verify_candidates = _uniq_top(accepted_below + (need_verify or []))

        verified_kept: List[Tuple[dict, float]] = []
        if verify_candidates:
            logger.info("[RAG][LLM_VERIFY] Verifying %d candidates (scores: %s)", 
                       len(verify_candidates), 
                       [f"{s:.3f}" for _, s in verify_candidates[:10]])
            item_type = "doc" if self.sql else "case"
            try:
                # Use parallel verification for faster processing
                vm = self.llmv.verify_many_parallel(current_text, verify_candidates, item_type=item_type, max_workers=5) or []
                verified_kept = [(d, s) for (d, s) in vm if s >= self.RAG_SEARCH]
                logger.info("[RAG][LLM_VERIFY] Verified %d/%d candidates passed (>=%.3f)", 
                           len(verified_kept), len(verify_candidates), self.RAG_SEARCH)
                # Log verification results
                for doc, score in verified_kept[:10]:
                    logger.info("[RAG][LLM_VERIFY][DETAIL] PASSED: title=%r score=%.4f", 
                               doc.get("title", "N/A"), score)
                for doc, score in vm:
                    if score < self.RAG_SEARCH:
                        logger.info("[RAG][LLM_VERIFY][DETAIL] FAILED: title=%r score=%.4f < %.3f", 
                                   doc.get("title", "N/A"), score, self.RAG_SEARCH)
            except Exception as e:
                logger.info("[RAG][MAIN] verify_many_parallel failed: %s", e)

        merged = _merge_keep_best(base_results, verified_kept)
        if not merged:
            return None

        # diagnostics
        try:
            preview_before = [(round(s, 3), (d.get("title") or d.get("case") or d.get("article") or str(d.get("_id"))))
                            for d, s in merged[:10]]
            logger.info("[GATES] before_gap (top 10): %s", preview_before)
        except Exception:
            pass

        # --- optional gap filter (disabled in light path by passing apply_gap=False) ---
        if apply_gap:
            logger.info("applying gap filter (FILTER_GAP=%.3f, top=%.3f)", self.FILTER_GAP, merged[0][1])
            before = len(merged)
            merged = self._gap_filter(merged)
            logger.info("[GATES] after_gap: kept=%d dropped=%d", len(merged), before - len(merged))
        else:
            logger.info("gap filter was not applied")

        # Final threshold enforcement - ensure ALL results >= RAG_SEARCH
        before_threshold = len(merged)
        merged = [(d, s) for (d, s) in merged if s >= self.RAG_SEARCH]
        if before_threshold != len(merged):
            logger.warning("[GATES][THRESHOLD] Filtered %d results below RAG_SEARCH=%.3f (kept %d)", 
                          before_threshold - len(merged), self.RAG_SEARCH, len(merged))
        else:
            logger.info("[GATES][THRESHOLD] All %d results meet RAG_SEARCH=%.3f", len(merged), self.RAG_SEARCH)

        # save only if all results are confident (and no SQL mode)
        if merged and not self.sql:
            confident_results = [(doc, score) for doc, score in merged if score >= self.CONFIDENT]
            if len(confident_results) == len(merged) and not self.debug_mode:
                self._saves_confident_results(current_text, confident_results)

        logger.info("[RAG][PATH] winner=MAIN (merge: base≥thr + verified≥thr)\n%s",
                    self._fmt_pairs(merged))
        return merged[: self.TOP_K_RET]
