# services/rag_dependencies/query_manager.py
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tiktoken
from bson import ObjectId
from pymongo.errors import PyMongoError
from services.rag.config import EMBEDDING_MODEL
from services.rag.rag_dependencies.ai_service import LLM
from services.rag.rag_dependencies.mongo_manager import MongoManager

logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 32000  # Voyage-3-large supports up to 32,000 tokens per embedding request


class QueryManager:
    """
    Generates embeddings, truncates inputs, and handles caching query embeddings in Mongo.

    Notes:
      - We no longer use or migrate any `case_reference`/`case_ids` fields.
      - Cached per-query items are handled via MongoManager helpers.
    """
    def __init__(self, config: Dict[str, Any], query_collection,sql):
        self.config = config
        self.embedding_model = config.get("embedding_model", EMBEDDING_MODEL)
        self.openAI = LLM(config=self.config)
        self.query_collection = query_collection
        self.sql=sql
        if self.sql:
            self.QUERY_SEARCH_THR = 0.9
        else:
            # query_search is fixed at 0.75 for all domains (per config comments)
            self.QUERY_SEARCH_THR = float(self.config.get("thresholds", {}).get("query_search", 0.75))

    # ---------------- Embeddings ----------------
    def truncate_text(self, text: str, max_tokens: int = MAX_TOTAL_TOKENS) -> str:
        try:
            enc = tiktoken.encoding_for_model(self.embedding_model)
        except KeyError:
            # Fallback to cl100k_base if model not recognized (e.g., Voyage models)
            enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text or "")
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
        return enc.decode(toks)

    def get_embedding(self, text: str) -> np.ndarray:
        text = self.truncate_text(text or "")
        vec = self.openAI.get_openai_embedding(text)
        if vec is None:
            return np.zeros(1024, dtype=np.float32)  # safe fallback (Voyage-3-large dimensions)
        return np.array(vec, dtype=np.float32)

    # ---------------- Embeddings ----------------
    def get_or_create_query_embedding(self, query: str, db: MongoManager, previous_rephrases: List) -> Tuple[np.ndarray, bool]:
        """
        Look up by case-insensitive 'query' (no query_norm).
        If exists+has embedding -> return cached.
        Else generate and upsert by 'query'.
        """
        # Normalize query at the start to ensure consistency
        norm_query = db.normalize_query(query)

        # 1) Exact match by 'query' (case-insensitive)
        existing = db.find_query_doc_ci(norm_query)
        if existing and existing.get("embedding") is not None:
            logger.info("[EMB][HIT] %r -> cached", norm_query)
            return np.array(existing["embedding"], dtype=np.float32).ravel(), True

        # 2) If doc exists but no embedding, generate and write
        if existing and existing.get("embedding") is None:
            logger.info("[EMB][MISS] %r -> doc w/o emb; generating", norm_query)
            emb = self.get_embedding(norm_query)
            db.upsert_query_embedding(norm_query, emb, original_query=existing.get("query") or norm_query)
            return emb, True

        logger.info("[EMB][MISS] %r no cached doc", norm_query)

        # 3) Optional semantic fallback using MongoDB Atlas Vector Search when no prior rephrases
        # Fix 3: Store generated embedding to reuse it later (avoid double generation)
        generated_emb = None
        if not previous_rephrases:
            try:
                # Generate embedding for new query (store it for potential reuse)
                generated_emb = self.get_embedding(norm_query)

                # Use MongoDB Atlas Vector Search to find similar queries
                from services.rag.rag_dependencies.vector_search import vector_search_by_filter

                results = vector_search_by_filter(
                    collection=self.query_collection,
                    query_vector=generated_emb,
                    k=1,  # Only need the top match
                    filter_dict=None,  # Search all queries
                    vector_index_name="vector_index",
                    vector_field="embedding"
                )

                if results:
                    best_doc = results[0]
                    best_sim = float(best_doc.get("score", 0.0))

                    logger.info("[EMB][VECTOR] best_sim=%.3f thr=%.3f", best_sim, self.QUERY_SEARCH_THR)
                    if best_sim >= self.QUERY_SEARCH_THR:
                        logger.info("[EMB][SEM] %r ≈ %r (%.3f)", norm_query, best_doc.get("query"), best_sim)
                        # Return the embedding from the similar query
                        emb = best_doc.get("embedding")
                        if emb:
                            return np.array(emb, dtype=np.float32).ravel(), True

            except (PyMongoError, ValueError, TypeError) as e:
                logger.info("[EMB][VECTOR] Vector search fallback failed: %s", e)

        # 4) Generate new embedding and upsert by 'query'
        # Fix 3: Reuse generated embedding instead of generating again
        if generated_emb is not None:
            emb = generated_emb
            logger.debug("[EMB][REUSE] Reusing embedding generated for vector search")
        else:
            emb = self.get_embedding(norm_query)
        db.upsert_query_embedding(norm_query, emb, original_query=norm_query)
        logger.info("[EMB][NEW] %r stored", norm_query)
        return emb, False


    # ---------------- insights/Results (pass-through to MongoManager) ----------------

    def get_query_with_results(self, db: MongoManager, query: str, limit: Optional[int] = None, collection_key: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            # If collection_key not provided, try to get it from config
            if collection_key is None:
                collection_key = self.config.get("collection_key")
            return db.get_query_with_result(query, limit=limit, collection_key=collection_key)
        except Exception as e:
            logger.exception("[QueryMgr] get results failed: %s", e)
            return []

    def update_query_with_results(self, db: MongoManager, query: str, results: Sequence[Any] | None, collection_key: Optional[str] = None) -> None:
        if not results:
            logger.info("[QM][RES] empty results for %r; skip.", query)
            return

        norm = db.normalize_query(query)
        ok = 0
        fail = 0
        logger.info("[QM][RES][BEGIN] raw=%r norm=%r count=%d", self._short(query), norm, len(results))

        for item in results:
            try:
                shape = (type(item).__name__, type(item[0]).__name__, type(item[1]).__name__) if isinstance(item, (tuple, list)) and len(item) >= 2 else type(item).__name__
                logger.debug("[QM][RES][WRITE] shape=%r sample=%r", shape, self._preview_result_item(item))
                db.update_query_with_result(norm, item, collection_key=collection_key)
                ok += 1
            except Exception:
                fail += 1
                logger.exception("[QM][RES][WRITE][ERR] item=%r", item)

        logger.info("[QM][RES][END] wrote=%d failed=%d norm=%r", ok, fail, norm)
    def get_query_with_insights(self, db: MongoManager, query: str, limit: Optional[int] = None, language: str = "en") -> List[Dict[str, Any]]:
            try:
                return db.get_query_with_insight(query, limit=limit, language=language)
            except Exception as e:
                logger.exception("[QueryMgr] get insights failed: %s", e)
                return []

    def update_query_with_insight(
        self,
        db: MongoManager,
        query: str,
        *,
        text: str,
        index: int,
        knowledge_id: Optional[str] = None,  # Deprecated: kept for backward compatibility but not used
        language: str = "en",
        insight_en: Optional[str] = None,
        insight_es: Optional[str] = None,
    ) -> None:
        """
        Store insight by index only (not knowledge_id).
        Insights are aligned with results array by index position.
        """
        try:
            db.update_query_with_insight(
                query,
                text,
                index,
                language=language,
                insight_en=insight_en,
                insight_es=insight_es
            )
        except Exception:
            logger.exception("[QueryMgr] update insight failed")
    # ---------------- Insights/Results checks (no query_norm) ----------------
    def check_query_has_insights(self, db: MongoManager, query: str) -> bool:
        doc = db.find_query_doc_ci(query)
        has = bool(doc and doc.get("insights"))
        logger.info("[QM][CHK][SUM] %r -> %s", query, has)
        return has

    def check_query_has_results(self, db: MongoManager, query: str) -> bool:
        doc = db.find_query_doc_ci(query)
        has = bool(doc and doc.get("results"))
        logger.info("[QM][CHK][RES] %r -> %s", query, has)
        return has


    # --- Small helpers ---
    def _short(self, s: str, limit: int = 120) -> str:
        if s is None:
            return ""
        return s if len(s) <= limit else (s[:limit - 1] + "…")

    def _preview_result_item(self, item: Any):
        try:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                d, s = item[0], item[1]
                kid = (d.get("_id") if isinstance(d, dict) else d)
                return {"kid_tail": str(kid)[-6:], "score": float(s)}
            if isinstance(item, dict):
                return {"kid_tail": str(item.get("_id"))[-6:], "score?": item.get("score")}
            return str(item)[:120]
        except Exception:
            return "<uninspectable>"
    # ---------------- Rephrase/link ----------------
    def update_query_rephrased_ref(self, db, original: str, rephrased: str) -> Optional[ObjectId]:
        """
        Link `original` -> `rephrased` by ObjectId.
        - Do NOT create placeholder docs.
        - If `rephrased` doesn't exist, create it WITH an embedding first.
        - If `original` doesn't exist, we bail (caller should have created it when embedding).
        """
        try:
            orig_doc = db.find_query_doc_ci(original)
            if not orig_doc:
                logger.warning("[QM][REPHRASE] original missing; not linking: %r", original)
                return None

            reph_doc = db.find_query_doc_ci(rephrased)
            if reph_doc:
                reph_id = reph_doc["_id"]
            else:
                # create a proper document WITH embedding
                emb = self.get_embedding(rephrased)
                reph_id = db.upsert_query_embedding_and_get_id(rephrased, emb)

            if orig_doc["_id"] == reph_id:
                logger.info("[QM][REPHRASE] skip self-link for %r", original)
                return reph_id

            db.link_rephrased_id(orig_doc["_id"], reph_id)
            logger.info("[QM][REPHRASE] linked '%s' -> '%s' (_id=%s)", original, rephrased, str(reph_id))
            return reph_id
        except Exception:
            logger.exception("[QM][REPHRASE][LINK] failed: %r -> %r", original, rephrased)
            return None

    def get_query_with_rephrase(self, db, current_text: str) -> Optional[str]:
        """
        Return the target query string if `current_text` has a rephrased_ref (ObjectId).
        """
        try:
            doc = db.find_query_doc_ci(current_text)
            if not doc:
                return None
            target_id = doc.get("rephrased_ref")
            if not target_id:
                return None
            tgt = db.find_query_doc_by_id(target_id)
            return (tgt or {}).get("query")
        except Exception:
            logger.exception("[QM][REPHRASE][GET] failed for %r", current_text)
            return None

    def check_query_has_update_reference(self, db, current_text: str) -> bool:
        try:
            doc = db.find_query_doc_ci(current_text)
            return bool(doc and doc.get("rephrased_ref"))
        except Exception:
            logger.exception("[QM][REPHRASE][CHK] failed for %r", current_text)
            return False


    def search_similar(
        self,
        db,  # MongoManager
        cleaned_query: str,
        vec,
        titles: List[Any],  # accepts List[str] OR List[dict]
    ) -> List[tuple[Dict[str, float]]]:
        """
        Search similar cases using MongoDB Atlas Vector Search.

        Returns: List of 1-tuples where each tuple is a dict mapping {title: score_float}
                e.g., [ ({"CASE-2025-0001": 0.873},), ({"CASE-2025-0007": 0.811},) ]

        Note: This now uses MongoDB Atlas Vector Search instead of brute-force cosine similarity.
        """

        # --- validate query embedding ---
        if vec is None:
            logger.info("[QM][find] no embedding returned for %r; skipping", getattr(self, "_short", lambda s: s)(cleaned_query))
            return []

        # --- normalize mixed input: docs vs titles ---
        want_titles: List[str] = []
        for it in (titles or []):
            if isinstance(it, dict):
                title = (it.get("title") or it.get("case_id") or "").strip()
                if title:
                    want_titles.append(title)
            elif isinstance(it, str):
                s = it.strip()
                if s:
                    want_titles.append(s)

        if not want_titles:
            logger.info("[QM][find] no titles provided; nothing to search.")
            return []

        try:
            # Use MongoDB Atlas Vector Search with title filter
            # Build filter to only search within the specified titles
            title_filter = {"title": {"$in": want_titles}}

            # Perform vector search on the cases collection
            from services.rag.rag_dependencies.vector_search import vector_search_by_filter

            # Get the collection - try main or cases depending on context
            collection = getattr(db, 'cases', None) or getattr(db, 'main', None)
            if collection is None:
                logger.warning("[QM][find] No collection available for vector search")
                return []

            # Execute vector search with filter
            results = vector_search_by_filter(
                collection=collection,
                query_vector=vec,
                k=len(want_titles),  # Return up to all requested titles
                filter_dict=title_filter,
                vector_index_name="vector_index",
                vector_field="embedding"
            )

            # Convert results to expected format
            pairs: List[Tuple[str, float]] = []
            for doc in results:
                title = (doc.get("title") or doc.get("case_id") or "").strip()
                score = float(doc.get("score", 0.0))
                if title:
                    pairs.append((title, score))

            if pairs:
                scores = np.array([s for (_t, s) in pairs], dtype=float)
                logger.info(
                    "[QM][find][VECTOR] cases=%d matched=%d score[min=%.3f max=%.3f mean=%.3f std=%.3f]",
                    len(want_titles), len(pairs),
                    float(scores.min()), float(scores.max()), float(scores.mean()),
                    float(scores.std()) if scores.size > 1 else 0.0
                )
            else:
                logger.info("[QM][find][VECTOR] no matches (cases=%d)", len(want_titles))

            # --- final shape: List[tuple[Dict[str, float]]]
            out: List[tuple[Dict[str, float]]] = [({title: score},) for title, score in pairs]
            return out

        except Exception as e:
            logger.exception("[QM][find][VECTOR] Vector search failed: %s", e)
            # Fallback: return empty results
            return []

    # ============ CLIENT CASE CACHING (High-Confidence Results) ============

    def store_case_query_pairs(
        self,
        db: MongoManager,
        query: str,
        case_results: List[Tuple[dict, float]],
        searched_case_ids: Optional[List[str]] = None,
        min_score: float = 0.90
    ) -> None:
        """
        Store high-confidence case results (score >= min_score) as cached pairs for the query.
        Also stores the ObjectIds of cases that were searched (the "search range").

        Args:
            db: MongoManager instance
            query: User query text
            case_results: List of (case_dict, score) tuples
            searched_case_ids: List of MongoDB ObjectIds (as strings) of all cases searched
            min_score: Minimum score threshold to cache (default 0.90)
        """
        # Normalize query and validate
        if not query or not query.strip():
            logger.warning("[QM][CASE_CACHE] Skipping storage - empty query")
            return

        norm = db.normalize_query(query)

        # Ensure the query document exists with the query field set (required for unique index)
        # This prevents duplicate key errors when query field is null
        # IMPORTANT: _ensure_query_doc will find and update existing documents from track_query_usage
        db._ensure_query_doc(norm)

        # Always store the search range first (even if no results)
        # This enables incremental search on subsequent queries
        # CRITICAL: Use $set to preserve ALL existing fields from track_query_usage (avg_relevance_score, language, searched_datetime, etc.)
        # $set only updates specified fields and preserves all others - this ensures proper merging
        if searched_case_ids:
            try:
                # Check if document exists (may have been created by track_query_usage)
                existing_doc = db.query.find_one({"query": norm})
                if existing_doc:
                    has_analytics = any(key in existing_doc for key in ["avg_relevance_score", "language", "searched_datetime"])
                    if has_analytics:
                        logger.info("[QM][CASE_CACHE][RANGE] Found existing document with analytics fields - will merge cache fields")

                # Use query (normalized) as filter to match _ensure_query_doc behavior
                # This ensures we're updating the same document that was created
                # Using $set preserves all existing fields (from track_query_usage) while adding incremental cache fields
                result = db.query.update_one(
                    {"query": norm},  # Use "query" field (normalized) as filter, not "query_norm"
                    {
                        "$set": {
                            "query_norm": norm,  # Also set query_norm for backward compatibility
                            "searched_case_ids": searched_case_ids,
                            "search_range_size": len(searched_case_ids),
                            "last_search_at": datetime.datetime.utcnow(),
                            "updated_at": datetime.datetime.utcnow()  # Always update timestamp
                        }
                    },
                    upsert=False  # Don't upsert - document should already exist from _ensure_query_doc or track_query_usage
                )

                # Verify merge succeeded
                if result.modified_count > 0:
                    verify_doc = db.query.find_one({"query": norm})
                    if verify_doc:
                        has_analytics_after = any(key in verify_doc for key in ["avg_relevance_score", "language", "searched_datetime"])
                        has_cache_after = "searched_case_ids" in verify_doc
                        if has_analytics_after and has_cache_after:
                            logger.info("[QM][CASE_CACHE][RANGE] ✅ Verified merge: document has both analytics and cache fields")
                        elif has_cache_after:
                            logger.debug("[QM][CASE_CACHE][RANGE] Document has cache fields (analytics may be added later)")
                elif existing_doc:
                    logger.warning("[QM][CASE_CACHE][RANGE] Document exists but update had no effect - this may indicate a merge issue")
                logger.info("[QM][CASE_CACHE][RANGE] Stored search range: %d case IDs for query: %r",
                           len(searched_case_ids), query)
            except PyMongoError:
                logger.exception("[QM][CASE_CACHE][RANGE] Failed to store search range for query: %r", query)

        # Store high-confidence results if any
        if not case_results:
            logger.info("[QM][CASE_CACHE] No results to cache, but search range stored")
            return

        high_conf_cases = [(case, score) for case, score in case_results if score >= min_score]

        if not high_conf_cases:
            logger.info("[QM][CASE_CACHE] No high-confidence results (>= %.2f) to cache for query: %r, but search range stored", min_score, query)
            return

        # Store each high-confidence case as a result
        # Get collection_key from config to identify which collection these results belong to
        collection_key = self.config.get("collection_key")
        stored_count = 0
        for case_doc, score in high_conf_cases:
            try:
                case_id = str(case_doc.get("_id", "N/A"))
                case_title = case_doc.get("title", "N/A")
                db.update_query_with_result(norm, (case_doc, score), collection_key=collection_key)
                stored_count += 1
                logger.info("[QM][CASE_CACHE][RESULT] Stored: %s (score: %.3f, id: %s)",
                           case_title[:50], score, case_id[:12])
            except PyMongoError as e:
                logger.exception("[QM][CASE_CACHE] Failed to cache case %s for query: %r - %s",
                               case_doc.get("title", "Unknown"), query, e)

        logger.info("[QM][CASE_CACHE] Successfully stored %d/%d high-confidence cases + search range for query: %r",
                   stored_count, len(high_conf_cases), query)

    def find_cached_similar_query(self, db: MongoManager, query: str, collection) -> Optional[Dict[str, Any]]:
        """
        Find a similar cached query using vector search on the query collection.
        Uses the query_search threshold from config.

        Args:
            db: MongoManager instance
            query: User query text
            collection: MongoDB collection to search

        Returns:
            Cached query document if found, None otherwise
        """
        try:
            # Generate embedding for the query
            vec = self.get_embedding(query)
            if vec is None:
                return None

            # Use vector search to find similar queries
            from services.rag.rag_dependencies.vector_search import vector_search_by_filter

            results = vector_search_by_filter(
                collection=collection,
                query_vector=vec,
                k=1,  # Number of results to return
                filter_dict=None,  # No pre-filter
                vector_index_name="vector_index",  # Correct parameter name
                vector_field="embedding"
            )

            if not results:
                return None

            # Check if similarity meets query_search threshold
            top_result = results[0]
            score = top_result.get("score", 0.0)

            # FIX: Increase threshold to 0.85 to prevent unrelated queries from returning cached results
            # Also require minimum similarity to ensure queries are actually related
            min_similarity_threshold = max(self.QUERY_SEARCH_THR, 0.85)

            if score >= min_similarity_threshold:
                # Fetch full document by _id to get all fields including query and query_norm
                doc_id = top_result.get("_id")
                if doc_id:
                    full_doc = collection.find_one({"_id": doc_id})
                    if full_doc:
                        # Add score to full document for consistency
                        full_doc["score"] = score
                        similar_query = full_doc.get("query") or full_doc.get("query_norm") or "unknown"
                        logger.info("[QM][SIMILAR_QUERY] Found similar cached query (score=%.3f): %r -> %r",
                                   score, query, similar_query)
                        return full_doc
                    else:
                        logger.warning("[QM][SIMILAR_QUERY] Found similar query by vector search but document not found by _id: %s", doc_id)
                        # Fallback to partial result
                        similar_query = top_result.get("query") or top_result.get("query_norm") or "unknown"
                        logger.info("[QM][SIMILAR_QUERY] Found similar cached query (score=%.3f): %r -> %r",
                                   score, query, similar_query)
                        return top_result
                else:
                    # No _id in result, use partial result
                    similar_query = top_result.get("query") or top_result.get("query_norm") or "unknown"
                    logger.info("[QM][SIMILAR_QUERY] Found similar cached query (score=%.3f): %r -> %r",
                               score, query, similar_query)
                    return top_result
            else:
                logger.info("[QM][SIMILAR_QUERY] No similar query above threshold (%.3f < %.3f)",
                           score, min_similarity_threshold)
                return None

        except Exception as e:
            logger.exception("[QM][SIMILAR_QUERY] Failed to find similar query: %s", e)
            return None

    def get_cached_case_titles(self, cached_query_doc: Dict[str, Any]) -> List[str]:
        """
        Extract case titles from a cached query document.

        Args:
            cached_query_doc: Query document from MongoDB

        Returns:
            List of case titles
        """
        case_titles = []
        results = cached_query_doc.get("results", [])

        for result in results:
            if isinstance(result, dict):
                title = result.get("title")
                if title:
                    case_titles.append(title)

        logger.info("[QM][CACHED_TITLES] Extracted %d case titles from cached query", len(case_titles))
        return case_titles

    def get_cached_search_range(self, cached_query_doc: Dict[str, Any]) -> List[str]:
        """
        Extract the search range (ObjectIds) from a cached query document.

        Args:
            cached_query_doc: Query document from MongoDB

        Returns:
            List of ObjectIds (as strings) that were previously searched
        """
        search_range = cached_query_doc.get("searched_case_ids", [])
        logger.info("[QM][CACHED_RANGE] Extracted search range: %d case IDs", len(search_range))
        return search_range

    def identify_new_cases(
        self,
        current_case_ids: List[str],
        cached_case_ids: List[str]
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Identify new, removed, and existing cases between current and cached ranges.

        Args:
            current_case_ids: ObjectIds of current visible cases
            cached_case_ids: ObjectIds from previous search

        Returns:
            Tuple of (new_case_ids, removed_case_ids, existing_case_ids)
        """
        current_set = set(current_case_ids)
        cached_set = set(cached_case_ids)

        new_ids = list(current_set - cached_set)  # In current but not in cache
        removed_ids = list(cached_set - current_set)  # In cache but not in current
        existing_ids = list(current_set & cached_set)  # In both

        logger.info("[QM][RANGE_DIFF] Current: %d, Cached: %d | New: %d, Removed: %d, Existing: %d",
                   len(current_case_ids), len(cached_case_ids),
                   len(new_ids), len(removed_ids), len(existing_ids))

        return new_ids, removed_ids, existing_ids
