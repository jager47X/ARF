# services/rag_dependencies/mongo_manager.py
from __future__ import annotations
import logging
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from bson import ObjectId
import numpy as np
from pymongo import MongoClient
from pymongo import ASCENDING, TEXT, errors, ReturnDocument
from services.rag.config import MONGO_URI
from pymongo.collation import Collation
from pymongo.errors import OperationFailure, NetworkTimeout

logger = logging.getLogger(__name__)

# Timezone setup for US-west (California)
try:
    from zoneinfo import ZoneInfo
    US_WEST_TZ = ZoneInfo("America/Los_Angeles")
except ImportError:
    # Fallback for Python < 3.9
    try:
        import pytz
        US_WEST_TZ = pytz.timezone("America/Los_Angeles")
    except ImportError:
        # No timezone support - use UTC
        US_WEST_TZ = None
        logger.warning("No timezone library available. Using UTC instead of US-west timezone.")
_QUERY_COLLATION = Collation(locale="en", strength=2)  # case-insensitive


class MongoManager:
    """
    Owns Mongo connections/collections and a few convenience helpers for queries.

    Design decisions:
      - No `query_norm` field: we store the **normalized string** in `query`.
      - All reads/writes for queries should pass `normalize_query(query)` first.
      - `rephrased_ref` stores an ObjectId pointing to another query document.
      - We never create placeholder docs for rephrase targets here; callers that
        need to create a rephrased doc should do so WITH an embedding (see
        `upsert_query_embedding_and_get_id`).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get TLS configuration if available
        tls_config = {}
        try:
            # Try importing from db.database_ssl (for local/dev)
            try:
                from db.database_ssl import get_mongodb_tls_config
                tls_config = get_mongodb_tls_config()
            except ImportError:
                # Try importing from backend.db.database_ssl (for Docker/production)
                try:
                    from backend.db.database_ssl import get_mongodb_tls_config
                    tls_config = get_mongodb_tls_config()
                except ImportError:
                    pass
            
            if tls_config:
                logger.info("MongoDB TLS configuration loaded for MongoManager")
        except Exception as e:
            logger.warning(f"Failed to load MongoDB TLS configuration: {e}")
        
        # MongoDB Atlas (mongodb+srv://) requires TLS - enable it if not already configured
        if MONGO_URI and "mongodb+srv://" in MONGO_URI:
            if not tls_config or "tls" not in tls_config:
                tls_config = tls_config.copy() if tls_config else {}
                tls_config["tls"] = True
                logger.info("MongoDB Atlas detected (mongodb+srv), enabling TLS")
        
        # If no TLS config and not using mongodb+srv, log a warning
        if not tls_config and (not MONGO_URI or "mongodb+srv://" not in MONGO_URI):
            logger.warning("MongoDB TLS not configured - connections may fail if server requires TLS")
        
        self._client = MongoClient(
            MONGO_URI,
            maxPoolSize=50, minPoolSize=5,
            connectTimeoutMS=30000,  # Increased from 5000ms to 30s for large collections
            socketTimeoutMS=300000,  # Increased from 10000ms to 5min for long-running vector searches on large collections
            serverSelectionTimeoutMS=30000,  # Increased from 5000ms to 30s
            **tls_config
        )
        self.db = self._client[config["db_name"]]
        self.query = self.db[config["query_collection_name"]]
        self.main = self.db[config["main_collection_name"]]
        # Only set cases collection if sql_attached is False AND cases_collection_name exists in config
        if not config.get("sql_attached", False) and "cases_collection_name" in config:
            self.cases = self.db[config["cases_collection_name"]]
        self.ensure_indexes()

    # ----------------- helpers -----------------
    @staticmethod
    def normalize_query(q: str) -> str:
        import re
        return re.sub(r"\s+", " ", (q or "").strip().lower())

    @staticmethod
    def normalize_summary_text(s: str) -> str:
        import re
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    
    @staticmethod
    def translate_text(text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language using OpenAI.
        
        Args:
            text: The text to translate
            source_lang: Source language code ('en' or 'es')
            target_lang: Target language code ('en' or 'es')
            
        Returns:
            Translated text or None if translation fails
        """
        if not text or source_lang == target_lang:
            return text
            
        try:
            from services.rag.rag_dependencies.ai_service import translate_insight
            return translate_insight(text, source_lang, target_lang)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None
    def _dedupe_user_queries_basic(self) -> None:
        """
        For each duplicated 'query' string, keep the newest doc by _id
        and delete all other duplicates. No other fields considered.
        """
        pipeline = [
            {"$group": {"_id": "$query", "ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}}
        ]
        dup_groups = list(self.query.aggregate(pipeline))
        if not dup_groups:
            logger.info("[DEDUPE] no duplicates found")
            return

        total_removed = 0
        for g in dup_groups:
            ids = g["ids"]
            # newest first by ObjectId timestamp
            ids_sorted = sorted(ids, key=lambda x: x.generation_time if isinstance(x, ObjectId) else 0, reverse=True)
            keep_id = ids_sorted[0]
            remove_ids = ids_sorted[1:]
            if remove_ids:
                self.query.delete_many({"_id": {"$in": remove_ids}})
                total_removed += len(remove_ids)
                logger.info("[DEDUPE] kept=%s removed=%d query='%s'", keep_id, len(remove_ids), g["_id"])

        logger.info("[DEDUPE] removed %d duplicate docs", total_removed)
    def find_query_doc_ci(self, query_str: str, projection: Optional[Dict[str, int]] = None) -> Optional[dict]:
        """
        Look up by normalized equality on `query`. For legacy docs, fall back to CI equality.
        
        Args:
            query_str: Query string to look up
            projection: Optional MongoDB projection dict (e.g., {"results": 1} to only fetch results field)
        """
        norm = self.normalize_query(query_str)
        # Primary: exact match on normalized `query`
        doc = self.query.find_one({"query": norm}, projection=projection)
        if doc:
            return doc
        # Legacy/compat: try CI equality on raw input
        raw = (query_str or "").strip()
        if not raw:
            return None
        return self.query.find_one({"query": raw}, projection=projection, collation=_QUERY_COLLATION)

    def upsert_query_embedding(self, normalized_query: str, embedding: Union[List[float], np.ndarray], original_query: Optional[str] = None) -> None:
        """
        Upsert embedding for a normalized query.
        NOTE: `query` is always stored as the **normalized** string.
        """
        arr = np.asarray(embedding, dtype=float).ravel().tolist()
        now = datetime.datetime.utcnow()
        self.query.update_one(
            {"query": normalized_query},
            {
                "$set": {
                    "query": normalized_query,
                    "embedding": arr,
                    "updated_at": now
                }
            },
            upsert=True
        )

    def upsert_query_embedding_and_get_id(
        self,
        query_str: str,
        embedding: Union[List[float], np.ndarray]
    ) -> ObjectId:
        """
        Upsert a document WITH `query` (normalized) and `embedding`, returning its _id.
        Intended for creating rephrased targets WITH an embedding before linking.
        """
        norm = self.normalize_query(query_str)
        arr = np.asarray(embedding, dtype=float).ravel().tolist()
        now = datetime.datetime.utcnow()

        doc = self.query.find_one_and_update(
            {"query": norm},
            {
                "$set": {"query": norm, "embedding": arr, "updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc["_id"]

    def _ensure_query_doc(self, normalized_query: str, language: Optional[str] = None) -> None:
        """Create the base query doc if missing (no-op if exists).
        Always ensures the 'query' field is set (required for unique index).
        Updates existing documents created by track_query_usage instead of creating duplicates.
        """
        now = datetime.datetime.utcnow()
        # Use $set for query field to ensure it's always set (even if document exists from track_query_usage)
        # Use $setOnInsert only for created_at (should only be set on creation)
        update_fields = {
            "$set": {"query": normalized_query, "updated_at": now},  # Always set query field
            "$setOnInsert": {"created_at": now}  # Only set created_at on insert
        }
        if language:
            update_fields["$set"]["language"] = language
        self.query.update_one(
            {"query": normalized_query},
            update_fields,
            upsert=True
        )

    def ensure_indexes(self):
        def create_or_replace(coll, keys, name, **opts):
            try:
                # Fast check: if index already exists, skip creation
                existing_indexes = coll.index_information()
                if name in existing_indexes:
                    logger.info("[IDX] %s already exists, skipping", name)
                    return
                
                # Create index with background=True for non-blocking creation
                # This prevents timeouts on large collections - MongoDB builds index in background
                index_opts = opts.copy()
                index_opts['background'] = True  # Non-blocking - doesn't lock collection
                
                try:
                    # Use a separate client with longer timeout for index creation
                    # But with background=True, this should return quickly
                    coll.create_index(keys, name=name, **index_opts)
                    logger.info("[IDX] %s creation started (background mode - non-blocking)", name)
                except NetworkTimeout:
                    # Index creation command timed out, but background process may still succeed
                    logger.warning(
                        "[IDX] %s creation command timed out, but background build may continue. "
                        "Startup continuing...", name
                    )
                    # Don't raise - allow startup to continue
                    return
                except (OperationFailure, Exception) as idx_err:
                    # Check if it's a "duplicate key" or "already exists" error (index might have been created)
                    error_msg = str(idx_err).lower()
                    if 'already exists' in error_msg or 'duplicate' in error_msg or getattr(idx_err, 'code', None) == 85:
                        logger.info("[IDX] %s already exists (detected via error)", name)
                        return
                    # Log but don't fail startup for other index creation errors
                    logger.warning(
                        "[IDX] %s creation had issue (may still complete in background): %s", 
                        name, idx_err
                    )
                    return
                    
            except OperationFailure as e:
                if getattr(e, "code", None) == 85:  # IndexOptionsConflict
                    logger.warning("[IDX] %s options changed; checking for conflicting indexes", name)
                    
                    # Check if we're creating a text index
                    is_text_index = False
                    if isinstance(keys, list):
                        is_text_index = any(isinstance(k, tuple) and len(k) > 1 and k[1] == TEXT for k in keys) or any(k == TEXT for k in keys)
                    elif isinstance(keys, tuple) and len(keys) > 1:
                        is_text_index = keys[1] == TEXT
                    
                    # Get all existing indexes
                    existing_indexes = coll.index_information()
                    conflicting_indexes = []
                    
                    if is_text_index:
                        # For text indexes, MongoDB only allows ONE text index per collection
                        # So we need to drop ALL existing text indexes
                        for idx_name, idx_info in existing_indexes.items():
                            if idx_name == "_id_" or idx_name == name:
                                continue
                            # Check if this is a text index by looking at the key structure
                            # Text indexes in MongoDB have keys like [("field", "text"), ...]
                            idx_keys = idx_info.get('key', [])
                            is_existing_text = False
                            
                            if isinstance(idx_keys, list):
                                # Check if any key uses TEXT type (can be "text" string or TEXT constant)
                                for k in idx_keys:
                                    if isinstance(k, tuple) and len(k) > 1:
                                        # Check if second element is TEXT (can be string "text" or constant)
                                        key_type = k[1]
                                        if str(key_type) == "text" or key_type == TEXT:
                                            is_existing_text = True
                                            break
                                    elif isinstance(k, str) and k == "text":
                                        is_existing_text = True
                                        break
                            elif isinstance(idx_keys, tuple) and len(idx_keys) > 1:
                                if str(idx_keys[1]) == "text" or idx_keys[1] == TEXT:
                                    is_existing_text = True
                            
                            # Also check for text index indicators in the index info
                            if not is_existing_text:
                                # Check if index has textIndexVersion (indicates text index)
                                if 'textIndexVersion' in idx_info:
                                    is_existing_text = True
                            
                            if is_existing_text:
                                conflicting_indexes.append(idx_name)
                    else:
                        # For non-text indexes, use the original logic
                        def normalize_keys(ks):
                            if isinstance(ks, list):
                                return tuple(sorted((k[0] if isinstance(k, tuple) else k, k[1] if isinstance(k, tuple) and len(k) > 1 else 1) for k in ks))
                            elif isinstance(ks, tuple):
                                return (ks[0] if isinstance(ks[0], str) else str(ks[0]), ks[1] if len(ks) > 1 else 1)
                            else:
                                return (ks, 1)
                        
                        target_keys = normalize_keys(keys)
                        
                        for idx_name, idx_info in existing_indexes.items():
                            if idx_name == "_id_":
                                continue
                            # Get the key definition from the index
                            idx_keys = idx_info.get('key', [])
                            normalized_idx_keys = normalize_keys(idx_keys)
                            
                            # Check if this index conflicts (same fields)
                            if normalized_idx_keys == target_keys:
                                conflicting_indexes.append(idx_name)
                    
                    # Drop all conflicting indexes
                    for idx_name in conflicting_indexes:
                        try:
                            logger.info("[IDX] Dropping conflicting index: %s", idx_name)
                            coll.drop_index(idx_name)
                        except Exception as drop_err:
                            logger.warning("[IDX] Failed to drop conflicting index %s: %s", idx_name, drop_err)
                    
                    # Now try to create the index again
                    try:
                        coll.create_index(keys, name=name, **opts)
                        logger.info("[IDX] %s recreated after dropping conflicts", name)
                    except Exception as recreate_err:
                        logger.exception("[IDX] %s recreate failed after dropping conflicts", name)
                        raise
                        
                elif getattr(e, "code", None) == 11000:  # DuplicateKey
                    logger.error("[IDX] %s duplicate keys; de-duplicate data then retry", name)
                    raise
                else:
                    logger.exception("[IDX] %s create failed", name)
                    raise

        # (cleanup) drop legacy query_norm index if it exists
        try:
            self.query.drop_index("query_norm_idx")
            logger.info("[IDX] dropped legacy query_norm_idx")
        except Exception:
            pass
        
        # Also drop the old query_text_idx if it exists (from init_empty_collection.py)
        try:
            self.query.drop_index("query_text_idx")
            logger.info("[IDX] dropped legacy query_text_idx")
        except Exception:
            pass
        
        # Drop legacy content_text_idx from init_empty_collection.py if it exists
        # This conflicts with metadata_text_index since MongoDB only allows one text index per collection
        try:
            existing_indexes = self.main.index_information()
            if "content_text_idx" in existing_indexes:
                logger.info("[IDX] Found legacy content_text_idx, dropping...")
                self.main.drop_index("content_text_idx")
                logger.info("[IDX] Successfully dropped legacy content_text_idx")
            else:
                logger.debug("[IDX] content_text_idx does not exist, skipping drop")
        except Exception as e:
            logger.warning("[IDX] Failed to drop legacy content_text_idx: %s", e)
            # Don't pass silently - log the error but continue

        # Proactively check for and drop any existing text indexes on main collection
        # MongoDB only allows one text index per collection
        try:
            existing_indexes = self.main.index_information()
            text_indexes_found = []
            for idx_name, idx_info in existing_indexes.items():
                if idx_name == "_id_":
                    continue
                # Check if this is a text index
                if 'textIndexVersion' in idx_info:
                    text_indexes_found.append(idx_name)
                else:
                    # Also check key structure for text indexes
                    idx_keys = idx_info.get('key', [])
                    if isinstance(idx_keys, list):
                        for k in idx_keys:
                            if isinstance(k, tuple) and len(k) > 1:
                                if str(k[1]) == "text" or k[1] == TEXT:
                                    text_indexes_found.append(idx_name)
                                    break
            
            # Drop all found text indexes (except the one we're about to create)
            for idx_name in text_indexes_found:
                if idx_name != "metadata_text_index":
                    try:
                        logger.info("[IDX] Dropping existing text index: %s", idx_name)
                        self.main.drop_index(idx_name)
                        logger.info("[IDX] Successfully dropped text index: %s", idx_name)
                    except Exception as e:
                        logger.warning("[IDX] Failed to drop text index %s: %s", idx_name, e)
        except Exception as e:
            logger.warning("[IDX] Error checking for existing text indexes: %s", e)

        # ---------- main collection ----------
        # Create indexes in parallel for faster startup
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        main_indexes = [
            (self.main, [("article", TEXT), ("section", TEXT), ("title", TEXT)], "metadata_text_index", {
                "default_language": "none",
                "language_override": "language",
                "weights": {"article": 1, "section": 1, "title": 1}
            }),
            (self.main, [("article", ASCENDING)], "article_idx", {}),
            (self.main, [("title", ASCENDING)], "title_idx", {}),
            (self.main, [("section", ASCENDING)], "section_idx", {}),
        ]
        
        # Create main collection indexes in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(create_or_replace, coll, keys, name, **opts): (name, coll)
                for coll, keys, name, opts in main_indexes
            }
            for future in as_completed(futures):
                name, coll = futures[future]
                try:
                    future.result()  # Check for exceptions
                except Exception as e:
                    logger.warning("[IDX] Parallel creation of %s had issue: %s", name, e)
        
        # Pre-warm indexes by running a quick query (loads index into MongoDB's cache)
        try:
            # These queries will load indexes into MongoDB's WiredTiger cache
            self.main.find_one({"article": {"$exists": True}}, {"article": 1, "_id": 0})
            self.main.find_one({"title": {"$exists": True}}, {"title": 1, "_id": 0})
            self.main.find_one({"section": {"$exists": True}}, {"section": 1, "_id": 0})
            logger.info("[IDX] Pre-warmed indexes in MongoDB cache")
        except Exception as e:
            logger.debug("[IDX] Pre-warming had minor issue (non-critical): %s", e)

        # ---------- query collection ----------
        create_or_replace(self.query, [("references", ASCENDING)], "references_idx")
        create_or_replace(self.query, [("summaries.text_norm", ASCENDING)], "summaries_textnorm_idx")

        # Text index on `query`
        create_or_replace(
            self.query,
            [("query", TEXT)],
            name="metadata_query_index",
            default_language="none",
            language_override="language",
            weights={"query": 1},
        )

        # Unique index on normalized `query`
        create_or_replace(
            self.query,
            [("query", ASCENDING)],
            name="uniq_query",
            unique=True,
        )

    # ================== NEW SUMMARY / REFERENCE API (separated) ==================
    def update_query_with_insight(
        self,
        query: str,
        insight: str,
        index: int,
        knowledge_id: Optional[str | ObjectId] = None,  # Deprecated: kept for backward compatibility but not stored
        language: str = "en",
        insight_en: Optional[str] = None,
        insight_es: Optional[str] = None,
    ) -> None:
        """
        Store insight by index only (not knowledge_id).
        Insights are aligned with results array by index position.
        Validates that index is within the saved results array length.
        """
        norm = self.normalize_query(query)
        self._ensure_query_doc(norm, language=language)
        
        # Validate index against saved results array
        # Since we only generate insights from MongoDB results, this should never happen
        # But we validate as a safety check
        doc = self.find_query_doc_ci(norm, projection={"results": 1})
        if doc:
            results = doc.get("results", [])
            max_index = len(results) - 1
            if index > max_index:
                logger.warning(
                    f"[INSIGHT][SAVE] Skipping insight save - index {index} is beyond results array length ({len(results)}). "
                    f"Max valid index: {max_index}. Insights should only be generated from MongoDB results."
                )
                return  # Don't save insights for invalid indices
        
        now = datetime.datetime.utcnow()

        row = {
            "index": int(index),
            "text": (insight or "").strip(),  # Legacy field - uses current language
            "text_norm": self.normalize_summary_text(insight or ""),
            "created_at": now,
            "updated_at": now,
        }
        
        # Add bilingual fields with normalized versions (NO automatic translation)
        if insight_en:
            row["text_en"] = insight_en.strip()
            row["text_en_norm"] = self.normalize_summary_text(insight_en)
        elif language == "en":
            row["text_en"] = (insight or "").strip()
            row["text_en_norm"] = self.normalize_summary_text(insight or "")
            # Note: Auto-translation removed - only store what is explicitly provided
            
        if insight_es:
            row["text_es"] = insight_es.strip()
            row["text_es_norm"] = self.normalize_summary_text(insight_es)
        elif language == "es":
            row["text_es"] = (insight or "").strip()
            row["text_es_norm"] = self.normalize_summary_text(insight or "")
            # Note: Auto-translation removed - only store what is explicitly provided
        
        # Match insights by index only (not knowledge_id)
        pull_filter = {"index": int(index)}

        self.query.update_one({"query": norm}, {"$pull": {"insights": pull_filter}})
        self.query.update_one(
            {"query": norm},
            {"$push": {"insights": row}, "$set": {"updated_at": now}}
        )

    def get_query_with_insight(self, query: str, limit: Optional[int] = None, language: str = "en") -> List[Dict[str, Any]]:
        """
        Get cached insights for a query, returning the appropriate language version.
        
        Args:
            query: The search query
            limit: Maximum number of insights to return
            language: Language code ('en' or 'es') to determine which text field to use
            
        Returns:
            List of insight objects with 'text' field set to the requested language version
        
        Optimized: Uses projection to only fetch the 'insights' field from MongoDB.
        """
        norm = self.normalize_query(query)
        # Use projection to only fetch the insights field - much faster for large documents
        doc = self.find_query_doc_ci(norm, projection={"insights": 1, "_id": 0})
        lst = (doc or {}).get("insights", []) or []
        lst.sort(key=lambda x: (int(x.get("index", 1_000_000)), x.get("updated_at") or datetime.datetime.min), reverse=False)
        
        # Map language-specific text to the 'text' field for each insight
        for insight in lst:
            if language == "es" and "text_es" in insight and insight["text_es"]:
                insight["text"] = insight["text_es"]
            elif language == "en" and "text_en" in insight and insight["text_en"]:
                insight["text"] = insight["text_en"]
            # else: use existing 'text' field (legacy or fallback)
        
        return lst[:limit] if limit else lst

    def add_references(self, query: str, refs: Union[List[Union[str, ObjectId]], Union[str, ObjectId]], language: Optional[str] = None) -> None:
        """
        Add one or many knowledge _ids into 'references' array (deduped).
        """
        norm_q = self.normalize_query(query)
        self._ensure_query_doc(norm_q, language=language)

        now = datetime.datetime.utcnow()
        ref_list = refs if isinstance(refs, list) else [refs]

        for r in ref_list:
            try:
                rid = ObjectId(r) if isinstance(r, str) else r
                self.query.update_one(
                    {"query": norm_q},
                    {"$addToSet": {"references": rid}, "$set": {"updated_at": now}}
                )
            except Exception:
                logger.warning("Invalid reference id skipped: %r", r)

    def get_references(self, query: str) -> List[ObjectId]:
        norm_q = self.normalize_query(query)
        # Use projection to only fetch the references field
        doc = self.find_query_doc_ci(norm_q, projection={"references": 1, "_id": 0})
        return (doc or {}).get("references", []) or []

    # ================== RESULTS API ==================
    def update_query_with_result(self, query: str, result: Any, language: Optional[str] = None, collection_key: Optional[str] = None) -> None:
        """
        Append a result record for a query.

        We persist ONLY:
        - knowledge_id (str of ObjectId from MAIN)
        - collection_key (string identifier for the collection/domain this knowledge_id belongs to)
        - title        (string label of the main doc)
        - cases        (only those whose from_ref == title)

        Accepted inputs:
        - (dict, score) / (ObjectId, score)
        - dict
        - ObjectId
        - str (title only; no knowledge_id)
        
        Args:
            collection_key: Optional collection key (e.g., "US_CONSTITUTION_SET", "US_CODE_SET") 
                          to identify which collection the knowledge_id refers to. Required when 
                          knowledge_id is present to prevent cross-domain collisions.
        """
        norm = self.normalize_query(query)
        self._ensure_query_doc(norm, language=language)

        def _pick_title(d: Dict[str, Any]) -> str:
            return (
                (d.get("title") or "").strip()
                or (d.get("article") or "").strip()
                or (d.get("section") or "").strip()
                or (d.get("summary") or "").strip()
                or (d.get("text") or "").strip()
            )

        def _simplify_cases(doc: Dict[str, Any], chosen_title: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for c in (doc.get("cases") or []):
                if not isinstance(c, dict):
                    if isinstance(c, str) and c.strip() and c.strip() == chosen_title:
                        out.append({"case": c})
                    continue
                fr = (c.get("from_ref") or "").strip()
                if fr and fr != chosen_title:
                    continue
                row = {
                    "case_id": str(c.get("_id")) if c.get("_id") else None,
                    "case": c.get("case"),
                    "from_ref": fr or None,
                }
                row = {k: v for k, v in row.items() if v is not None}
                if row:
                    out.append(row)
            return out

        def _coerce(res: Any) -> Tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]], Optional[float]]:
            # (doc, score) / [doc, score]
            score = None
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                doc = res[0]
                score = float(res[1]) if res[1] is not None else None
                
                if isinstance(doc, dict):
                    kid = str(doc["_id"]) if doc.get("_id") else None
                    title = _pick_title(doc)
                    cases = _simplify_cases(doc, title) if title else []
                    return kid, (title or None), (cases or None), score
                if isinstance(doc, ObjectId):
                    return str(doc), None, None, score

            if isinstance(res, dict):
                kid = str(res["_id"]) if res.get("_id") else None
                title = _pick_title(res)
                cases = _simplify_cases(res, title) if title else []
                # Try to get score from dict if available
                score = float(res.get("score")) if res.get("score") is not None else None
                return kid, (title or None), (cases or None), score

            if isinstance(res, ObjectId):
                return str(res), None, None, None

            if isinstance(res, str):
                return None, (res or None), None, None

            return None, (str(res) if res is not None else None), None, None

        knowledge_id, title, cases, score = _coerce(result)

        rec: Dict[str, Any] = {}
        if knowledge_id:
            rec["knowledge_id"] = knowledge_id
            # Store collection_key when knowledge_id is present to identify which collection it belongs to
            if collection_key:
                rec["collection_key"] = collection_key
        if title:
            rec["title"] = title
        if cases:
            rec["cases"] = cases
        if score is not None:
            rec["score"] = score

        # Use $addToSet behavior: only add if not already present
        # Check for existing result with same knowledge_id AND collection_key to prevent duplicates
        # CRITICAL: Must check both knowledge_id and collection_key since same knowledge_id can exist in different collections
        query_filter = {"query": norm}
        if knowledge_id:
            # Check if this (knowledge_id, collection_key) combination already exists
            duplicate_filter = {
                "query": norm,
                "results.knowledge_id": knowledge_id
            }
            if collection_key:
                duplicate_filter["results.collection_key"] = collection_key
            else:
                # If no collection_key provided but knowledge_id exists, check for any result with this knowledge_id
                # that also has no collection_key (backward compatibility)
                duplicate_filter["$or"] = [
                    {"results.collection_key": {"$exists": False}},
                    {"results.collection_key": None}
                ]
            
            existing = self.query.find_one(duplicate_filter)
            if existing:
                # Already exists, skip to prevent duplicate
                return
        
        self.query.update_one(
            query_filter,
            {"$push": {"results": rec}}
        )

    def get_query_with_result(self, query: str, limit: Optional[int] = None, collection_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return results for a query in insertion order (most recent last).
        Deduplicates by (knowledge_id, collection_key) to prevent returning the same document multiple times.
        
        Optimized: Uses projection to only fetch the 'results' field from MongoDB, avoiding
        large document transfers (embeddings, etc.) for faster cache retrieval.
        
        Args:
            collection_key: Optional collection key to filter results. If provided, only returns
                          results matching this collection_key. If None, returns all results.
        """
        norm = self.normalize_query(query)
        # Use projection to only fetch the results field - much faster for large documents
        doc = self.find_query_doc_ci(norm, projection={"results": 1, "_id": 0})
        lst = (doc or {}).get("results", []) or []

        def _project(r: Dict[str, Any]) -> Dict[str, Any]:
            out = {}
            if "knowledge_id" in r and r["knowledge_id"]:
                out["knowledge_id"] = r["knowledge_id"]
            if "collection_key" in r and r["collection_key"]:
                out["collection_key"] = r["collection_key"]
            title = r.get("title") or r.get("text")
            if title:
                out["title"] = title
            if "cases" in r and isinstance(r["cases"], list):
                out["cases"] = r["cases"]
            # Include score if it exists
            if "score" in r and r["score"] is not None:
                out["score"] = float(r["score"])
            return out

        projected = [_project(r) for r in lst]
        
        # Filter by collection_key if provided
        if collection_key:
            projected = [r for r in projected if r.get("collection_key") == collection_key]
        
        # Deduplicate by (knowledge_id, collection_key) tuple (keep first occurrence to maintain insertion order)
        # CRITICAL: Must use both knowledge_id and collection_key for deduplication since same knowledge_id
        # can exist in different collections
        seen_keys = set()
        deduped = []
        for item in projected:
            kid = item.get("knowledge_id")
            coll_key = item.get("collection_key")
            # Create deduplication key: (knowledge_id, collection_key) if both exist, or just knowledge_id/title
            if kid:
                # Use (knowledge_id, collection_key) tuple for deduplication
                dedup_key = (kid, coll_key) if coll_key else (kid, None)
            else:
                # Use title as fallback for deduplication if no knowledge_id
                dedup_key = item.get("title")
            
            if dedup_key and dedup_key in seen_keys:
                continue
            if dedup_key:
                seen_keys.add(dedup_key)
            deduped.append(item)
        
        return deduped[:limit] if limit else deduped

    # ============= REPHRASE LINKING (ObjectId; no placeholders here) =============
    def find_query_doc_by_id(self, _id: ObjectId) -> Optional[dict]:
        return self.query.find_one({"_id": _id})

    def link_rephrased_id(self, original_id: ObjectId, rephrased_id: ObjectId) -> None:
        """Store a one-way link: original --rephrased_ref(ObjectId)--> rephrased."""
        if not isinstance(original_id, ObjectId) or not isinstance(rephrased_id, ObjectId):
            raise TypeError("original_id and rephrased_id must be ObjectId")
        if original_id == rephrased_id:
            logger.warning("[QM][REPHRASE] skip self-link for _id=%s", original_id)
            return
        self.query.update_one(
            {"_id": original_id},
            {"$set": {"rephrased_ref": rephrased_id}},
            upsert=False,
        )

    def link_rephrased(self, original_query: str, rephrased_query: str) -> Optional[ObjectId]:
        """
        Convenience: link via ObjectId IFF both docs already exist.
        Does NOT create placeholder docs. Returns target ObjectId or None.
        """
        orig = self.find_query_doc_ci(original_query)
        reph = self.find_query_doc_ci(rephrased_query)
        if not orig or not reph:
            logger.warning("[QM][REPHRASE] missing doc(s): orig=%s reph=%s", bool(orig), bool(reph))
            return None
        self.link_rephrased_id(orig["_id"], reph["_id"])
        logger.info("[QM][REPHRASE] linked '%s' -> '%s' (_id=%s)", original_query, rephrased_query, str(reph["_id"]))
        return reph["_id"]
    
    def get_cases_by_titles(self, titles: List[str]) -> List[Dict[str, Any]]:
        """
        only for client 
        Resolve a list of case/document titles to full docs.

        - Case-insensitive match on `title` (uses _QUERY_COLLATION).
        - Works with either self.cases (if present) or falls back to self.main.
        - Deduplicates by normalized title; prefers the newest by ObjectId timestamp.
        - Preserves the input order of `titles` in the returned list.
        """
        if not titles:
            return []

        # Clean the incoming list and keep original order
        cleaned: List[str] = []
        seen = set()
        for t in titles:
            if isinstance(t, str):
                s = t.strip()
                if s and s.lower() not in seen:
                    cleaned.append(s)
                    seen.add(s.lower())

        if not cleaned:
            return []

        coll = getattr(self, "cases", None) or self.main

        # Fetch all in one call (case-insensitive via collation)
        try:
            cursor = coll.find({"title": {"$in": cleaned}}, collation=_QUERY_COLLATION)
            found = list(cursor)
        except Exception as e:
            logger.exception("[DB][titles] find by titles failed: %s", e)
            return []

        # Deduplicate by normalized title; prefer newest
        from bson import ObjectId  # local import to avoid top-level assumptions
        def _norm_title(s: str) -> str:
            return " ".join((s or "").split()).lower()

        choose_by_title: Dict[str, Dict[str, Any]] = {}
        for d in found:
            t = d.get("title") or ""
            key = _norm_title(t)
            prev = choose_by_title.get(key)
            if prev is None:
                choose_by_title[key] = d
            else:
                try:
                    prev_id, cur_id = prev.get("_id"), d.get("_id")
                    if isinstance(prev_id, ObjectId) and isinstance(cur_id, ObjectId):
                        # Keep the newest (greater generation_time)
                        if cur_id.generation_time >= prev_id.generation_time:
                            choose_by_title[key] = d
                except Exception:
                    # On any issue, keep the existing one
                    pass

        # Preserve input order
        out: List[Dict[str, Any]] = []
        missing = 0
        for t in cleaned:
            key = _norm_title(t)
            doc = choose_by_title.get(key)
            if doc is not None:
                out.append(doc)
            else:
                missing += 1
                logger.info("[DB][titles] not found: %r", t)

        logger.info(
            "[DB][titles] resolved titles -> docs: requested=%d found=%d missing=%d",
            len(cleaned), len(out), missing,
        )
        return out
    
    def _process_atlas_search_results(
        self, 
        cursor, 
        case_titles_set: Optional[set], 
        limit: int
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], int, int]:
        """
        Helper function to process Atlas Search results.
        
        Returns:
            Tuple of (results, raw_count, filtered_count)
        """
        results: List[Tuple[Dict[str, Any], float]] = []
        raw_results_count = 0
        filtered_count = 0
        
        for doc in cursor:
            raw_results_count += 1
            # Extract score from MongoDB Atlas Search
            score_value = doc.get("score")
            if score_value is None:
                logger.warning(
                    "[KEYWORD] Atlas Search result missing score field. This may indicate an issue with index configuration."
                )
                base_score = 0.5
            else:
                try:
                    base_score = float(score_value)
                    # Normalize score to 0.0-1.0 range if needed
                    if base_score > 1.0:
                        base_score = min(base_score / 10.0, 1.0)
                except (ValueError, TypeError):
                    base_score = 0.5
            
            # Remove score from doc to avoid confusion
            doc.pop("score", None)
            
            # Additional filtering: ensure title matches (double-check)
            title = doc.get("title", "").strip()
            
            # If we have case_titles_set, check if title matches (case-insensitive, flexible matching)
            if case_titles_set:
                # Try exact match first
                if title in case_titles_set:
                    result_doc = {k: v for k, v in doc.items() if k != "embedding"}
                    results.append((result_doc, base_score))
                else:
                    # Try case-insensitive match
                    title_lower = title.lower()
                    matched = any(mysql_title.lower() == title_lower for mysql_title in case_titles_set)
                    
                    if matched:
                        result_doc = {k: v for k, v in doc.items() if k != "embedding"}
                        results.append((result_doc, base_score))
                    else:
                        filtered_count += 1
                        logger.debug(f"[KEYWORD] Filtered out result: '{title}' (not in MySQL cases). MySQL titles: {list(case_titles_set)[:3]}...")
            else:
                # No filtering needed - include all results
                result_doc = {k: v for k, v in doc.items() if k != "embedding"}
                results.append((result_doc, base_score))
        
        # Sort by score descending (should already be sorted, but ensure)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        results = results[:limit]
        
        return results, raw_results_count, filtered_count
    
    def keyword_search_cases(self, query: str, case_docs: List[Dict[str, Any]], limit: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform keyword search using MongoDB Atlas Search.
        Searches the MongoDB collection directly using full-text search index.
        
        Args:
            query: User search query
            case_docs: List of case documents from MySQL (used for filtering results)
            limit: Maximum number of results to return
        
        Returns:
            List of (doc, score) tuples sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Get the full-text search index name from config
        fulltext_index_name = self.config.get("main_fulltext_index", "fulltext_index")
        collection = self.main  # client_cases collection
        
        if collection is None:
            logger.warning("[KEYWORD] MongoDB collection not available, falling back to empty results")
            return []
        
        # Extract case titles from case_docs for filtering (if provided)
        # This ensures we only return results that match the MySQL cases
        case_titles_set = None
        if case_docs:
            case_titles_set = {doc.get("title", "").strip() for doc in case_docs if doc.get("title")}
            logger.info(f"[KEYWORD] Filtering to {len(case_titles_set)} case titles from MySQL")
            if case_titles_set:
                # Log sample titles for debugging
                sample_titles = list(case_titles_set)[:3]
                logger.debug(f"[KEYWORD] Sample MySQL titles: {sample_titles}")
        
        try:
            # Check if collection has any documents first
            doc_count = collection.count_documents({})
            if doc_count == 0:
                logger.warning(f"[KEYWORD] Collection '{collection.name}' is empty. No documents to search.")
                return []
            
            logger.info(f"[KEYWORD] Collection '{collection.name}' has {doc_count} documents")
            
            # Build MongoDB Atlas Search aggregation pipeline
            # Use compound search with multiple text fields for better results
            # Try multiple field paths explicitly instead of wildcard for better compatibility
            search_stage = {
                "$search": {
                    "index": fulltext_index_name,
                    "compound": {
                        "should": [
                            {
                                "text": {
                                    "query": query,
                                    "path": "title",
                                    "score": {"boost": {"value": 2.0}}  # Boost title matches
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "summary",
                                    "score": {"boost": {"value": 1.0}}  # Standard weight for summary
                                }
                            }
                        ],
                        "minimumShouldMatch": 1  # At least one field must match
                    }
                }
            }
            
            pipeline = [
                search_stage,
                {
                    "$project": {
                        "_id": 1,
                        "title": 1,
                        "summary": 1,
                        "text": 1,
                        "score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$limit": limit * 2  # Get more results initially for filtering
                }
            ]
            
            # Execute MongoDB Atlas Search
            logger.info(f"[KEYWORD] Searching MongoDB collection '{collection.name}' with Atlas Search (index: {fulltext_index_name})")
            logger.info(f"[KEYWORD] Query: '{query}'")
            logger.debug(f"[KEYWORD] Pipeline: {pipeline}")
            
            cursor = collection.aggregate(pipeline)
            results, raw_results_count, filtered_count = self._process_atlas_search_results(
                cursor, case_titles_set, limit
            )
            
            logger.info(f"[KEYWORD] Atlas Search: {raw_results_count} raw results, {filtered_count} filtered out, {len(results)} final matches")
            if raw_results_count > 0 and len(results) == 0:
                logger.warning(
                    f"[KEYWORD] WARNING: Atlas Search returned {raw_results_count} results but all were filtered out. "
                    f"This suggests a title mismatch between MongoDB and MySQL. "
                    f"MySQL titles: {list(case_titles_set)[:5] if case_titles_set else 'None'}"
                )
            if results:
                logger.info(f"[KEYWORD] Top {min(5, len(results))} results:")
                for i, (doc, score) in enumerate(results[:5]):
                    logger.info(f"[KEYWORD]   {i+1}. '{doc.get('title', 'N/A')}' (score: {score:.3f})")
            
            return results
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if error is due to missing index
            if "index" in error_str and ("not found" in error_str or "does not exist" in error_str or "unknown" in error_str):
                logger.error(
                    "[KEYWORD] MongoDB Atlas Search index '%s' not found or not configured for collection '%s'. "
                    "Atlas Search requires a search index to be created in MongoDB Atlas UI. "
                    "Error: %s",
                    fulltext_index_name,
                    collection.name if collection else "None",
                    e
                )
                logger.error(
                    "[KEYWORD] To fix: Create a search index named '%s' in MongoDB Atlas for collection '%s' "
                    "with fields: title (text), summary (text)",
                    fulltext_index_name,
                    collection.name if collection else "None"
                )
            elif "compound" in error_str or "filter" in error_str:
                # Try simpler search query as fallback
                logger.warning(f"[KEYWORD] Compound search failed, trying simpler text search: {e}")
                try:
                    simple_search_stage = {
                        "$search": {
                            "index": fulltext_index_name,
                            "text": {
                                "query": query,
                                "path": {"wildcard": "*"}  # Try wildcard path
                            }
                        }
                    }
                    simple_pipeline = [
                        simple_search_stage,
                        {
                            "$project": {
                                "_id": 1,
                                "title": 1,
                                "summary": 1,
                                "text": 1,
                                "score": {"$meta": "searchScore"}
                            }
                        },
                        {"$limit": limit * 2}
                    ]
                    logger.info(f"[KEYWORD] Retrying with simpler search query...")
                    cursor = collection.aggregate(simple_pipeline)
                    results, raw_results_count, filtered_count = self._process_atlas_search_results(
                        cursor, case_titles_set, limit
                    )
                    logger.info(f"[KEYWORD] Simple search: {raw_results_count} raw results, {filtered_count} filtered, {len(results)} final")
                    return results
                except Exception as fallback_error:
                    logger.error(f"[KEYWORD] Fallback search also failed: {fallback_error}", exc_info=True)
            else:
                logger.error(
                    "[KEYWORD] MongoDB Atlas Search failed for query '%s': %s",
                    query[:50], e,
                    exc_info=True
                )
            
            # Return empty results on error (graceful degradation)
            return []
    
    def atlas_search_main(self, query: str, limit: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform BM25 keyword search using MongoDB Atlas Search on the main collection.
        Used for hybrid search combining BM25 with semantic search.
        
        Args:
            query: User search query
            limit: Maximum number of results to return
            
        Returns:
            List of (doc, bm25_score) tuples sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Get the full-text search index name from config
        fulltext_index_name = self.config.get("main_fulltext_index", "fulltext_index")
        collection = self.main
        
        if collection is None:
            logger.warning("[ATLAS][MAIN] MongoDB collection not available, falling back to empty results")
            return []
        
        try:
            # Build MongoDB Atlas Search aggregation pipeline
            # Search multiple fields: title, article, section, summary, text
            search_stage = {
                "$search": {
                    "index": fulltext_index_name,
                    "compound": {
                        "should": [
                            {
                                "text": {
                                    "query": query,
                                    "path": "title",
                                    "score": {"boost": {"value": 3.0}}  # Higher weight for title
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "article",
                                    "score": {"boost": {"value": 2.0}}  # Medium weight for article
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "section",
                                    "score": {"boost": {"value": 2.0}}  # Medium weight for section
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "summary",
                                    "score": {"boost": {"value": 1.0}}  # Lower weight for summary
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "text",
                                    "score": {"boost": {"value": 0.5}}  # Lowest weight for full text
                                }
                            }
                        ],
                        "minimumShouldMatch": 1  # At least one field must match
                    }
                }
            }
            
            pipeline = [
                search_stage,
                {
                    "$project": {
                        "_id": 1,
                        "title": 1,
                        "article": 1,
                        "section": 1,
                        "summary": 1,
                        "text": 1,
                        "bm25_score": {"$meta": "searchScore"}
                    }
                },
                {
                    "$limit": limit
                }
            ]
            
            # Execute MongoDB Atlas Search
            logger.info(f"[ATLAS][MAIN] Searching collection '{collection.name}' with Atlas Search (index: {fulltext_index_name})")
            logger.info(f"[ATLAS][MAIN] Query: '{query}'")
            
            cursor = collection.aggregate(pipeline)
            results: List[Tuple[Dict[str, Any], float]] = []
            
            for doc in cursor:
                # Extract BM25 score from MongoDB Atlas Search
                score_value = doc.get("bm25_score")
                if score_value is None:
                    logger.warning(
                        "[ATLAS][MAIN] Atlas Search result missing score field. This may indicate an issue with index configuration."
                    )
                    bm25_score = 0.5
                else:
                    try:
                        bm25_score = float(score_value)
                        # Normalize BM25 score to 0.0-1.0 range if needed
                        # Atlas Search BM25 scores can vary, normalize if > 1.0
                        if bm25_score > 1.0:
                            bm25_score = min(bm25_score / 10.0, 1.0)
                    except (ValueError, TypeError):
                        bm25_score = 0.5
                
                # Remove score from doc to avoid confusion
                doc.pop("bm25_score", None)
                
                # Remove embedding field from result (not needed for keyword search results)
                result_doc = {k: v for k, v in doc.items() if k != "embedding"}
                results.append((result_doc, bm25_score))
            
            # Sort by score descending (should already be sorted, but ensure)
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"[ATLAS][MAIN] Atlas Search found {len(results)} matches")
            if results:
                logger.info(f"[ATLAS][MAIN] Top {min(5, len(results))} BM25 results:")
                for i, (doc, score) in enumerate(results[:5]):
                    logger.info(f"[ATLAS][MAIN]   {i+1}. '{doc.get('title', 'N/A')}' (BM25 score: {score:.3f})")
            
            return results
            
        except Exception as e:
            # Check if error is due to missing index
            error_str = str(e).lower()
            if "index" in error_str and ("not found" in error_str or "does not exist" in error_str):
                logger.warning(
                    "[ATLAS][MAIN] MongoDB Atlas Search index '%s' not found for collection '%s'. "
                    "Falling back to semantic-only search. Error: %s",
                    fulltext_index_name,
                    collection.name if collection else "None",
                    e
                )
            else:
                logger.error(
                    "[ATLAS][MAIN] MongoDB Atlas Search failed for query '%s': %s",
                    query[:50], e,
                    exc_info=True
                )
            # Return empty results on error (graceful degradation - will use semantic-only)
            return []
    
    def track_query_cache_hit(self, normalized_query: str, cache_type: str = "query", language: Optional[str] = None) -> None:
        """
        Track a cache hit for a query.
        cache_type: "query" for query cache hit, "insight" for insight cache hit
        Stores both the count and an array of datetime strings for time series analysis.
        Each datetime string represents one cache hit event with full time information.
        language: Optional language to ensure document has correct language field (important for Spanish queries)
        """
        try:
            # Get current time in US-west timezone (California)
            if US_WEST_TZ:
                now = datetime.datetime.now(US_WEST_TZ)
            else:
                now = datetime.datetime.utcnow()
            
            # Store datetime with time (YYYY-MM-DDTHH:MM:SS) for accurate time tracking
            datetime_str = now.strftime("%Y-%m-%dT%H:%M:%S")
            date_array_name = "query_cache_hit_dates" if cache_type == "query" else "insight_cache_hit_dates"
            
            # Build update operation - only use query_cache_hit_dates array, no query_cache_hits counter
            update_op = {
                "$set": {"updated_at": now},
                "$push": {date_array_name: datetime_str}  # Add datetime with time to track each event
            }
            
            # If language is provided, ensure it's set on the document (important for Spanish queries)
            if language:
                update_op["$set"]["language"] = language
            
            self.query.update_one(
                {"query": normalized_query},
                update_op,
                upsert=False  # Don't create document if it doesn't exist - it should exist from track_query_usage
            )
        except Exception as e:
            logger.warning(f"[TRACK] Failed to track cache hit: {e}")
    
    def track_query_usage(self, normalized_query: str, language: str = "en", avg_relevance_score: Optional[float] = None, en_query_ref: Optional[ObjectId] = None) -> None:
        """
        Track query usage with language and average relevance score.
        Creates or updates the query document. Cache hits are tracked separately via track_cache_hit().
        
        Stats are calculated from query_cache_hit_dates array length:
        - Total Calls = len(query_cache_hit_dates) + 1 (the +1 is for the initial query)
        - Cache Hits = len(query_cache_hit_dates)
        
        For Spanish queries (language='es'), use en_query_ref to reference the English query
        instead of storing avg_relevance_score separately.
        """
        try:
            if not normalized_query or not normalized_query.strip():
                logger.warning(f"[TRACK] Cannot track empty or None query")
                return
                
            # Get current time in US-west timezone (California)
            if US_WEST_TZ:
                now = datetime.datetime.now(US_WEST_TZ)
            else:
                now = datetime.datetime.utcnow()
            
            # Store datetime with time (YYYY-MM-DDTHH:MM:SS) for searched_datetime array
            datetime_str = now.strftime("%Y-%m-%dT%H:%M:%S")
            
            # Check if document already exists (may have been created by incremental caching via store_case_query_pairs)
            existing_doc = self.query.find_one({"query": normalized_query})
            
            # Build update fields - separate $set and $setOnInsert to avoid conflicts
            # Note: We no longer use usage_count or usage_dates - everything is calculated from query_cache_hit_dates
            # IMPORTANT: Use $set to update existing documents (including those created by incremental caching)
            # and $setOnInsert only for fields that should only be set on creation
            # CRITICAL: $set only updates specified fields, preserving all other existing fields (from store_case_query_pairs)
            set_fields = {
                "language": language, 
                "updated_at": now,
                "query": normalized_query  # Always set query field (required for analytics and unique index)
            }
            
            set_on_insert_fields = {
                "created_at": now
                # Note: query_cache_hit_dates will be created by $addToSet in track_cache_hit
                # Note: query field is set in $set, not here, to avoid MongoDB conflict
                # Note: If document exists from incremental caching, we update it instead of creating a new one
            }
            
            # Add language-specific fields
            if language == "es" and en_query_ref is not None:
                # For Spanish queries, store reference to English query instead of avg_relevance_score
                set_fields["en_query_ref"] = en_query_ref
            elif avg_relevance_score is not None:
                # For English queries, store average relevance score (0-1 range, will be converted to percentage in analytics)
                set_fields["avg_relevance_score"] = float(avg_relevance_score)
            
            # Build the update operation
            # Note: We don't track usage_count or usage_dates anymore - stats are calculated from query_cache_hit_dates length
            # Add searched_datetime to track when queries are searched
            # CRITICAL: $set preserves ALL existing fields from store_case_query_pairs (searched_case_ids, search_range_size, 
            # last_search_at, query_norm, results, embedding) - it only updates the fields we specify
            update_fields = {
                "$set": set_fields,  # This will update/merge with existing fields, NOT replace the document
                "$setOnInsert": set_on_insert_fields,
                "$push": {"searched_datetime": datetime_str}  # Track each search with full datetime
            }
            
            # Log if document exists from incremental caching to verify merge
            if existing_doc:
                has_cache_fields = any(key in existing_doc for key in ["searched_case_ids", "search_range_size", "last_search_at", "query_norm"])
                if has_cache_fields:
                    logger.info(f"[TRACK] Found existing document with cache fields - will merge analytics fields (query: {normalized_query[:50]}...)")
                else:
                    logger.debug(f"[TRACK] Found existing document without cache fields - will update (query: {normalized_query[:50]}...)")
            
            try:
                result = self.query.update_one(
                {"query": normalized_query},
                update_fields,
                upsert=True
            )
                
                # Verify the document was created/updated
                if result.upserted_id:
                    logger.info(f"[TRACK] Created new query document: {normalized_query[:50]}... (id: {result.upserted_id})")
                elif result.modified_count > 0:
                    logger.debug(f"[TRACK] Updated existing query document: {normalized_query[:50]}...")
                else:
                    logger.warning(f"[TRACK] No document was created or modified for query: {normalized_query[:50]}...")
                    
                # Double-check: verify document exists with query field and verify merge
                verify_doc = self.query.find_one({"query": normalized_query})
                if not verify_doc:
                    logger.error(f"[TRACK] CRITICAL: Document not found after upsert for query: {normalized_query[:50]}...")
                elif "query" not in verify_doc or not verify_doc.get("query"):
                    logger.error(f"[TRACK] CRITICAL: Document exists but missing 'query' field: {normalized_query[:50]}...")
                else:
                    # Verify merge: check if both analytics and cache fields exist
                    has_analytics = any(key in verify_doc for key in ["avg_relevance_score", "language", "searched_datetime"])
                    has_cache = any(key in verify_doc for key in ["searched_case_ids", "search_range_size", "last_search_at"])
                    if has_analytics and has_cache:
                        logger.info(f"[TRACK] ✅ Verified merge: document has both analytics and cache fields (query: {normalized_query[:50]}...)")
                    elif has_analytics:
                        logger.debug(f"[TRACK] Document has analytics fields only (cache fields may be added later)")
                    elif has_cache:
                        logger.debug(f"[TRACK] Document has cache fields only (analytics fields may be added later)")
                    else:
                        logger.debug(f"[TRACK] Document exists with query field: {normalized_query[:50]}...")
                    
            except Exception as update_error:
                # Handle specific MongoDB errors
                error_code = getattr(update_error, 'code', None)
                error_msg = str(update_error)
                
                if error_code == 11000:  # Duplicate key error
                    logger.warning(f"[TRACK] Duplicate key error (likely race condition). Retrying with find-then-update: {normalized_query[:50]}...")
                    # Retry with find-then-update approach
                    try:
                        # Try to find existing document first
                        existing = self.query.find_one({"query": normalized_query})
                        if existing:
                            # Update existing document
                            update_existing = {
                                "$set": set_fields
                            }
                            self.query.update_one(
                                {"_id": existing["_id"]},
                                update_existing
                            )
                            logger.info(f"[TRACK] Successfully updated existing document after retry: {normalized_query[:50]}...")
                        else:
                            # Document doesn't exist, try insert
                            new_doc = {
                                "query": normalized_query,
                                "language": language,
                                "created_at": now,
                                "updated_at": now
                            }
                            if language == "es" and en_query_ref is not None:
                                new_doc["en_query_ref"] = en_query_ref
                            elif avg_relevance_score is not None:
                                new_doc["avg_relevance_score"] = float(avg_relevance_score)
                            
                            self.query.insert_one(new_doc)
                            logger.info(f"[TRACK] Successfully inserted new document after retry: {normalized_query[:50]}...")
                    except Exception as retry_error:
                        logger.error(f"[TRACK] Retry also failed for '{normalized_query[:50]}...': {retry_error}", exc_info=True)
                else:
                    # Re-raise if it's not a duplicate key error
                    raise
                
        except Exception as e:
            logger.error(f"[TRACK] Failed to track query usage for '{normalized_query[:50] if normalized_query else 'None'}...': {e}", exc_info=True)
    