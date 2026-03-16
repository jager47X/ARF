# services/rag/rag_dependencies/vector_search.py
"""
MongoDB Atlas Vector Search implementation.
Replaces FAISS and brute-force cosine similarity with native MongoDB vector search.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pymongo.collection import Collection
from pymongo.errors import ExecutionTimeout, NetworkTimeout, OperationFailure, PyMongoError

logger = logging.getLogger(__name__)

# Default vector search configuration
VECTOR_INDEX_NAME = "vector_index"
VECTOR_FIELD = "embedding"


class VectorSearch:
    """
    MongoDB Atlas Vector Search wrapper.

    Provides semantic search using MongoDB's $vectorSearch aggregation pipeline.
    Replaces FAISS index-based search with native MongoDB vector operations.
    """

    def __init__(
        self,
        collection: Collection,
        vector_index_name: str = VECTOR_INDEX_NAME,
        vector_field: str = VECTOR_FIELD,
        unique_index: str = "title",
        bias_map: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Vector Search.

        Args:
            collection: MongoDB collection with vector index
            vector_index_name: Name of the Atlas vector search index
            vector_field: Field name containing embeddings
            unique_index: Field used for bias lookups (e.g., 'title', 'query')
            bias_map: Optional dict mapping unique_index values to score adjustments
        """
        self.collection = collection
        self.vector_index_name = vector_index_name
        self.vector_field = vector_field
        self.unique_index = unique_index
        self.bias_map = bias_map or {}

        logger.info(
            "VectorSearch initialized: collection=%s, index=%s, field=%s",
            collection.name, vector_index_name, vector_field
        )

    def search_similar(
        self,
        query_embedding: Union[List[float], np.ndarray],
        k: int = 20,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Tuple[dict, float]]]:
        """
        Search for similar documents using MongoDB Atlas Vector Search.

        Args:
            query_embedding: Query vector (list or numpy array)
            k: Number of results to return
            extra_filter: Optional MongoDB filter for pre-filtering results

        Returns:
            List of (document, adjusted_score) tuples, sorted by score descending,
            or None if no results
        """
        if query_embedding is None:
            logger.warning("Query embedding is None")
            return None

        # Convert to list for MongoDB
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.tolist()
        else:
            query_vector = list(query_embedding)

        try:
            # Build $vectorSearch pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "path": self.vector_field,
                        "queryVector": query_vector,
                        "numCandidates": max(k * 5, 100),  # Search more candidates for better recall
                        "limit": k,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        # Standard fields - try all possible variations
                        "title": 1,
                        "article": 1,
                        "section": 1,
                        "part": 1,  # Used by CFR
                        "chapter": 1,  # Used by US Code and CFR
                        "subchapter": 1,  # Used by CFR
                        # Text fields - try all possible variations
                        "text": 1,
                        "summary": 1,  # Used by client cases
                        "content": 1,  # Alternative text field
                        "body": 1,  # Alternative text field
                        # Nested structures
                        "sections": 1,  # Used by CFR
                        "clauses": 1,  # Used by US Code
                        # Other fields
                        "case": 1,
                        "query": 1,
                        "references": 1,
                        "cases": 1,
                        "document_type": 1,  # CRITICAL: Include document_type to preserve correct type
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            # Add filter if provided
            if extra_filter:
                # MongoDB Atlas Vector Search supports filters within the $vectorSearch stage
                pipeline[0]["$vectorSearch"]["filter"] = extra_filter

            # Execute search with explicit timeout (5 minutes for large collections)
            # maxTimeMS must be longer than socketTimeoutMS to prevent premature timeouts
            cursor = self.collection.aggregate(pipeline, maxTimeMS=300000)  # 5 minutes
            results: List[Tuple[dict, float]] = []

            for doc in cursor:
                # Extract score from metadata
                # MongoDB Atlas Vector Search returns score via $meta: "vectorSearchScore"
                score_value = doc.get("score")
                if score_value is None:
                    logger.warning(
                        "Vector search result missing score field. Document keys: %s. "
                        "This may indicate an issue with MongoDB Atlas Vector Search configuration.",
                        list(doc.keys())
                    )
                    base_score = 0.0
                else:
                    try:
                        base_score = float(score_value)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Vector search result has invalid score value: %r (type: %s). Error: %s",
                            score_value, type(score_value).__name__, e
                        )
                        base_score = 0.0

                # Remove score from doc to avoid confusion
                doc.pop("score", None)

                # Apply bias if configured
                key = doc.get(self.unique_index)
                bias = float(self.bias_map.get(key, 0.0)) if key is not None else 0.0

                # Adjust score (clamp to [0, 1])
                adjusted_score = max(0.0, min(1.0, base_score + bias))

                results.append((doc, adjusted_score))

            if not results:
                logger.info("No results found for vector search")
                return None

            # Sort by adjusted score
            results.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                "Vector search returned %d results (top score: %.3f)",
                len(results), results[0][1] if results else 0.0
            )

            # Log detailed results for debugging
            logger.info("[VECTOR][DETAIL] Top 10 vector search results:")
            for i, (doc, score) in enumerate(results[:10], 1):
                title = doc.get("title", "N/A")
                article = doc.get("article", "")
                section = doc.get("section", "")

                # Log document structure if fields are missing
                if title == "N/A" or not article or not section:
                    logger.warning(
                        "[VECTOR][DETAIL] #%d: Missing fields - doc keys: %s, "
                        "has_clauses: %s, has_sections: %s, document_type: %s",
                        i, list(doc.keys()),
                        bool(doc.get("clauses")),
                        bool(doc.get("sections")),
                        doc.get("document_type", "N/A")
                    )
                    # Try to extract title from clauses or sections if available
                    if title == "N/A":
                        if doc.get("clauses") and isinstance(doc["clauses"], list) and len(doc["clauses"]) > 0:
                            first_clause = doc["clauses"][0]
                            title = first_clause.get("title", "N/A")
                            logger.info("[VECTOR][DETAIL] #%d: Extracted title from first clause: %r", i, title)
                        elif doc.get("sections") and isinstance(doc["sections"], list) and len(doc["sections"]) > 0:
                            first_section = doc["sections"][0]
                            title = first_section.get("title", "N/A")
                            logger.info("[VECTOR][DETAIL] #%d: Extracted title from first section: %r", i, title)

                logger.info("[VECTOR][DETAIL] #%d: score=%.4f title=%r article=%r section=%r",
                           i, score, title, article, section)

            return results

        except (NetworkTimeout, ExecutionTimeout) as e:
            logger.error("Vector search timed out for collection %s: %s", self.collection.name, e)
            logger.warning("This may indicate the collection is too large or the query is too complex. Consider increasing timeout or optimizing the query.")
            return None
        except OperationFailure as e:
            logger.error("Vector search operation failed for collection %s: %s", self.collection.name, e)
            # Check if it's an index error
            error_str = str(e).lower()
            if "index" in error_str and ("not found" in error_str or "does not exist" in error_str):
                logger.warning("Vector search index '%s' not found for collection '%s'", self.vector_index_name, self.collection.name)
            return None
        except Exception as e:
            logger.exception("Vector search failed for collection %s: %s", self.collection.name, e)
            return None

    def search_similar_with_clauses(
        self,
        query_embedding: Union[List[float], np.ndarray],
        k: int = 20,
        k_clauses: int = 10,
        extra_filter: Optional[Dict[str, Any]] = None,
        use_same_index: bool = True,
    ) -> Optional[List[Tuple[dict, float]]]:
        """
        Search for similar documents using both document-level and subdocument-level (clauses/sections) embeddings.
        Supports both 'clauses.embedding' and 'sections.embedding' paths.
        Uses the same vector index for both searches (index must include both 'embedding' and subdocument embedding paths).
        Merges results from both searches, deduplicating by document ID.

        Args:
            query_embedding: Query vector (list or numpy array)
            k: Number of document-level results to return
            k_clauses: Number of subdocument-level results to return
            extra_filter: Optional MongoDB filter for pre-filtering results
            use_same_index: If True, use the same index for both searches (default: True)

        Returns:
            List of (document, adjusted_score) tuples, sorted by score descending,
            or None if no results
        """
        # Search document-level embeddings
        doc_results = self.search_similar(query_embedding, k=k, extra_filter=extra_filter) or []

        # Detect which subdocument path exists by checking a sample document
        subdoc_paths = []
        try:
            # Try to find a sample document to detect structure
            sample_doc = self.collection.find_one({}, {"clauses": 1, "sections": 1})
            if sample_doc:
                if "clauses" in sample_doc and isinstance(sample_doc["clauses"], list) and len(sample_doc["clauses"]) > 0:
                    # Check if clauses have embeddings
                    if any(c.get("embedding") for c in sample_doc["clauses"] if isinstance(c, dict)):
                        subdoc_paths.append("clauses.embedding")
                if "sections" in sample_doc and isinstance(sample_doc["sections"], list) and len(sample_doc["sections"]) > 0:
                    # Check if sections have embeddings
                    if any(s.get("embedding") for s in sample_doc["sections"] if isinstance(s, dict)):
                        subdoc_paths.append("sections.embedding")
        except PyMongoError as e:
            logger.debug("Could not detect subdocument structure: %s", e)

        # If no subdocument paths detected, try both (one may work, or both will fail gracefully)
        # This ensures backward compatibility - if neither path exists, searches will just fail gracefully
        if not subdoc_paths:
            subdoc_paths = ["clauses.embedding", "sections.embedding"]

        # Search subdocument-level embeddings using the same index
        subdoc_results = []
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.tolist()
        else:
            query_vector = list(query_embedding)

        for subdoc_path in subdoc_paths:
            try:
                # Determine which fields to project based on path
                if subdoc_path == "clauses.embedding":
                    project_fields = {
                        "_id": 1,
                        "title": 1,
                        "article": 1,
                        "section": 1,
                        "part": 1,  # Add part for CFR
                        "chapter": 1,  # Add chapter for CFR
                        "subchapter": 1,  # Add subchapter for CFR
                        "text": 1,
                        "summary": 1,
                        "clauses": 1,
                        "document_type": 1,  # CRITICAL: Include document_type to preserve correct type
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                else:  # sections.embedding
                    project_fields = {
                        "_id": 1,
                        "title": 1,
                        "article": 1,
                        "part": 1,
                        "chapter": 1,
                        "section": 1,
                        "subchapter": 1,  # Add subchapter for CFR
                        "text": 1,
                        "summary": 1,
                        "sections": 1,
                        "document_type": 1,  # CRITICAL: Include document_type to preserve correct type
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }

                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": self.vector_index_name,  # Same index
                            "path": subdoc_path,  # Subdocument-level path
                            "queryVector": query_vector,
                            "numCandidates": max(k_clauses * 5, 100),
                            "limit": k_clauses,
                        }
                    },
                    {
                        "$project": project_fields
                    }
                ]

                if extra_filter:
                    pipeline[0]["$vectorSearch"]["filter"] = extra_filter

                cursor = self.collection.aggregate(pipeline, maxTimeMS=300000)  # 5 minutes for large collections
                for doc in cursor:
                    score_value = doc.get("score")
                    base_score = float(score_value) if score_value is not None else 0.0
                    doc.pop("score", None)

                    # Apply bias if configured
                    key = doc.get(self.unique_index)
                    bias = float(self.bias_map.get(key, 0.0)) if key is not None else 0.0
                    adjusted_score = max(0.0, min(1.0, base_score + bias))

                    subdoc_results.append((doc, adjusted_score))

            except (OperationFailure, NetworkTimeout, ExecutionTimeout) as e:
                logger.debug("Subdocument-level vector search failed for path %s (this is expected if index doesn't include this path): %s", subdoc_path, e)

        # Track which documents came from which search type
        doc_ids_from_subdoc_level: set = {doc.get("_id") for doc, _ in subdoc_results if doc.get("_id")}

        # Merge and deduplicate results by document _id
        merged_results: Dict[Any, Tuple[dict, float, bool]] = {}  # (doc, score, from_subdoc)

        # Add document-level results
        for doc, score in doc_results:
            doc_id = doc.get("_id")
            if doc_id:
                # Check if document has sections/clauses
                has_sections = doc.get("sections") and isinstance(doc.get("sections"), list) and len(doc.get("sections", [])) > 0
                has_clauses = doc.get("clauses") and isinstance(doc.get("clauses"), list) and len(doc.get("clauses", [])) > 0
                has_subdoc_structure = has_sections or has_clauses

                # For documents with sections/clauses, prefer subdocument-level results if available
                # But if subdocument search failed (no index), fall back to document-level results
                # For documents without subdoc structure (e.g., agency guidance), always keep document-level results
                if not has_subdoc_structure:
                    # Documents without subdoc structure: always keep document-level results
                    if doc_id not in merged_results or score > merged_results[doc_id][1]:
                        merged_results[doc_id] = (doc, score, False)
                else:
                    # Documents with subdoc structure: keep document-level as fallback
                    # Will be replaced by subdocument results if available (see below)
                    if doc_id not in merged_results:
                        merged_results[doc_id] = (doc, score, False)

        # Add subdocument-level results (these take priority for documents with sections/clauses)
        for doc, score in subdoc_results:
            doc_id = doc.get("_id")
            if doc_id:
                # Check if document has sections/clauses
                has_sections = doc.get("sections") and isinstance(doc.get("sections"), list) and len(doc.get("sections", [])) > 0
                has_clauses = doc.get("clauses") and isinstance(doc.get("clauses"), list) and len(doc.get("clauses", [])) > 0
                has_subdoc_structure = has_sections or has_clauses

                if doc_id in merged_results:
                    # Use max score (document-level or subdocument-level)
                    existing_score = merged_results[doc_id][1]
                    # For documents with sections/clauses, always prefer subdocument results
                    if has_subdoc_structure:
                        merged_results[doc_id] = (doc, score, True)
                    else:
                        # For documents without subdoc structure (e.g., agency guidance), use max score
                        merged_results[doc_id] = (doc, max(existing_score, score), True)
                else:
                    # New document found via subdocument search
                    merged_results[doc_id] = (doc, score, True)

        if not merged_results:
            logger.info("No results found for combined vector search")
            return None

        # Filter: For documents with sections/clauses, prefer subdocument results but allow document-level as fallback
        filtered_results: List[Tuple[dict, float]] = []
        for doc_id, (doc, score, from_subdoc) in merged_results.items():
            # Check if this document has sections or clauses
            has_sections = doc.get("sections") and isinstance(doc.get("sections"), list) and len(doc.get("sections", [])) > 0
            has_clauses = doc.get("clauses") and isinstance(doc.get("clauses"), list) and len(doc.get("clauses", [])) > 0
            has_subdoc_structure = has_sections or has_clauses

            # If document has sections/clauses and we have subdocument results, prefer those
            # But if subdocument search failed (from_subdoc=False), keep document-level results as fallback
            # This handles cases where the vector index doesn't support subdocument paths
            if has_subdoc_structure and not from_subdoc:
                # Check if we actually got any subdocument results for this document
                # If not, it means subdocument search failed, so keep document-level result
                doc_has_subdoc_results = doc_id in doc_ids_from_subdoc_level
                if doc_has_subdoc_results:
                    # We have subdocument results for this doc, skip document-level
                    logger.debug("[VECTOR][FILTER] Skipping document-level result (has subdocument results): %r (title=%r)",
                               doc_id, doc.get("title", "N/A"))
                    continue
                else:
                    # No subdocument results (search failed), keep document-level as fallback
                    logger.debug("[VECTOR][FILTER] Keeping document-level result (subdocument search failed): %r (title=%r)",
                               doc_id, doc.get("title", "N/A"))

            # Keep the result
            filtered_results.append((doc, score))

        # Sort by score descending
        final_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        logger.info(
            "Combined vector search returned %d results (doc-level: %d, subdocument-level: %d, top score: %.3f)",
            len(final_results), len(doc_results), len(subdoc_results),
            final_results[0][1] if final_results else 0.0
        )

        # Log detailed breakdown of merged results
        logger.info("[VECTOR][COMBINED][DETAIL] Document-level results: %d", len(doc_results))
        for i, (doc, score) in enumerate(doc_results[:5], 1):
            logger.info("[VECTOR][COMBINED][DETAIL]   doc#%d: score=%.4f title=%r",
                       i, score, doc.get("title", "N/A"))

        logger.info("[VECTOR][COMBINED][DETAIL] Subdocument-level results: %d", len(subdoc_results))
        for i, (doc, score) in enumerate(subdoc_results[:5], 1):
            logger.info("[VECTOR][COMBINED][DETAIL]   subdoc#%d: score=%.4f title=%r",
                       i, score, doc.get("title", "N/A"))

        logger.info("[VECTOR][COMBINED][DETAIL] Final merged results (top 10):")
        for i, (doc, score) in enumerate(final_results[:10], 1):
            logger.info("[VECTOR][COMBINED][DETAIL]   #%d: score=%.4f title=%r",
                       i, score, doc.get("title", "N/A"))

        return final_results[:k]  # Return top k results


class VectorSearchManager:
    """
    Manages multiple VectorSearch instances for different collections.

    Replaces FaissManager with MongoDB Atlas Vector Search.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_manager,  # MongoManager instance
    ):
        """
        Initialize vector search managers for main, cases, and query collections.

        Args:
            config: Configuration dict with collection names and settings
            db_manager: MongoManager instance providing access to collections
        """
        self.config = config
        self.db = db_manager

        unique_field = config.get("unique_index", "title")
        bias_map = config.get("bias", {}) or {}

        # Determine vector index name based on collection
        # Default pattern: "{collection_name}_vector_index"
        def get_index_name(coll_name: str) -> str:
            return config.get(f"{coll_name}_vector_index", "vector_index")

        # Initialize main collection search
        try:
            main_coll_name = config.get("main_collection_name")
            if main_coll_name:
                self.search_main = VectorSearch(
                    collection=self.db.main,
                    vector_index_name=get_index_name("main"),
                    vector_field=VECTOR_FIELD,
                    unique_index=unique_field,
                    bias_map=bias_map,
                )
                logger.info("Main collection vector search initialized")

                # Subdocument-level search is enabled (uses same index with clauses.embedding or sections.embedding path)
                # Will be detected automatically based on document structure
                self.use_clause_search = True
                logger.info("Subdocument-level vector search enabled (supports both clauses.embedding and sections.embedding paths)")
            else:
                self.search_main = _NoopSearch(self.db.main)
                self.use_clause_search = False
                logger.warning("No main collection configured")
        except Exception as e:
            logger.warning("Failed to initialize main vector search: %s", e)
            self.search_main = _NoopSearch(getattr(self.db, 'main', None))
            self.use_clause_search = False

        # Initialize cases collection search (if not SQL mode)
        if not config.get("sql_attached", False):
            try:
                cases_coll_name = config.get("cases_collection_name")
                if cases_coll_name:
                    self.search_cases = VectorSearch(
                        collection=self.db.cases,
                        vector_index_name=get_index_name("cases"),
                        vector_field=VECTOR_FIELD,
                        unique_index=unique_field,
                        bias_map=bias_map,
                    )
                    logger.info("Cases collection vector search initialized")
                else:
                    self.search_cases = _NoopSearch(getattr(self.db, 'cases', None))
            except Exception as e:
                logger.warning("Failed to initialize cases vector search: %s", e)
                self.search_cases = _NoopSearch(getattr(self.db, 'cases', None))
        else:
            self.search_cases = _NoopSearch(None)

        # Initialize query collection search
        try:
            query_coll_name = config.get("query_collection_name")
            if query_coll_name:
                self.search_query = VectorSearch(
                    collection=self.db.query,
                    vector_index_name=get_index_name("query"),
                    vector_field=VECTOR_FIELD,
                    unique_index="query",  # Queries use 'query' field
                    bias_map={},  # No bias for queries
                )
                logger.info("Query collection vector search initialized")
            else:
                self.search_query = _NoopSearch(self.db.query)
        except Exception as e:
            logger.warning("Failed to initialize query vector search: %s", e)
            self.search_query = _NoopSearch(getattr(self.db, 'query', None))

        logger.info("VectorSearchManager initialized successfully")

    def search_main_with_clauses(
        self,
        query_embedding: Union[List[float], np.ndarray],
        k: int = 20,
        k_clauses: int = 10,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Tuple[dict, float]]]:
        """
        Search main collection using both document-level and subdocument-level (clauses/sections) embeddings.
        Automatically detects and uses the appropriate path (clauses.embedding or sections.embedding).
        Uses the same vector index for both searches.

        Args:
            query_embedding: Query vector
            k: Number of document-level results
            k_clauses: Number of subdocument-level results
            extra_filter: Optional MongoDB filter

        Returns:
            Merged and deduplicated results from both searches
        """
        if self.use_clause_search and hasattr(self.search_main, 'search_similar_with_clauses'):
            return self.search_main.search_similar_with_clauses(
                query_embedding,
                k=k,
                k_clauses=k_clauses,
                extra_filter=extra_filter,
                use_same_index=True,
            )
        else:
            # Fallback to document-level only
            return self.search_main.search_similar(query_embedding, k=k, extra_filter=extra_filter)


class _NoopSearch:
    """
    No-op search implementation for graceful degradation.
    Returns None for all searches.
    """

    def __init__(self, collection=None):
        self.collection = collection

    def search_similar(self, *args, **kwargs):
        logger.warning("NoopSearch called - vector search not available")
        return None

    def append_new_embeddings(self, *args, **kwargs):
        # For compatibility with old code that might call this
        return None


def vector_search_by_filter(
    collection: Collection,
    query_vector: Union[List[float], np.ndarray],
    k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None,
    vector_index_name: str = VECTOR_INDEX_NAME,
    vector_field: str = VECTOR_FIELD,
) -> List[Dict[str, Any]]:
    """
    Standalone helper function for MongoDB Atlas Vector Search with filtering.

    Args:
        collection: MongoDB collection with vector index
        query_vector: Query embedding vector
        k: Number of results to return
        filter_dict: Optional MongoDB filter
        vector_index_name: Name of the Atlas vector index
        vector_field: Field containing embeddings

    Returns:
        List of documents with scores
    """
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index_name,
                "path": vector_field,
                "queryVector": query_vector,
                "numCandidates": max(k * 5, 100),
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 1,
                "title": 1,  # For client cases
                "case": 1,  # For US Constitution cases
                "text": 1,  # Fallback
                "summary": 1,  # For client cases
                "query": 1,  # For query collection
                "query_norm": 1,  # For query collection
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    if filter_dict:
        pipeline[0]["$vectorSearch"]["filter"] = filter_dict

    return list(collection.aggregate(pipeline, maxTimeMS=300000))  # 5 minutes for large collections

