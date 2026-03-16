# services/rag/RAG_interface.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from services.rag.config import (
    CLIENT_CASE_THRESHOLDS,
    EMBEDDING_MODEL,
    KEYWORD_MATCH_SCORE,
    REPHRASE_LIMIT,
    TOP_QUERY_RESULT,
)
from services.rag.rag_dependencies.alias_manager import AliasManager
from services.rag.rag_dependencies.keyword_matcher import KeywordMatcher
from services.rag.rag_dependencies.llm_verifier import LLMVerifier
from services.rag.rag_dependencies.mongo_manager import MongoManager
from services.rag.rag_dependencies.query_manager import QueryManager
from services.rag.rag_dependencies.query_processor import QueryProcessor
from services.rag.rag_dependencies.vector_search import VectorSearchManager

logger = logging.getLogger(__name__)

class RAG:
    """
    Orchestrator that wires all sub-systems together.
    Keep high-level pipeline methods here; delegate heavy lifting to modules.
    """
    def __init__(self, collection: Dict[str, Any], debug_mode: bool = False):
        # Priority order for thresholds:
        # 1. Use thresholds from collection config if explicitly provided (required for all domains)
        # 2. Use CLIENT_CASE_THRESHOLDS for client cases
        # 3. Fall back to us_constitution thresholds as last resort (should not happen if config is correct)
        document_type = collection.get("document_type", "")

        if "thresholds" in collection and collection["thresholds"]:
            # Use domain-specific thresholds from collection config
            thresholds = collection["thresholds"]
            logger.info(f"Using domain-specific thresholds from collection config for {document_type}")
        elif document_type == "client_case":
            thresholds = CLIENT_CASE_THRESHOLDS
            logger.info(f"Using CLIENT_CASE_THRESHOLDS for {document_type}")
        else:
            # Fallback to us_constitution thresholds (should not happen if all domains are properly configured)
            from services.rag.config import DOMAIN_THRESHOLDS
            thresholds = DOMAIN_THRESHOLDS.get("us_constitution", {})
            logger.warning(f"No thresholds found in collection config for {document_type}, using us_constitution thresholds as fallback. Please add domain-specific thresholds to DOMAIN_THRESHOLDS.")

        self.config = {
            "thresholds": thresholds,
            "embedding_model": EMBEDDING_MODEL,
            "rephrase_limit": REPHRASE_LIMIT,
            "top_k": TOP_QUERY_RESULT,
            "KEYWORD_MATCH_SCORE": KEYWORD_MATCH_SCORE,
            **collection,
        }
        self.sql: bool = collection["sql_attached"]
        self.debug_mode = debug_mode
        # Core services
        self.db = MongoManager(self.config)
        self.query_manager = QueryManager(self.config, self.db.query, self.sql)
        if not self.sql:
            logger.info("sql is not attached")
            self.vector_search = VectorSearchManager(self.config, self.db)  # MongoDB Atlas Vector Search
            # AliasManager is configurable per collection (only enabled for US Constitution for now)
            use_alias_search = self.config.get("use_alias_search", False)
            if use_alias_search:
                self.alias = AliasManager(self.db, self.config)
                logger.info("AliasManager enabled for collection (use_alias_search=True)")
            else:
                self.alias = None
                logger.info("AliasManager disabled for collection (use_alias_search=False)")
            # KeywordMatcher is configurable per collection (useful for US Constitution with structured articles/sections)
            # Disable for large collections to avoid timeout issues
            use_keyword_matcher = self.config.get("use_keyword_matcher", False)
            if use_keyword_matcher:
                self.keyword = KeywordMatcher(self.db)
            else:
                self.keyword = None
                logger.info("KeywordMatcher disabled for collection (use_keyword_matcher=False)")
        self.llmv = LLMVerifier(self.config)
        # Pipeline processor (depends on everything above)
        self.processor = QueryProcessor(self, self.debug_mode)
        logger.info(
            "RAG initialized for document_type=%s",
            self.config.get("document_type", "unknown")
        )
        # self.result_list: List[Tuple[dict, float]] = []
        # self.last_query: Optional[str] = None
    # ----------------- public API -----------------
    def process_query(self, query: str, filtered_cases: Optional[List[str]] = None, jurisdiction: Optional[str] = None, language: str = "en", skip_pre_checks: bool = False, skip_cases_search: bool = False):
        if self.sql:
            if not filtered_cases:
                logger.error("[RAG][process_query] SQL path requires non-empty filtered_cases.")
                raise ValueError("filtered_cases must be provided and non-empty when sql=True.")
            results = self.processor.process_query_light(filtered_cases, query)
            # SQL/light path returns LIST ONLY
            return results

        # non-SQL path unchanged (tuple)
        results, current_query = self.processor.process_query(query, jurisdiction=jurisdiction, language=language, skip_pre_checks=skip_pre_checks, skip_cases_search=skip_cases_search)
        return results, current_query

    def process_summary(self, query: str, result_list: List[Tuple[dict, float]], index: int, language: str = "en", query_en: Optional[str] = None) -> str:
        # Use the cached list + last query
        if self.sql:
            return self.processor.get_summary(
            query=query,
            result_list=result_list,
            index=index,
        )
        # For backward compatibility, return string (requested language)
        # The insight generation function handles bilingual internally
        # Pass pre-translated English query to avoid redundant translation
        return self.processor.get_or_create_insight_by_index(
            query=query,
            result_list=result_list,
            index=index,
            language=language,
            query_en=query_en,
        )

    def process_summary_bilingual(self, query: str, result_list: List[Tuple[dict, float]], index: int, language: str = "en") -> Tuple[str, Optional[str]]:
        """
        Get both English and Spanish insights.
        Returns: (insight_en, insight_es) tuple
        """
        if self.sql:
            # SQL path doesn't support bilingual yet
            insight = self.processor.get_summary(
                query=query,
                result_list=result_list,
                index=index,
            )
            return (insight, None)

        # Get insights - the function will generate both if language is es
        insight = self.processor.get_or_create_insight_by_index(
            query=query,
            result_list=result_list,
            index=index,
            language=language,
        )

        # For English requests, only return English
        if language == "en":
            return (insight, None)

        # For Spanish requests, we need to fetch both from cache
        # The insight generation already stored both, so we can retrieve them
        # For now, return the Spanish insight and try to get English from cache
        # This is a limitation - we'd need to modify get_or_create_insight_by_index to return both
        # For now, return Spanish and None for English (can be improved later)
        return (None, insight)



