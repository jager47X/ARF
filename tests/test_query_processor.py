"""Unit tests for QueryProcessor — mocked dependencies, no MongoDB/OpenAI required."""

import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_dependencies.query_processor import QueryProcessor

# ---------------------------------------------------------------------------
# Helpers to build a QueryProcessor with fully mocked dependencies
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = {
    "RAG_SEARCH_min": 0.70,
    "RAG_SEARCH": 0.80,
    "LLM_VERIFication": 0.65,
    "alias_search": 0.60,
    "FILTER_GAP": 0.10,
    "LLM_SCORE": 0.75,
    "confident": 0.85,
}

_DEFAULT_CONFIG = {
    "thresholds": _DEFAULT_THRESHOLDS,
    "sql_attached": False,
    "REPHRASE_LIMIT": 0,
    "top_k": 10,
    "document_type": "us_constitution",
    "collection_key": "US_CONSTITUTION_SET",
    "unique_index": "title",
    "KEYWORD_MATCH_SCORE": 0.70,
    "main_collection_name": "us_constitution",
    "bias": {},
}


def _make_processor(config_overrides=None) -> QueryProcessor:
    """Build a QueryProcessor with mocked RAG and sub-services."""
    cfg = {**_DEFAULT_CONFIG}
    if config_overrides:
        cfg.update(config_overrides)
    cfg["thresholds"] = {**_DEFAULT_THRESHOLDS, **(config_overrides or {}).get("thresholds", {})}

    rag = MagicMock()
    rag.sql = cfg["sql_attached"]
    rag.config = cfg
    rag.debug_mode = False

    # Sub-services
    rag.db = MagicMock()
    rag.db.normalize_query = lambda q: " ".join((q or "").lower().split())
    rag.db.find_query_doc_ci = MagicMock(return_value=None)
    rag.db.find_query_doc_by_id = MagicMock(return_value=None)
    rag.db.main = MagicMock()

    rag.query_manager = MagicMock()
    rag.query_manager.openAI = MagicMock()
    rag.query_manager.get_or_create_query_embedding = MagicMock(
        return_value=(np.random.rand(1024).astype(np.float32), False)
    )
    rag.query_manager.get_query_with_results = MagicMock(return_value=None)

    rag.vector_search = MagicMock()
    rag.vector_search.search_main = MagicMock()
    rag.vector_search.search_cases = None
    rag.vector_search.search_main_with_clauses = MagicMock(return_value=None)

    rag.keyword = MagicMock()
    rag.keyword.find_textual = MagicMock(return_value=[])

    rag.alias = MagicMock()
    rag.alias.alias_cache = ["dummy"]
    rag.alias.clean_query = MagicMock(side_effect=lambda q: q)
    rag.alias.find_semantic_aliases = MagicMock(return_value=[])

    rag.llmv = MagicMock()
    rag.llmv.verify_many_parallel = MagicMock(return_value=[])

    # Construct with MLP disabled (avoid import issues in test env)
    cfg["thresholds"]["use_mlp_reranker"] = False
    proc = QueryProcessor(rag, debug_mode=False)
    return proc


def _doc(title, _id=None, article="", section=""):
    """Create a minimal document dict."""
    from bson import ObjectId

    return {
        "_id": _id or ObjectId(),
        "title": title,
        "article": article,
        "section": section,
        "text": f"Text for {title}",
    }


# ===========================================================================
# Tests
# ===========================================================================


class TestProcessQueryCachePath:
    """Test the cache fast-path in process_query."""

    def test_cache_hit_returns_cached_results(self):
        """When query doc has cached results, return them without searching."""
        proc = _make_processor()
        doc1 = _doc("First Amendment")
        cached_result = {
            "knowledge_id": str(doc1["_id"]),
            "title": "First Amendment",
            "score": 0.95,
            "collection_key": "US_CONSTITUTION_SET",
        }
        proc.db.find_query_doc_ci.return_value = {
            "query": "first amendment",
            "embedding": list(np.random.rand(1024)),
            "results": [cached_result],
        }
        # The enrichment batch-fetch returns the full doc
        proc.db.main.find.return_value = [doc1]

        results, query_text = proc.process_query("First Amendment")

        assert len(results) >= 1
        assert results[0][1] == 0.95  # score preserved

    def test_cache_miss_triggers_search(self):
        """When no cached results exist, vector search is called."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        doc1 = _doc("Article I", article="Article I")
        proc.vector_search.search_main_with_clauses.return_value = [
            (doc1, 0.90),
        ]
        # openAI pre-checks
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = True
        proc.openAI.remove_personal_info.return_value = "article i"
        proc.openAI.fix_query.return_value = "article i"

        results, _ = proc.process_query("Article I")

        assert len(results) == 1
        assert results[0][0]["title"] == "Article I"
        assert results[0][1] >= 0.80


class TestProcessQuerySearchPath:
    """Test the main vector search path."""

    def test_high_score_result_accepted(self):
        """Documents scoring above RAG_SEARCH are directly accepted."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        doc1 = _doc("14th Amendment", article="14th Amendment")
        proc.vector_search.search_main_with_clauses.return_value = [
            (doc1, 0.92),
        ]
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = True
        proc.openAI.remove_personal_info.side_effect = lambda q: q
        proc.openAI.fix_query.side_effect = lambda q: q

        results, _ = proc.process_query("14th amendment rights")

        assert len(results) == 1
        assert results[0][1] >= 0.80

    def test_low_score_result_rejected(self):
        """Documents scoring below LLM_VERIF threshold are rejected."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        doc1 = _doc("Obscure Doc")
        proc.vector_search.search_main_with_clauses.return_value = [
            (doc1, 0.30),  # Well below any threshold
        ]
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = True
        proc.openAI.remove_personal_info.side_effect = lambda q: q
        proc.openAI.fix_query.side_effect = lambda q: q

        results, _ = proc.process_query("something unrelated")

        assert len(results) == 0

    def test_empty_search_returns_empty(self):
        """When vector search returns nothing, result is empty."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        proc.vector_search.search_main_with_clauses.return_value = []
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = True
        proc.openAI.remove_personal_info.side_effect = lambda q: q
        proc.openAI.fix_query.side_effect = lambda q: q

        results, _ = proc.process_query("nothing here")

        assert results == []

    def test_multiple_results_sorted_by_score(self):
        """Multiple results should be returned sorted by score descending."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        doc1 = _doc("Amendment 1", article="Amendment 1")
        doc2 = _doc("Amendment 2", article="Amendment 2")
        doc3 = _doc("Amendment 3", article="Amendment 3")
        proc.vector_search.search_main_with_clauses.return_value = [
            (doc2, 0.85),
            (doc1, 0.92),
            (doc3, 0.88),
        ]
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = True
        proc.openAI.remove_personal_info.side_effect = lambda q: q
        proc.openAI.fix_query.side_effect = lambda q: q

        results, _ = proc.process_query("amendments")

        assert len(results) >= 2
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestProcessQueryModeration:
    """Test the safety/moderation layer."""

    def test_flagged_query_returns_empty(self):
        """Queries flagged by moderation return empty results."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        proc.openAI.check_moderation.return_value = {
            "flagged": True,
            "categories": ["violence"],
            "scores": {"violence": 0.99},
        }

        results, _ = proc.process_query("violent query")

        assert results == []

    def test_moderation_failure_continues_search(self):
        """If moderation check fails, search continues as fallback."""
        import openai as openai_mod

        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        proc.openAI.check_moderation.side_effect = openai_mod.APIConnectionError(request=MagicMock())
        proc.openAI.check_us_constitution_relevance.return_value = True
        proc.openAI.remove_personal_info.side_effect = lambda q: q
        proc.openAI.fix_query.side_effect = lambda q: q
        doc1 = _doc("Safe Doc", article="Safe Doc")
        proc.vector_search.search_main_with_clauses.return_value = [
            (doc1, 0.90),
        ]

        results, _ = proc.process_query("safe query")

        # Should still return results despite moderation failure
        assert len(results) >= 1


class TestProcessQueryTopicCheck:
    """Test the US Constitution relevance check."""

    def test_irrelevant_query_no_jurisdiction_requests_it(self):
        """Non-constitutional query without jurisdiction returns jurisdiction request."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = False

        results, _ = proc.process_query("parking ticket in LA")

        assert len(results) == 1
        assert results[0][0].get("_request_jurisdiction") is True

    def test_irrelevant_query_with_jurisdiction_returns_general_info(self):
        """Non-constitutional query with jurisdiction returns general info."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None
        proc.openAI.check_moderation.return_value = {"flagged": False, "categories": [], "scores": {}}
        proc.openAI.check_us_constitution_relevance.return_value = False
        proc.openAI.generate_general_info.return_value = "Here is info about parking in LA."

        results, _ = proc.process_query("parking ticket", jurisdiction="Los Angeles")

        assert len(results) == 1
        assert results[0][0].get("_general_info") is True
        assert "parking" in results[0][0]["text"].lower()


class TestFilterKwAlias:
    """Test the _filter_kw_alias scoring method."""

    def test_keyword_match_boosts_score(self):
        """Documents matching keyword terms get boosted to at least KEYWORD_MATCH_SCORE."""
        proc = _make_processor()
        doc1 = _doc("First Amendment", article="First Amendment")
        proc.kw.find_textual.return_value = ["First Amendment"]
        proc.alias.find_semantic_aliases.return_value = []
        emb = np.random.rand(1024).astype(np.float32)

        accepted, verify = proc._filter_kw_alias(
            current_text="first amendment",
            sem_main=[(doc1, 0.60)],  # Below RAG_SEARCH but keyword match
            current_decay=0.0,
            emb=emb,
        )

        # With keyword match, score should be max(0.60, 0.70) = 0.70
        # 0.70 >= LLM_VERIF (0.65) so goes to verify pool
        assert len(verify) > 0 or len(accepted) > 0

    def test_high_semantic_score_accepted_without_keywords(self):
        """High semantic score passes without keyword match."""
        proc = _make_processor()
        doc1 = _doc("Article V", article="Article V")
        proc.kw.find_textual.return_value = []
        proc.alias.find_semantic_aliases.return_value = []
        emb = np.random.rand(1024).astype(np.float32)

        accepted, verify = proc._filter_kw_alias(
            current_text="amending the constitution",
            sem_main=[(doc1, 0.90)],
            current_decay=0.0,
            emb=emb,
        )

        assert len(accepted) == 1
        assert accepted[0][1] >= 0.80

    def test_very_low_semantic_score_rejected(self):
        """Very low semantic score is rejected even with decay."""
        proc = _make_processor()
        doc1 = _doc("Unrelated")
        proc.kw.find_textual.return_value = []
        proc.alias.find_semantic_aliases.return_value = []
        emb = np.random.rand(1024).astype(np.float32)

        accepted, verify = proc._filter_kw_alias(
            current_text="something",
            sem_main=[(doc1, 0.20)],
            current_decay=0.0,
            emb=emb,
        )

        assert len(accepted) == 0
        assert len(verify) == 0


class TestApplyMainAbcGates:
    """Test the ABC gate logic."""

    def test_no_candidates_returns_none(self):
        """No accepted or verify candidates returns None."""
        proc = _make_processor()
        result = proc._apply_main_abc_gates(
            current_text="test",
            accepted=None,
            need_verify=None,
            apply_gap=True,
        )
        assert result is None

    def test_accepted_above_threshold_passes(self):
        """Accepted docs above RAG_SEARCH pass through."""
        proc = _make_processor()
        doc1 = _doc("14th Amendment")
        result = proc._apply_main_abc_gates(
            current_text="14th amendment",
            accepted=[(doc1, 0.92)],
            need_verify=[],
            apply_gap=True,
        )
        assert result is not None
        assert len(result) == 1
        assert result[0][1] == 0.92

    def test_verify_candidates_go_to_llm(self):
        """Candidates in verify zone trigger LLM verification."""
        proc = _make_processor()
        doc1 = _doc("Some Amendment")
        # LLM verifier returns the doc with boosted score
        proc.llmv.verify_many_parallel.return_value = [(doc1, 0.85)]

        result = proc._apply_main_abc_gates(
            current_text="some amendment",
            accepted=[],
            need_verify=[(doc1, 0.70)],
            apply_gap=True,
        )

        proc.llmv.verify_many_parallel.assert_called_once()
        assert result is not None
        assert len(result) >= 1

    def test_gap_filter_removes_distant_results(self):
        """Gap filter removes results too far from the top score."""
        proc = _make_processor()
        doc1 = _doc("Top Doc")
        doc2 = _doc("Gap Doc")
        result = proc._apply_main_abc_gates(
            current_text="test",
            accepted=[(doc1, 0.95), (doc2, 0.81)],
            need_verify=[],
            apply_gap=True,
        )
        assert result is not None
        # With FILTER_GAP=0.10, doc2 (0.81) is within gap of doc1 (0.95)?
        # 0.95 - 0.81 = 0.14 > 0.10, so doc2 should be filtered out
        scores = [s for _, s in result]
        assert 0.81 not in scores

    def test_merge_deduplicates_by_id(self):
        """Same document from accepted and verified is deduplicated (best score kept)."""
        proc = _make_processor()
        from bson import ObjectId

        shared_id = ObjectId()
        doc1 = _doc("Same Doc", _id=shared_id)
        doc2 = _doc("Same Doc", _id=shared_id)
        proc.llmv.verify_many_parallel.return_value = [(doc2, 0.88)]

        result = proc._apply_main_abc_gates(
            current_text="same doc",
            accepted=[(doc1, 0.85)],
            need_verify=[(doc2, 0.70)],
            apply_gap=True,
        )

        assert result is not None
        # Should have only 1 result (deduplicated), with the higher score
        assert len(result) == 1
        assert result[0][1] == 0.88


class TestGapFilter:
    """Test the _gap_filter helper."""

    def test_empty_list(self):
        proc = _make_processor()
        assert proc._gap_filter([]) == []

    def test_single_item(self):
        proc = _make_processor()
        doc = _doc("Single")
        result = proc._gap_filter([(doc, 0.90)])
        assert len(result) == 1

    def test_filters_distant_results(self):
        proc = _make_processor()
        doc1 = _doc("Close")
        doc2 = _doc("Far")
        result = proc._gap_filter([(doc1, 0.95), (doc2, 0.80)])
        # Gap = 0.95 - 0.80 = 0.15 > FILTER_GAP (0.10)
        assert len(result) == 1
        assert result[0][0]["title"] == "Close"

    def test_keeps_close_results(self):
        proc = _make_processor()
        doc1 = _doc("Top")
        doc2 = _doc("Near")
        result = proc._gap_filter([(doc1, 0.95), (doc2, 0.90)])
        # Gap = 0.05 < FILTER_GAP (0.10)
        assert len(result) == 2


class TestFollowRephrasesOrCached:
    """Test the rephrase chain traversal."""

    def test_no_doc_returns_seed(self):
        """When no query doc exists, returns seed text with no cache."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = None

        state = proc._follow_rephrases_or_cached("test query")

        assert state["final_text"] == "test query"
        assert state["cached_results"] is None
        assert state["hops"] == 0

    def test_cached_results_returned(self):
        """When query doc has results, returns them."""
        proc = _make_processor()
        proc.db.find_query_doc_ci.return_value = {
            "query": "test",
            "results": [{"title": "Result 1", "score": 0.9}],
        }
        proc.qm.get_query_with_results.return_value = [
            {"title": "Result 1", "score": 0.9},
        ]

        state = proc._follow_rephrases_or_cached("test")

        assert state["cached_results"] is not None
        assert len(state["cached_results"]) == 1

    def test_loop_detection(self):
        """Circular rephrase chains are detected and broken."""
        proc = _make_processor()
        from bson import ObjectId

        id_a = ObjectId()
        id_b = ObjectId()

        # A -> B -> A (loop)
        doc_a = {"query": "query a", "results": None, "rephrased_ref": id_b}
        doc_b = {"query": "query b", "results": None, "rephrased_ref": id_a}

        def find_ci(q):
            if "query a" in q:
                return doc_a
            if "query b" in q:
                return doc_b
            return None

        def find_by_id(oid):
            if oid == id_b:
                return doc_b
            if oid == id_a:
                return doc_a
            return None

        proc.db.find_query_doc_ci.side_effect = find_ci
        proc.db.find_query_doc_by_id.side_effect = find_by_id

        state = proc._follow_rephrases_or_cached("query a", max_hops=5)

        assert state["loop_detected"] is True


class TestProcessQueryLight:
    """Test the light path (client case search)."""

    def test_empty_cases_returns_empty(self):
        proc = _make_processor()
        result = proc.process_query_light([], "test query")
        assert result == []
