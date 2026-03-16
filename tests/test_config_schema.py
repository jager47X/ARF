"""Tests for Pydantic config schema validation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from pydantic import ValidationError

    from config_schema import CollectionConfig, DomainThresholds, FieldMapping
    _has_pydantic = True
except ImportError:
    _has_pydantic = False

pytestmark = pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")


class TestDomainThresholds:
    def test_valid_thresholds(self):
        t = DomainThresholds(
            query_search=0.65,
            alias_search=0.85,
            RAG_SEARCH_min=0.65,
            LLM_VERIFication=0.70,
            RAG_SEARCH=0.85,
            confident=0.85,
            FILTER_GAP=0.20,
            LLM_SCORE=0.10,
        )
        assert t.RAG_SEARCH == 0.85

    def test_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            DomainThresholds(
                query_search=1.5,  # out of range
                alias_search=0.85,
                RAG_SEARCH_min=0.65,
                LLM_VERIFication=0.70,
                RAG_SEARCH=0.85,
                confident=0.85,
                FILTER_GAP=0.20,
                LLM_SCORE=0.10,
            )

    def test_ordering_violation_rejected(self):
        with pytest.raises(ValidationError, match="RAG_SEARCH_min"):
            DomainThresholds(
                query_search=0.65,
                alias_search=0.85,
                RAG_SEARCH_min=0.90,  # > RAG_SEARCH
                LLM_VERIFication=0.70,
                RAG_SEARCH=0.85,
                confident=0.85,
                FILTER_GAP=0.20,
                LLM_SCORE=0.10,
            )

    def test_llm_verif_ordering_rejected(self):
        with pytest.raises(ValidationError, match="LLM_VERIFication"):
            DomainThresholds(
                query_search=0.65,
                alias_search=0.85,
                RAG_SEARCH_min=0.65,
                LLM_VERIFication=0.95,  # > RAG_SEARCH
                RAG_SEARCH=0.85,
                confident=0.85,
                FILTER_GAP=0.20,
                LLM_SCORE=0.10,
            )

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            DomainThresholds(
                query_search=0.65,
                # alias_search missing
                RAG_SEARCH_min=0.65,
                LLM_VERIFication=0.70,
                RAG_SEARCH=0.85,
                confident=0.85,
                FILTER_GAP=0.20,
                LLM_SCORE=0.10,
            )


class TestCollectionConfig:
    def test_valid_config(self):
        c = CollectionConfig(
            db_name="public",
            query_collection_name="User_queries",
            main_collection_name="us_constitution",
            document_type="US Constitution",
            sql_attached=False,
            thresholds=DomainThresholds(
                query_search=0.65, alias_search=0.85, RAG_SEARCH_min=0.65,
                LLM_VERIFication=0.70, RAG_SEARCH=0.85, confident=0.85,
                FILTER_GAP=0.20, LLM_SCORE=0.10,
            ),
            field_mapping=FieldMapping(title="title"),
        )
        assert c.document_type == "US Constitution"
        assert c.main_vector_index == "vector_index"  # default

    def test_missing_required_rejected(self):
        with pytest.raises(ValidationError):
            CollectionConfig(
                db_name="public",
                # missing main_collection_name, document_type, etc.
            )
