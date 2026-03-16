"""Tests for data integrity of Knowledge JSON files and the evaluation test set."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
KNOWLEDGE_DIR = DATA_DIR / "Knowledge"
TEST_SET_CSV = DATA_DIR / "us-con-test-set.csv"

# Collect all JSON files in Knowledge/
KNOWLEDGE_JSON_FILES = sorted(KNOWLEDGE_DIR.glob("*.json")) if KNOWLEDGE_DIR.exists() else []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_documents(data: dict) -> list[dict]:
    """Walk common data layouts and return the list of document/article dicts."""
    # Layout 1: {"data": {"<topic>": {"articles": [...]}}}
    # Layout 2: {"data": {"<topic>": {"documents": [...]}}}
    # Layout 3: {"cases": {"<category>": [...]}}
    docs: list[dict] = []
    if "data" in data:
        for topic_val in data["data"].values():
            if isinstance(topic_val, dict):
                for key in ("articles", "documents"):
                    if key in topic_val and isinstance(topic_val[key], list):
                        docs.extend(topic_val[key])
    if "cases" in data:
        for category_list in data["cases"].values():
            if isinstance(category_list, list):
                docs.extend(category_list)
    return docs


# ===========================================================================
# 1. Knowledge JSON file-level tests
# ===========================================================================

class TestKnowledgeFilesLoadable:
    """Every JSON file in data/Knowledge/ must be valid JSON."""

    @pytest.mark.parametrize("json_path", KNOWLEDGE_JSON_FILES, ids=lambda p: p.name)
    def test_valid_json(self, json_path):
        data = _load_json(json_path)
        assert isinstance(data, (dict, list)), f"{json_path.name} root is not dict or list"

    @pytest.mark.parametrize("json_path", KNOWLEDGE_JSON_FILES, ids=lambda p: p.name)
    def test_not_empty(self, json_path):
        data = _load_json(json_path)
        if isinstance(data, dict):
            assert len(data) > 0, f"{json_path.name} is an empty dict"
        else:
            assert len(data) > 0, f"{json_path.name} is an empty list"


# ===========================================================================
# 2. Document-level schema tests for legal knowledge files
# ===========================================================================

# Files that use the standard {data: {topic: {articles/documents: [...]}}} layout
STANDARD_KNOWLEDGE_FILES = [
    p for p in KNOWLEDGE_JSON_FILES
    if p.name not in ("supreme_court_cases.json",)
]


class TestKnowledgeDocumentSchema:
    """Documents inside standard knowledge files must have required fields."""

    @pytest.mark.parametrize("json_path", STANDARD_KNOWLEDGE_FILES, ids=lambda p: p.name)
    def test_documents_have_title(self, json_path):
        data = _load_json(json_path)
        docs = _extract_documents(data)
        if not docs:
            pytest.skip(f"{json_path.name} has no documents to validate")
        for i, doc in enumerate(docs):
            assert "title" in doc, f"{json_path.name} doc #{i} missing 'title'"
            assert doc["title"], f"{json_path.name} doc #{i} has empty title"

    @pytest.mark.parametrize("json_path", STANDARD_KNOWLEDGE_FILES, ids=lambda p: p.name)
    def test_documents_have_clauses(self, json_path):
        data = _load_json(json_path)
        docs = _extract_documents(data)
        if not docs:
            pytest.skip(f"{json_path.name} has no documents to validate")
        for i, doc in enumerate(docs):
            assert "clauses" in doc, f"{json_path.name} doc #{i} ('{doc.get('title')}') missing 'clauses'"
            assert isinstance(doc["clauses"], list), (
                f"{json_path.name} doc #{i} 'clauses' is not a list"
            )

    @pytest.mark.parametrize("json_path", STANDARD_KNOWLEDGE_FILES, ids=lambda p: p.name)
    def test_clauses_have_text(self, json_path):
        data = _load_json(json_path)
        docs = _extract_documents(data)
        if not docs:
            pytest.skip(f"{json_path.name} has no documents to validate")
        for doc in docs:
            for clause in doc.get("clauses", []):
                assert "text" in clause, (
                    f"{json_path.name} clause in '{doc.get('title')}' missing 'text'"
                )
                assert clause["text"].strip(), (
                    f"{json_path.name} clause in '{doc.get('title')}' has empty text"
                )


# ===========================================================================
# 3. Supreme Court cases schema
# ===========================================================================

SUPREME_COURT_FILE = KNOWLEDGE_DIR / "supreme_court_cases.json"


@pytest.mark.skipif(not SUPREME_COURT_FILE.exists(), reason="supreme_court_cases.json not found")
class TestSupremeCourtCases:
    @pytest.fixture(scope="class")
    def cases_data(self):
        return _load_json(SUPREME_COURT_FILE)

    def test_has_cases_key(self, cases_data):
        assert "cases" in cases_data

    def test_categories_non_empty(self, cases_data):
        for category, case_list in cases_data["cases"].items():
            assert isinstance(case_list, list), f"Category '{category}' is not a list"
            assert len(case_list) > 0, f"Category '{category}' is empty"

    def test_each_case_has_required_fields(self, cases_data):
        for category, case_list in cases_data["cases"].items():
            for case in case_list:
                assert "case" in case, f"Case in '{category}' missing 'case' field"
                assert "summary" in case, (
                    f"Case '{case.get('case')}' in '{category}' missing 'summary'"
                )

    def test_references_are_lists(self, cases_data):
        for category, case_list in cases_data["cases"].items():
            for case in case_list:
                if "references" in case:
                    assert isinstance(case["references"], list), (
                        f"Case '{case.get('case')}' references is not a list"
                    )


# ===========================================================================
# 4. US Constitution test set (CSV)
# ===========================================================================

@pytest.mark.skipif(not TEST_SET_CSV.exists(), reason="us-con-test-set.csv not found")
class TestUSConTestSet:
    @pytest.fixture(scope="class")
    def rows(self):
        with open(TEST_SET_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_has_rows(self, rows):
        assert len(rows) >= 50, f"Expected at least 50 test queries, got {len(rows)}"

    def test_required_columns(self, rows):
        required = {"Question", "Case", "Related to"}
        assert required.issubset(rows[0].keys()), (
            f"Missing columns: {required - rows[0].keys()}"
        )

    def test_questions_non_empty(self, rows):
        for i, row in enumerate(rows):
            assert row["Question"].strip(), f"Row {i+2} has empty Question"

    def test_case_values_are_numeric(self, rows):
        for i, row in enumerate(rows):
            assert row["Case"].strip().isdigit(), (
                f"Row {i+2} Case='{row['Case']}' is not a digit"
            )

    def test_case_values_in_expected_range(self, rows):
        for i, row in enumerate(rows):
            val = int(row["Case"].strip())
            assert 1 <= val <= 5, f"Row {i+2} Case={val} outside expected range 1-5"

    def test_related_to_non_empty(self, rows):
        for i, row in enumerate(rows):
            assert row["Related to"].strip(), f"Row {i+2} has empty 'Related to'"

    def test_no_duplicate_questions(self, rows):
        questions = [r["Question"].strip() for r in rows]
        seen = set()
        dupes = []
        for q in questions:
            if q in seen:
                dupes.append(q)
            seen.add(q)
        assert not dupes, f"Duplicate questions found: {dupes[:5]}"


# ===========================================================================
# 5. Cross-validation: test set topics vs knowledge base
# ===========================================================================

US_CON_LAW_FILE = KNOWLEDGE_DIR / "us_con_law.json"


@pytest.mark.skipif(
    not (TEST_SET_CSV.exists() and US_CON_LAW_FILE.exists()),
    reason="Requires both us-con-test-set.csv and us_con_law.json",
)
class TestCrossValidation:
    @pytest.fixture(scope="class")
    def test_set_topics(self):
        """Collect all unique 'Related to' topics from the test set."""
        topics = set()
        with open(TEST_SET_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for topic in row["Related to"].split(","):
                    t = topic.strip().lower()
                    if t and t != "none":
                        topics.add(t)
        return topics

    @pytest.fixture(scope="class")
    def knowledge_titles(self):
        """Collect all titles from us_con_law.json."""
        data = _load_json(US_CON_LAW_FILE)
        docs = _extract_documents(data)
        return {d["title"].lower() for d in docs if d.get("title")}

    def test_test_set_has_topics(self, test_set_topics):
        assert len(test_set_topics) > 0

    def test_most_topics_exist_in_knowledge(self, test_set_topics, knowledge_titles):
        """At least 50% of test set topics should match a knowledge title."""
        matched = sum(1 for t in test_set_topics if t in knowledge_titles)
        coverage = matched / len(test_set_topics) if test_set_topics else 0
        assert coverage >= 0.5, (
            f"Only {matched}/{len(test_set_topics)} ({coverage:.0%}) test set topics "
            f"found in knowledge base titles"
        )
