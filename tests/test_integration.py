"""
Integration tests for the ARF query pipeline.

These tests run against the fixture dataset (no MongoDB required) to verify
that the KeywordMatcher find_textual pipeline returns expected documents for
known queries. They serve as both regression tests and a living benchmark.

For live integration tests against MongoDB, set ARF_LIVE_TESTS=1 and
provide API keys in .env.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FIXTURE_FILE = FIXTURES_DIR / "us_constitution_sample.json"


@pytest.fixture(scope="module")
def fixture_docs():
    with open(FIXTURE_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def keyword_matcher(fixture_docs):
    """Build a KeywordMatcher populated with fixture data (no MongoDB)."""
    from rag_dependencies.keyword_matcher import KeywordMatcher

    articles = list({d["article"] for d in fixture_docs if d.get("article")})
    titles = [d["title"] for d in fixture_docs]

    # Build article->sections map
    article_sections = {}
    for d in fixture_docs:
        art = d.get("article")
        sec = d.get("section")
        if art and sec:
            article_sections.setdefault(art, set()).add(sec)

    # Build matcher without DB
    km = object.__new__(KeywordMatcher)
    km.db = type("FakeDB", (), {"main": type("FC", (), {"aggregate": lambda *a, **k: iter([])})()})()
    km.main = km.db.main
    km._articles = articles
    km._titles = titles
    km._title_lc_to_title = {t.lower(): t for t in titles}
    km._article_lc_to_article = {a.lower(): a for a in articles}
    km.article_to_sections = article_sections
    km._sections_by_article_lc = {art.lower(): {s.lower() for s in secs} for art, secs in article_sections.items()}
    km.important_terms = list({*(articles + titles)})
    km._number_word_map = {
        "first": "1st",
        "second": "2nd",
        "third": "3rd",
        "fourth": "4th",
        "fifth": "5th",
        "sixth": "6th",
        "seventh": "7th",
        "eighth": "8th",
        "ninth": "9th",
        "tenth": "10th",
        "eleventh": "11th",
        "twelfth": "12th",
        "thirteenth": "13th",
        "fourteenth": "14th",
        "fifteenth": "15th",
        "sixteenth": "16th",
        "seventeenth": "17th",
        "eighteenth": "18th",
        "nineteenth": "19th",
        "twentieth": "20th",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "twenty one": "21",
        "twenty two": "22",
        "twenty three": "23",
        "twenty four": "24",
        "twenty five": "25",
        "twenty six": "26",
    }
    return km


# ---- Integration tests: query -> expected doc in results ----


class TestKeywordMatcherIntegration:
    """Given a query, verify the expected document title appears in find_textual results."""

    @pytest.mark.parametrize(
        "query, expected_titles",
        [
            ("14th amendment", ["14th Amendment Section 1", "Amendment"]),
            ("first amendment", ["1st Amendment"]),
            ("second amendment", ["2nd Amendment"]),
            ("4th amendment", ["4th Amendment"]),
            ("fifth amendment", ["5th Amendment"]),
            ("10th amendment", ["10th Amendment"]),
            ("13th amendment", ["13th Amendment Section 1", "Amendment"]),
            ("supremacy clause", ["Supremacy Clause"]),
            ("powers of congress", ["Powers of Congress"]),
        ],
    )
    def test_keyword_finds_expected(self, keyword_matcher, query, expected_titles):
        results = keyword_matcher.find_textual(query)
        found_any = any(t in results for t in expected_titles)
        assert found_any, f"Expected one of {expected_titles} in results for query '{query}', got: {results}"


class TestFixtureDataIntegrity:
    """Validate the fixture dataset is well-formed."""

    def test_all_docs_have_title(self, fixture_docs):
        for doc in fixture_docs:
            assert doc.get("title"), f"Document missing title: {doc}"

    def test_all_docs_have_text(self, fixture_docs):
        for doc in fixture_docs:
            assert doc.get("text"), f"Document missing text: {doc.get('title')}"

    def test_no_duplicate_titles(self, fixture_docs):
        titles = [d["title"] for d in fixture_docs]
        assert len(titles) == len(set(titles)), "Duplicate titles found"

    def test_minimum_doc_count(self, fixture_docs):
        assert len(fixture_docs) >= 10, "Fixture should have at least 10 documents"


# ---- Live integration tests (require API keys + MongoDB) ----

LIVE = os.getenv("ARF_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="Set ARF_LIVE_TESTS=1 to run live tests")
class TestLiveIntegration:
    """
    Live integration tests against MongoDB.
    Run with: ARF_LIVE_TESTS=1 pytest tests/test_integration.py -v
    """

    @pytest.fixture(scope="class")
    def rag(self):
        from config import COLLECTION
        from RAG_interface import RAG

        return RAG(COLLECTION["US_CONSTITUTION_SET"], debug_mode=False)

    @pytest.mark.parametrize(
        "query, expected_title",
        [
            ("14th Amendment equal protection", "14th Amendment Section 1"),
            ("freedom of speech", "1st Amendment"),
            ("right to bear arms", "2nd Amendment"),
        ],
    )
    def test_live_retrieval(self, rag, query, expected_title):
        results, _ = rag.process_query(query, language="en")
        titles = [doc.get("title", "") for doc, score in results]
        assert expected_title in titles, (
            f"Expected '{expected_title}' in live results for '{query}', got top-5: {titles[:5]}"
        )
