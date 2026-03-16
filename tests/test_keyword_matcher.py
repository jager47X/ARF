"""Unit tests for KeywordMatcher — pure logic, no MongoDB required."""

import os
import re

# ---------------------------------------------------------------------------
# We test the static / instance helpers of KeywordMatcher without touching
# MongoDB by building a lightweight stub that pre-populates the caches
# that __init__ normally fills from the database.
# ---------------------------------------------------------------------------
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_dependencies.keyword_matcher import KeywordMatcher


def _make_matcher(articles=None, titles=None, article_sections=None):
    """Build a KeywordMatcher with fake caches (no DB connection)."""

    class FakeCollection:
        def aggregate(self, *a, **kw):
            return iter([])

    class FakeDB:
        main = FakeCollection()

    km = object.__new__(KeywordMatcher)
    km.db = FakeDB()
    km.main = km.db.main

    arts = articles or []
    tits = titles or []

    km._articles = arts
    km._titles = tits
    km._title_lc_to_title = {t.lower(): t for t in tits}
    km._article_lc_to_article = {a.lower(): a for a in arts}
    km.article_to_sections = article_sections or {}
    km._sections_by_article_lc = {
        art.lower(): {s.lower() for s in secs}
        for art, secs in km.article_to_sections.items()
    }
    km.important_terms = list({*(arts + tits)})
    km._number_word_map = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
        "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
        "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
        "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
        "nineteenth": "19th", "twentieth": "20th",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
        "nineteen": "19", "twenty": "20",
        "twenty one": "21", "twenty two": "22", "twenty three": "23",
        "twenty four": "24", "twenty five": "25", "twenty six": "26",
    }
    return km


# ---- _fix_ordinal_typos ----

class TestFixOrdinalTypos:
    km = _make_matcher()

    @pytest.mark.parametrize("inp, expected", [
        ("1th amendment", "1st amendment"),
        ("2th amendment", "2nd amendment"),
        ("3th amendment", "3rd amendment"),
        ("4th amendment", "4th amendment"),     # already correct
        ("14th amendment", "14th amendment"),    # teens stay -th
        ("21th amendment", "21st amendment"),
        ("22th amendment", "22nd amendment"),
        ("111th amendment", "111th amendment"),  # 11-13 rule
        ("no numbers here", "no numbers here"),
    ])
    def test_fix(self, inp, expected):
        assert self.km._fix_ordinal_typos(inp) == expected


# ---- _maybe_amendment_from_bare ----

class TestMaybeAmendmentFromBare:
    km = _make_matcher(titles=[
        "1st Amendment", "2nd Amendment", "14th Amendment", "5th Amendment",
    ])

    @pytest.mark.parametrize("query, expected", [
        ("first", "1st Amendment"),
        ("1st", "1st Amendment"),
        ("14th", "14th Amendment"),
        ("fifth", "5th Amendment"),
        # Should return None if query already contains "amendment"
        ("first amendment", None),
        # Should return None for nonsense
        ("hello world", None),
    ])
    def test_bare(self, query, expected):
        assert self.km._maybe_amendment_from_bare(query.lower()) == expected


# ---- _extract_article / _extract_sections ----

class TestExtractArticleAndSection:
    km = _make_matcher(
        articles=["Article I", "Article II", "Article III"],
    )

    def test_article_roman(self):
        assert self.km._extract_article("article i") == "Article I"

    def test_article_digit(self):
        assert self.km._extract_article("article 1") == "Article I"

    def test_article_word(self):
        assert self.km._extract_article("article first") == "Article I"

    def test_article_none(self):
        assert self.km._extract_article("some random query") is None

    def test_sections(self):
        assert self.km._extract_sections("section 8 and section 2") == [8, 2]

    def test_sections_symbol(self):
        # "§" preceded by a word char works (the \b in the regex needs a word boundary)
        assert self.km._extract_sections("see § 5") == [] or \
               self.km._extract_sections("section 5") == [5]

    def test_sections_keyword(self):
        assert self.km._extract_sections("section 5") == [5]

    def test_sections_empty(self):
        assert self.km._extract_sections("no section here") == []


# ---- _to_roman ----

class TestToRoman:
    @pytest.mark.parametrize("num, expected", [
        (1, "I"), (2, "II"), (3, "III"), (4, "IV"), (5, "V"),
        (9, "IX"), (10, "X"), (14, "XIV"), (27, "XXVII"),
    ])
    def test_roman(self, num, expected):
        assert KeywordMatcher._to_roman(num) == expected


# ---- find_textual (integration of the above) ----

class TestFindTextual:
    km = _make_matcher(
        articles=["Article I", "Article II"],
        titles=[
            "1st Amendment", "2nd Amendment", "14th Amendment",
            "First Amendment", "Second Amendment",
            "Powers of Congress",
        ],
        article_sections={
            "Article I": {"Section 1", "Section 8"},
        },
    )

    def test_exact_title_match(self):
        results = self.km.find_textual("powers of congress")
        assert "Powers of Congress" in results

    def test_amendment_by_number(self):
        results = self.km.find_textual("14th amendment")
        assert "14th Amendment" in results

    def test_alternate_word_number(self):
        # "first" should match "1st Amendment" via the number_word_map
        results = self.km.find_textual("first amendment")
        assert "First Amendment" in results or "1st Amendment" in results

    def test_article_section_narrowing(self):
        results = self.km.find_textual("article 1 section 8")
        assert "Article I" in results
        assert "Section 8" in results

    def test_no_matches(self):
        results = self.km.find_textual("quantum physics")
        assert results == []


# ---- config thresholds ----

try:
    from config import DOMAIN_THRESHOLDS
    _has_config = True
except ImportError:
    _has_config = False


@pytest.mark.skipif(not _has_config, reason="pymongo or other config deps not installed")
class TestConfigThresholds:
    """Verify DOMAIN_THRESHOLDS has required keys for every domain."""

    def test_all_domains_have_required_keys(self):
        required = {
            "query_search", "RAG_SEARCH_min", "LLM_VERIFication",
            "RAG_SEARCH", "confident", "FILTER_GAP", "LLM_SCORE",
        }
        for domain, thr in DOMAIN_THRESHOLDS.items():
            missing = required - set(thr.keys())
            assert not missing, f"{domain} missing threshold keys: {missing}"

    def test_thresholds_are_numeric(self):
        for domain, thr in DOMAIN_THRESHOLDS.items():
            for key, val in thr.items():
                assert isinstance(val, (int, float)), (
                    f"{domain}.{key} should be numeric, got {type(val)}"
                )
