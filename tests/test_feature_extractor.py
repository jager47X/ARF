"""Unit tests for FeatureExtractor — pure logic, no MongoDB required."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_dependencies.feature_extractor import (
    _FEATURE_NAMES,
    DOMAIN_ENCODING,
    FeatureExtractor,
    _bias_for_document,
    _compute_bm25_score,
    _cosine_similarity,
    _get_document_text,
    _has_nested,
    _jaccard_similarity,
    _section_depth,
    _tokenize,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

US_CONST_CONFIG = {
    "bias": {"The U.S. Constitution": -0.2, "Second Amendment": -0.05},
    "field_mapping": {
        "title": "title",
        "article": "article",
        "section": "section",
        "chapter": None,
        "part": None,
        "subchapter": None,
        "text": ["text", "summary", "content", "body"],
        "nested_text": ["clauses", "sections"],
    },
}

CFR_CONFIG = {
    "bias": {},
    "field_mapping": {
        "title": "title",
        "article": "article",
        "part": "part",
        "chapter": "chapter",
        "subchapter": "subchapter",
        "section": "section",
        "text": ["text", "summary", "content", "body"],
        "nested_text": ["sections"],
    },
}


def _make_doc(**kwargs):
    """Build a minimal document dict."""
    defaults = {"title": "Test Document", "text": "Some legal text about rights."}
    defaults.update(kwargs)
    return defaults


@pytest.fixture
def extractor_us_const():
    return FeatureExtractor(config=US_CONST_CONFIG, domain="us_constitution")


@pytest.fixture
def extractor_cfr():
    return FeatureExtractor(config=CFR_CONFIG, domain="code_of_federal_regulations")


# ---------------------------------------------------------------------------
# Test individual feature extraction
# ---------------------------------------------------------------------------


class TestSemanticScore:
    def test_passthrough(self, extractor_us_const):
        feats = extractor_us_const.extract_features("query", _make_doc(), semantic_score=0.87)
        assert feats["semantic_score"] == pytest.approx(0.87)

    def test_zero_when_none(self, extractor_us_const):
        feats = extractor_us_const.extract_features("query", _make_doc(), semantic_score=None)
        assert feats["semantic_score"] == pytest.approx(0.0)


class TestBM25Score:
    def test_positive_for_matching_terms(self, extractor_us_const):
        doc = _make_doc(text="The right to bear arms is a fundamental right.")
        feats = extractor_us_const.extract_features("right to bear arms", doc, semantic_score=0.5)
        assert feats["bm25_score"] > 0.0

    def test_zero_for_no_overlap(self, extractor_us_const):
        doc = _make_doc(text="Quantum mechanics describes subatomic particles.")
        feats = extractor_us_const.extract_features("right to bear arms", doc, semantic_score=0.5)
        assert feats["bm25_score"] == pytest.approx(0.0)

    def test_includes_nested_text(self, extractor_us_const):
        doc = _make_doc(
            text="Article I.",
            clauses=[{"text": "Congress shall make no law abridging free speech."}],
        )
        feats = extractor_us_const.extract_features("free speech", doc, semantic_score=0.5)
        assert feats["bm25_score"] > 0.0


class TestAliasMatch:
    def test_true_when_match(self, extractor_us_const):
        doc = _make_doc(title="Second Amendment")
        alias_matches = [("2A", "Second Amendment", 0.95)]
        feats = extractor_us_const.extract_features("second amendment", doc, 0.5, alias_matches=alias_matches)
        assert feats["alias_match"] == 1

    def test_false_when_no_match(self, extractor_us_const):
        doc = _make_doc(title="First Amendment")
        alias_matches = [("2A", "Second Amendment", 0.95)]
        feats = extractor_us_const.extract_features("first amendment", doc, 0.5, alias_matches=alias_matches)
        assert feats["alias_match"] == 0

    def test_false_when_none(self, extractor_us_const):
        feats = extractor_us_const.extract_features("query", _make_doc(), 0.5, alias_matches=None)
        assert feats["alias_match"] == 0


class TestKeywordMatch:
    def test_true_when_match(self, extractor_us_const):
        doc = _make_doc(title="Powers of Congress")
        feats = extractor_us_const.extract_features("powers", doc, 0.5, keyword_matches=["Powers of Congress"])
        assert feats["keyword_match"] == 1

    def test_false_when_no_match(self, extractor_us_const):
        doc = _make_doc(title="First Amendment")
        feats = extractor_us_const.extract_features("first", doc, 0.5, keyword_matches=["Powers of Congress"])
        assert feats["keyword_match"] == 0


class TestDomainType:
    def test_us_constitution(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), 0.5)
        assert feats["domain_type"] == 0

    def test_cfr(self, extractor_cfr):
        feats = extractor_cfr.extract_features("q", _make_doc(), 0.5)
        assert feats["domain_type"] == 1

    def test_all_domains_encoded(self):
        for domain, expected_id in DOMAIN_ENCODING.items():
            ext = FeatureExtractor(config={}, domain=domain)
            feats = ext.extract_features("q", _make_doc(), 0.5)
            assert feats["domain_type"] == expected_id

    def test_unknown_domain(self):
        ext = FeatureExtractor(config={}, domain="unknown_domain")
        feats = ext.extract_features("q", _make_doc(), 0.5)
        assert feats["domain_type"] == -1


class TestDocumentLength:
    def test_log_scaled(self, extractor_us_const):
        text = "a" * 1000
        doc = _make_doc(text=text)
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["document_length"] == pytest.approx(math.log1p(1000), rel=1e-3)

    def test_empty_text(self, extractor_us_const):
        doc = _make_doc(text="")
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["document_length"] == pytest.approx(0.0)


class TestQueryLength:
    def test_char_count(self, extractor_us_const):
        feats = extractor_us_const.extract_features("what is the first amendment", _make_doc(), 0.5)
        assert feats["query_length"] == len("what is the first amendment")


class TestSectionDepth:
    def test_full_hierarchy(self, extractor_cfr):
        doc = _make_doc(
            article="Article I",
            chapter="Chapter 1",
            subchapter="Subchapter A",
            part="Part 100",
            section="Section 1",
        )
        feats = extractor_cfr.extract_features("q", doc, 0.5)
        # title + article + chapter + subchapter + part + section = 6
        assert feats["section_depth"] == 6

    def test_partial_hierarchy(self, extractor_us_const):
        doc = _make_doc(title="14th Amendment", article="Amendment XIV", section="Section 1")
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        # title + article + section = 3 (chapter/part/subchapter are None in mapping)
        assert feats["section_depth"] == 3

    def test_empty_document(self, extractor_us_const):
        doc = {}
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["section_depth"] == 0


class TestEmbeddingCosineSimilarity:
    def test_identical_vectors(self, extractor_us_const):
        vec = [1.0, 0.0, 0.0]
        feats = extractor_us_const.extract_features("q", _make_doc(), 0.5, query_embedding=vec, doc_embedding=vec)
        assert feats["embedding_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self, extractor_us_const):
        feats = extractor_us_const.extract_features(
            "q",
            _make_doc(),
            0.5,
            query_embedding=[1.0, 0.0, 0.0],
            doc_embedding=[0.0, 1.0, 0.0],
        )
        assert feats["embedding_cosine_similarity"] == pytest.approx(0.0, abs=1e-6)

    def test_none_embedding(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), 0.5, query_embedding=None, doc_embedding=None)
        assert feats["embedding_cosine_similarity"] == pytest.approx(0.0)


class TestMatchType:
    def test_exact_keyword(self, extractor_us_const):
        doc = _make_doc(title="Second Amendment")
        feats = extractor_us_const.extract_features("second amendment", doc, 0.5, keyword_matches=["Second Amendment"])
        assert feats["match_type"] == 2

    def test_partial_keyword(self, extractor_us_const):
        doc = _make_doc(title="14th Amendment Section 1")
        feats = extractor_us_const.extract_features("14th amendment", doc, 0.5, keyword_matches=["14th Amendment"])
        assert feats["match_type"] == 1

    def test_no_match(self, extractor_us_const):
        doc = _make_doc(title="Supremacy Clause")
        feats = extractor_us_const.extract_features("quantum physics", doc, 0.5, keyword_matches=[], alias_matches=[])
        assert feats["match_type"] == 0

    def test_exact_alias(self, extractor_us_const):
        doc = _make_doc(title="Second Amendment")
        feats = extractor_us_const.extract_features("2A", doc, 0.5, alias_matches=[("2A", "Second Amendment", 0.95)])
        assert feats["match_type"] == 2


class TestScoreGapFromTop:
    def test_gap_computation(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), semantic_score=0.7, top_score=0.9)
        assert feats["score_gap_from_top"] == pytest.approx(0.2, abs=1e-6)

    def test_top_result_gap_zero(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), semantic_score=0.9, top_score=0.9)
        assert feats["score_gap_from_top"] == pytest.approx(0.0)

    def test_no_top_score(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), semantic_score=0.7, top_score=None)
        assert feats["score_gap_from_top"] == pytest.approx(0.0)


class TestQueryTermCoverage:
    def test_full_coverage(self, extractor_us_const):
        doc = _make_doc(text="the right to bear arms")
        feats = extractor_us_const.extract_features("right to bear arms", doc, 0.5)
        assert feats["query_term_coverage"] == pytest.approx(1.0)

    def test_partial_coverage(self, extractor_us_const):
        doc = _make_doc(text="the right to free speech")
        feats = extractor_us_const.extract_features("right to bear arms", doc, 0.5)
        # "right" and "to" match, "bear" and "arms" don't => 2/4 = 0.5
        assert feats["query_term_coverage"] == pytest.approx(0.5)

    def test_no_coverage(self, extractor_us_const):
        doc = _make_doc(text="quantum mechanics")
        feats = extractor_us_const.extract_features("right to bear arms", doc, 0.5)
        assert feats["query_term_coverage"] == pytest.approx(0.0)


class TestTitleSimilarity:
    def test_identical(self, extractor_us_const):
        doc = _make_doc(title="Second Amendment")
        feats = extractor_us_const.extract_features("second amendment", doc, 0.5)
        assert feats["title_similarity"] == pytest.approx(1.0)

    def test_no_overlap(self, extractor_us_const):
        doc = _make_doc(title="Supremacy Clause")
        feats = extractor_us_const.extract_features("quantum physics", doc, 0.5)
        assert feats["title_similarity"] == pytest.approx(0.0)

    def test_partial(self, extractor_us_const):
        doc = _make_doc(title="First Amendment")
        feats = extractor_us_const.extract_features("first amendment rights", doc, 0.5)
        # tokens: {first, amendment} & {first, amendment, rights} => 2/3
        assert feats["title_similarity"] == pytest.approx(2.0 / 3.0, abs=0.01)


class TestHasNestedContent:
    def test_with_clauses(self, extractor_us_const):
        doc = _make_doc(clauses=[{"text": "clause 1"}])
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["has_nested_content"] == 1

    def test_with_sections(self, extractor_us_const):
        doc = _make_doc(sections=[{"text": "section 1"}])
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["has_nested_content"] == 1

    def test_no_nested(self, extractor_us_const):
        doc = _make_doc()
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["has_nested_content"] == 0

    def test_empty_list(self, extractor_us_const):
        doc = _make_doc(clauses=[])
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["has_nested_content"] == 0


class TestBiasAdjustment:
    def test_known_bias(self, extractor_us_const):
        doc = _make_doc(title="Second Amendment")
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["bias_adjustment"] == pytest.approx(-0.05)

    def test_no_bias(self, extractor_us_const):
        doc = _make_doc(title="First Amendment")
        feats = extractor_us_const.extract_features("q", doc, 0.5)
        assert feats["bias_adjustment"] == pytest.approx(0.0)

    def test_empty_bias_map(self, extractor_cfr):
        doc = _make_doc(title="Something")
        feats = extractor_cfr.extract_features("q", doc, 0.5)
        assert feats["bias_adjustment"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test batch extraction
# ---------------------------------------------------------------------------


class TestExtractBatch:
    def test_batch_returns_correct_count(self, extractor_us_const):
        results = [
            (_make_doc(title="Doc A"), 0.9),
            (_make_doc(title="Doc B"), 0.8),
            (_make_doc(title="Doc C"), 0.7),
        ]
        batch = extractor_us_const.extract_batch("some query", results)
        assert len(batch) == 3

    def test_batch_top_score_propagated(self, extractor_us_const):
        results = [
            (_make_doc(title="Doc A"), 0.9),
            (_make_doc(title="Doc B"), 0.7),
        ]
        batch = extractor_us_const.extract_batch("query", results)
        # First doc is the top => gap = 0
        assert batch[0]["score_gap_from_top"] == pytest.approx(0.0)
        # Second doc gap = 0.9 - 0.7 = 0.2
        assert batch[1]["score_gap_from_top"] == pytest.approx(0.2, abs=1e-6)

    def test_batch_empty_results(self, extractor_us_const):
        batch = extractor_us_const.extract_batch("query", [])
        assert batch == []

    def test_batch_with_keyword_and_alias(self, extractor_us_const):
        results = [
            (_make_doc(title="Second Amendment"), 0.9),
        ]
        batch = extractor_us_const.extract_batch(
            "2nd amendment",
            results,
            keyword_matches=["Second Amendment"],
            alias_matches=[("2A", "Second Amendment", 0.95)],
        )
        assert batch[0]["keyword_match"] == 1
        assert batch[0]["alias_match"] == 1


# ---------------------------------------------------------------------------
# Test vector conversion
# ---------------------------------------------------------------------------


class TestToVector:
    def test_correct_length(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), 0.5)
        vec = extractor_us_const.to_vector(feats)
        assert len(vec) == len(_FEATURE_NAMES)

    def test_order_matches_feature_names(self, extractor_us_const):
        feats = extractor_us_const.extract_features(
            "second amendment",
            _make_doc(title="Second Amendment"),
            semantic_score=0.87,
        )
        vec = extractor_us_const.to_vector(feats)
        names = FeatureExtractor.feature_names()
        # First element should be semantic_score
        assert names[0] == "semantic_score"
        assert vec[0] == pytest.approx(0.87)

    def test_missing_keys_default_to_zero(self, extractor_us_const):
        incomplete = {"semantic_score": 0.5}
        vec = extractor_us_const.to_vector(incomplete)
        assert vec[0] == pytest.approx(0.5)
        # All others should be 0.0
        for v in vec[1:]:
            assert v == pytest.approx(0.0)

    def test_all_floats(self, extractor_us_const):
        feats = extractor_us_const.extract_features("q", _make_doc(), 0.5)
        vec = extractor_us_const.to_vector(feats)
        for v in vec:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Test feature_names static method
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_returns_list(self):
        names = FeatureExtractor.feature_names()
        assert isinstance(names, list)
        assert len(names) == 15

    def test_all_strings(self):
        for name in FeatureExtractor.feature_names():
            assert isinstance(name, str)

    def test_no_duplicates(self):
        names = FeatureExtractor.feature_names()
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Test graceful handling of missing/None values
# ---------------------------------------------------------------------------


class TestGracefulDefaults:
    def test_none_query(self, extractor_us_const):
        feats = extractor_us_const.extract_features(None, _make_doc(), 0.5)
        assert isinstance(feats, dict)
        assert feats["query_length"] == 0

    def test_none_document(self, extractor_us_const):
        feats = extractor_us_const.extract_features("query", None, 0.5)
        assert isinstance(feats, dict)
        assert feats["document_length"] == pytest.approx(0.0)

    def test_empty_document(self, extractor_us_const):
        feats = extractor_us_const.extract_features("query", {}, 0.5)
        assert isinstance(feats, dict)
        assert feats["section_depth"] == 0

    def test_none_config(self):
        ext = FeatureExtractor(config=None, domain="us_constitution")
        feats = ext.extract_features("query", _make_doc(), 0.5)
        assert isinstance(feats, dict)


# ---------------------------------------------------------------------------
# Test helper functions directly
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_tokenize(self):
        assert _tokenize("Hello World!") == ["hello", "world"]
        assert _tokenize("") == []
        assert _tokenize(None) == []

    def test_jaccard_identical(self):
        assert _jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        assert _jaccard_similarity("hello world", "foo bar") == pytest.approx(0.0)

    def test_jaccard_partial(self):
        # {hello, world} & {hello, foo} = {hello} / {hello, world, foo} = 1/3
        assert _jaccard_similarity("hello world", "hello foo") == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_cosine_similarity_identical(self):
        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_opposite(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_similarity_none(self):
        assert _cosine_similarity(None, [1, 0]) == pytest.approx(0.0)

    def test_bm25_basic(self):
        score = _compute_bm25_score("legal rights", "legal rights are fundamental in the constitution")
        assert score > 0.0

    def test_bm25_no_match(self):
        score = _compute_bm25_score("quantum", "legal rights are fundamental")
        assert score == pytest.approx(0.0)

    def test_get_document_text_fallback(self):
        assert _get_document_text({"summary": "hello"}) == "hello"
        assert _get_document_text({"body": "world"}) == "world"
        assert _get_document_text({}) == ""

    def test_has_nested_true(self):
        assert _has_nested({"clauses": [{"text": "x"}]}) is True

    def test_has_nested_false(self):
        assert _has_nested({"clauses": []}) is False
        assert _has_nested({}) is False

    def test_section_depth_no_mapping(self):
        doc = {"title": "T", "article": "A", "section": "S"}
        assert _section_depth(doc) == 3

    def test_bias_for_document_found(self):
        assert _bias_for_document({"title": "X"}, {"X": -0.1}) == pytest.approx(-0.1)

    def test_bias_for_document_missing(self):
        assert _bias_for_document({"title": "Y"}, {"X": -0.1}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test documents from different domains
# ---------------------------------------------------------------------------


class TestCrossDomain:
    def test_us_code_domain(self):
        config = {
            "bias": {},
            "field_mapping": {
                "title": "title",
                "article": "article",
                "chapter": "chapter",
                "section": "section",
                "part": None,
                "subchapter": None,
                "text": ["text", "summary", "content", "body"],
                "nested_text": ["clauses"],
            },
        }
        ext = FeatureExtractor(config=config, domain="us_code")
        doc = _make_doc(
            title="42 USC 1983",
            article="Title 42",
            chapter="Chapter 21",
            section="Section 1983",
            text="Every person who under color of law deprives any citizen...",
            clauses=[{"text": "Civil action for deprivation of rights."}],
        )
        feats = ext.extract_features("civil rights section 1983", doc, 0.85)
        assert feats["domain_type"] == 2
        assert feats["has_nested_content"] == 1
        assert feats["section_depth"] == 4  # title + article + chapter + section
        assert feats["bm25_score"] > 0.0

    def test_uscis_policy_domain(self):
        config = {
            "bias": {},
            "field_mapping": {
                "title": "title",
                "article": None,
                "chapter": None,
                "section": None,
                "part": None,
                "subchapter": None,
                "text": ["text", "summary", "content", "body"],
                "nested_text": ["clauses"],
            },
        }
        ext = FeatureExtractor(config=config, domain="uscis_policy")
        doc = _make_doc(
            title="Volume 7 - Adjustment of Status",
            text="An applicant for adjustment of status must...",
        )
        feats = ext.extract_features("adjustment of status requirements", doc, 0.78)
        assert feats["domain_type"] == 3
        # Only title is populated (all hierarchy fields are None in mapping)
        assert feats["section_depth"] == 1
        assert feats["query_term_coverage"] > 0.0
