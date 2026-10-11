"""Tests for arf.features — FeatureExtractor and CorpusStats."""

import math

import pytest

from arf.document import DocumentConfig
from arf.features import FEATURE_NAMES, FEATURE_VERSION, CorpusStats, FeatureExtractor


def _sample_doc():
    return {
        "title": "14th Amendment",
        "article": "XIV",
        "section": "1",
        "text": "All persons born or naturalized in the United States are citizens.",
        "clauses": [
            {"text": "No State shall make or enforce any law."},
        ],
    }


def test_extract_features_basic():
    ext = FeatureExtractor()
    doc = _sample_doc()
    features = ext.extract_features("what is due process", doc, 0.85)

    assert isinstance(features, dict)
    assert features["semantic_score"] == 0.85
    assert features["bm25_score"] >= 0
    assert features["query_length"] == pytest.approx(math.log1p(4))  # 4 tokens, log-scaled
    assert features["section_depth"] >= 0
    assert features["has_nested_content"] == 1
    assert features["document_length"] > 0


def test_extract_features_all_16():
    ext = FeatureExtractor()
    features = ext.extract_features("test query", _sample_doc(), 0.75)
    assert len(features) == 16
    for name in FEATURE_NAMES:
        assert name in features
    assert FEATURE_VERSION == 2
    assert FeatureExtractor.feature_version() == FEATURE_VERSION


def test_to_vector_order():
    ext = FeatureExtractor()
    features = ext.extract_features("test", _sample_doc(), 0.9)
    vec = ext.to_vector(features)
    assert len(vec) == 16
    assert vec[0] == features["semantic_score"]
    assert vec[-1] == features["bias_adjustment"]


def test_extract_batch():
    ext = FeatureExtractor()
    results = [
        (_sample_doc(), 0.90),
        ({"title": "3rd Amendment", "text": "No soldier shall..."}, 0.70),
    ]
    batch = ext.extract_batch("quartering soldiers", results)
    assert len(batch) == 2
    assert batch[0]["score_gap_from_top"] == 0.0
    assert batch[1]["score_gap_from_top"] > 0


def test_extract_batch_with_explicit_top_score():
    # When the batch is a subset (e.g. only the review band), the caller passes
    # the query's real top score so the gap is measured against it.
    ext = FeatureExtractor()
    batch = ext.extract_batch("q", [({"title": "A", "text": "..."}, 0.78)], top_score=0.92)
    assert round(batch[0]["score_gap_from_top"], 4) == 0.14


def test_feature_names_static():
    names = FeatureExtractor.feature_names()
    assert names == FEATURE_NAMES
    assert len(names) == 16


# ---------------------------------------------------------------------------
# Feature version 2: BM25 with IDF, cosine fallback, exact phrase match
# ---------------------------------------------------------------------------


_CORPUS = [
    {"title": "3rd Amendment", "text": "No soldier shall be quartered in any house without consent."},
    {"title": "4th Amendment", "text": "The right of the people to be secure against unreasonable searches."},
    {"title": "14th Amendment", "text": "All persons born or naturalized are citizens; equal protection."},
]


def test_corpus_stats_from_documents():
    stats = CorpusStats.from_documents(_CORPUS)
    assert stats.n_docs == 3
    assert stats.doc_freq.get("amendment", 0) == 0  # titles are not part of the text fields
    assert stats.doc_freq["be"] == 2  # "shall be quartered", "to be secure"
    assert stats.doc_freq["unreasonable"] == 1
    assert stats.avgdl == pytest.approx(sum(len(d["text"].split()) for d in _CORPUS) / 3, rel=0.1)
    # Rare terms carry more weight than common ones; unknown terms weigh the most.
    assert stats.idf("unreasonable") > stats.idf("be") > 0
    assert stats.idf("zebra") > stats.idf("unreasonable")


def test_corpus_stats_roundtrip_and_empty():
    stats = CorpusStats.from_documents(_CORPUS)
    assert CorpusStats.from_dict(stats.to_dict()) == stats
    assert CorpusStats().idf("anything") == 1.0  # no corpus → plain TF overlap


def test_bm25_uses_idf_and_corpus_avgdl_when_stats_are_given():
    # In doc 1 both "unreasonable" (rare: df=1) and "be" (common: df=2) occur exactly once.
    doc = _CORPUS[1]
    stats = CorpusStats.from_documents(_CORPUS)
    plain_ext = FeatureExtractor()
    idf_ext = FeatureExtractor(corpus_stats=stats)

    def bm25(ext, query):
        return ext.extract_features(query, doc, 0.5)["bm25_score"]

    assert bm25(plain_ext, "unreasonable be") > 0
    assert bm25(idf_ext, "unreasonable be") != bm25(plain_ext, "unreasonable be")
    # With IDF the rare term outweighs the common one ...
    assert bm25(idf_ext, "unreasonable") > bm25(idf_ext, "be")
    # ... without IDF they score the same (same tf, same document).
    assert bm25(plain_ext, "unreasonable") == pytest.approx(bm25(plain_ext, "be"))


def test_cosine_feature_falls_back_to_semantic_score_without_embeddings():
    ext = FeatureExtractor()
    doc = {"title": "A", "text": "..."}
    assert ext.extract_features("q", doc, 0.73)["embedding_cosine_similarity"] == 0.73
    assert ext.extract_features("q", doc, 0.73, query_embedding=[1.0, 0.0])["embedding_cosine_similarity"] == 0.73
    assert (
        ext.extract_features("q", doc, 0.73, query_embedding=[1.0, 0.0], doc_embedding=[0.0, 1.0])[
            "embedding_cosine_similarity"
        ]
        == 0.0
    )


def test_exact_phrase_match():
    ext = FeatureExtractor()
    doc = {"title": "Fifth Amendment", "text": "No person shall be deprived of life without Due-Process of law."}
    assert ext.extract_features("due process", doc, 0.5)["exact_phrase_match"] == 1  # case/punctuation-insensitive
    assert ext.extract_features("process due", doc, 0.5)["exact_phrase_match"] == 0  # order matters
    assert ext.extract_features("fifth amendment", doc, 0.5)["exact_phrase_match"] == 1  # title counts
    assert ext.extract_features("person deprived", doc, 0.5)["exact_phrase_match"] == 0  # not contiguous
    assert ext.extract_features("", doc, 0.5)["exact_phrase_match"] == 0


def test_with_document_config():
    cfg = DocumentConfig(
        title_field="name",
        text_fields=["body"],
        children_fields=["items"],
        hierarchy=["name", "category"],
        domain_id=3,
        bias_map={"Test Doc": -0.1},
    )
    ext = FeatureExtractor(cfg)
    # No "title" key at all — every lookup must go through the configured fields.
    doc = {"name": "Test Doc", "body": "Some content", "category": "Legal"}
    features = ext.extract_features("test doc query", doc, 0.80)
    assert features["domain_type"] == 3
    assert features["bias_adjustment"] == -0.1
    assert features["section_depth"] == 2  # name + category
    assert features["title_similarity"] > 0
    assert features["document_length"] > 0  # body via text_fields


def test_custom_title_field_drives_match_features():
    ext = FeatureExtractor(DocumentConfig(title_field="name", text_fields=["text"]))
    doc = {"name": "Spicy Ramen", "text": "Noodles in a hot broth."}
    features = ext.extract_features(
        "spicy ramen", doc, 0.7, keyword_matches=["Spicy Ramen"], alias_matches=[("ramen", "Spicy Ramen")]
    )
    assert features["keyword_match"] == 1
    assert features["alias_match"] == 1
    assert features["match_type"] == 2


def test_custom_hierarchy_section_depth():
    ext = FeatureExtractor(DocumentConfig(hierarchy=["cuisine", "category", "title"]))
    full = {"cuisine": "Japanese", "category": "Soup", "title": "Ramen", "text": "..."}
    partial = {"cuisine": "Japanese", "title": "Ramen", "text": "..."}
    assert ext.extract_features("q", full, 0.5)["section_depth"] == 3
    assert ext.extract_features("q", partial, 0.5)["section_depth"] == 2
    # Default hierarchy still counts the legal fields of the sample doc.
    assert FeatureExtractor().extract_features("q", _sample_doc(), 0.5)["section_depth"] == 3


def test_custom_embedding_field_in_batch():
    ext = FeatureExtractor(DocumentConfig(embedding_field="vec"))
    batch = ext.extract_batch(
        "q", [({"title": "A", "text": "...", "vec": [1.0, 0.0]}, 0.9)], query_embedding=[1.0, 0.0]
    )
    assert round(batch[0]["embedding_cosine_similarity"], 6) == 1.0


def test_keyword_match_flag():
    ext = FeatureExtractor()
    doc = {"title": "14th Amendment", "text": "..."}
    features = ext.extract_features("14th amendment", doc, 0.80, keyword_matches=["14th Amendment"])
    assert features["keyword_match"] == 1


def test_empty_doc():
    ext = FeatureExtractor()
    features = ext.extract_features("test", {}, 0.5)
    assert features["semantic_score"] == 0.5
    assert features["document_length"] == 0.0
    assert features["has_nested_content"] == 0
