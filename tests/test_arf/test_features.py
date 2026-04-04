"""Tests for arf.features — FeatureExtractor."""

from arf.document import DocumentConfig
from arf.features import FEATURE_NAMES, FeatureExtractor


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
    assert features["query_length"] == len("what is due process")
    assert features["section_depth"] >= 0
    assert features["has_nested_content"] == 1
    assert features["document_length"] > 0


def test_extract_features_all_15():
    ext = FeatureExtractor()
    features = ext.extract_features("test query", _sample_doc(), 0.75)
    assert len(features) == 15
    for name in FEATURE_NAMES:
        assert name in features


def test_to_vector_order():
    ext = FeatureExtractor()
    features = ext.extract_features("test", _sample_doc(), 0.9)
    vec = ext.to_vector(features)
    assert len(vec) == 15
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


def test_feature_names_static():
    names = FeatureExtractor.feature_names()
    assert names == FEATURE_NAMES
    assert len(names) == 15


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
    # The bias_map lookup uses the "title" key from the raw dict,
    # so we need to also include "title" or the feature extractor
    # looks up doc.get("title") which is the default field name.
    doc = {"name": "Test Doc", "title": "Test Doc", "body": "Some content", "category": "Legal"}
    features = ext.extract_features("test query", doc, 0.80)
    assert features["domain_type"] == 3
    assert features["bias_adjustment"] == -0.1


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
