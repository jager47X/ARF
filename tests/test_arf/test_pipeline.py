"""Tests for arf.pipeline — Pipeline end-to-end with mocks."""

from arf.pipeline import Pipeline
from arf.triage import Triage


def _mock_embed(text):
    return [0.1] * 10


def _mock_search(embedding, top_k):
    return [
        ({"_id": "1", "title": "14th Amendment", "text": "All persons born..."}, 0.92),
        ({"_id": "2", "title": "5th Amendment", "text": "No person shall..."}, 0.78),
        ({"_id": "3", "title": "Random doc", "text": "Unrelated content."}, 0.50),
    ]


def test_minimal_pipeline():
    pipeline = Pipeline(search_fn=_mock_search, embed_fn=_mock_embed)
    results = pipeline.run("what is due process")
    assert isinstance(results, list)


def test_pipeline_with_triage():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        triage=Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70),
    )
    results = pipeline.run("what is due process")
    assert len(results) >= 1
    assert results[0]["score"] >= 0.85


def test_pipeline_with_predict_fn():
    def mock_predict(vectors):
        return [0.9, 0.5, 0.1]  # first high, second uncertain, third low

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        predict_fn=mock_predict,
        triage=Triage(min_score=0.50, accept_threshold=0.85, verify_threshold=0.70),
    )
    results = pipeline.run("what is due process")
    assert isinstance(results, list)


def test_pipeline_with_cache_hit():
    cached = [
        ({"title": "Cached Result", "text": "From cache."}, 0.95),
    ]

    def mock_lookup(query):
        return {"results": cached}

    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        cache_lookup=mock_lookup,
    )
    results = pipeline.run("cached query")
    assert len(results) == 1
    assert results[0]["document"].title == "Cached Result"


def test_pipeline_moderation_blocks():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        moderate_fn=lambda q: False,
    )
    results = pipeline.run("blocked query")
    assert results == []


def test_pipeline_preprocess():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        preprocess_fn=lambda q: q.upper(),
    )
    results = pipeline.run("what is due process")
    assert isinstance(results, list)


def test_pipeline_with_summarize():
    pipeline = Pipeline(
        search_fn=_mock_search,
        embed_fn=_mock_embed,
        summarize_fn=lambda q, doc, ctx: f"Summary for {doc.title}",
    )
    results = pipeline.run("what is due process")
    assert any(r.get("summary") for r in results)


def test_pipeline_empty_search():
    pipeline = Pipeline(
        search_fn=lambda emb, k: [],
        embed_fn=_mock_embed,
    )
    results = pipeline.run("nothing here")
    assert results == []
