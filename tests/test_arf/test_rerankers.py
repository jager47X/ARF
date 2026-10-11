"""Tests for arf.rerankers — dependency-free parts, with a fake cross-encoder."""

import importlib.util

import pytest

from arf.document import Document, DocumentConfig
from arf.rerankers import cross_encoder_rerank_fn, document_text_fn, sentence_transformers_reranker


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder.predict."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, batch_size=32):
        self.calls.append((list(pairs), batch_size))
        return list(self.scores)


def test_document_text_fn_honours_config_and_truncates():
    text_of = document_text_fn(DocumentConfig(title_field="name", text_fields=["body", "text"]), max_chars=20)
    assert text_of({"name": "Ramen", "body": "Noodles in broth, served hot."}) == "Ramen\nNoodles in bro"  # 20 chars
    assert text_of({"name": "Ramen", "text": "fallback field"}) == "Ramen\nfallback field"[:20]
    # Default config and Document objects
    assert document_text_fn()({"title": "T", "text": "body"}) == "T\nbody"
    assert document_text_fn()(Document(title="T", content="body")) == "T\nbody"
    assert document_text_fn()({}) == ""


def test_cross_encoder_rerank_fn_passes_pairs_and_keeps_probabilities():
    model = FakeCrossEncoder([0.9, 0.2])
    rerank = cross_encoder_rerank_fn(model, text_fn=document_text_fn(), batch_size=8)
    items = [{"title": "A", "text": "a"}, {"title": "B", "text": "b"}]
    assert rerank("q", items) == [0.9, 0.2]
    assert model.calls == [([("q", "A\na"), ("q", "B\nb")], 8)]
    assert rerank("q", []) == []


def test_cross_encoder_rerank_fn_squashes_logits_with_sigmoid():
    rerank = cross_encoder_rerank_fn(FakeCrossEncoder([3.0, -3.0]), text_fn=document_text_fn())
    hi, lo = rerank("q", [{"title": "A"}, {"title": "B"}])
    assert 0.95 < hi < 1.0 and 0.0 < lo < 0.05


def test_cross_encoder_rerank_fn_accepts_array_like_outputs():
    class ArrayLike:
        def tolist(self):
            return [0.5]

    class Model:
        def predict(self, pairs, batch_size=32):
            return ArrayLike()

    assert cross_encoder_rerank_fn(Model(), text_fn=document_text_fn())("q", [{"title": "A"}]) == [0.5]


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None, reason="sentence-transformers is installed"
)
def test_sentence_transformers_reranker_explains_missing_dependency():
    with pytest.raises(ImportError, match=r"advanced-rag-framework\[rerank\]"):
        sentence_transformers_reranker()
