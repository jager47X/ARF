"""Tests for arf.ingest — ingest_documents."""

from arf.document import DocumentConfig
from arf.ingest import ingest_documents


def test_basic_ingest():
    stored = []

    def mock_embed(text):
        return [0.1] * 10

    def mock_store(doc):
        stored.append(doc)

    docs = [
        {"title": "14th Amendment", "text": "All persons born..."},
        {"title": "5th Amendment", "text": "No person shall..."},
    ]

    result = ingest_documents(docs, embed_fn=mock_embed, store_fn=mock_store)
    assert result.processed == 2
    assert result.errors == 0
    assert len(stored) == 2


def test_ingest_adds_embedding():
    stored = []

    def mock_embed(text):
        return [float(len(text))]

    result = ingest_documents(
        [{"title": "Test", "text": "Hello"}],
        embed_fn=mock_embed,
        store_fn=lambda doc: stored.append(doc),
    )
    assert result.processed == 1
    assert "embedding" in stored[0]
    assert isinstance(stored[0]["embedding"], list)


def test_ingest_adds_hierarchy_metadata():
    stored = []

    ingest_documents(
        [{"title": "14th Amendment", "article": "XIV", "section": "1", "text": "..."}],
        embed_fn=lambda t: [0.1],
        store_fn=lambda doc: stored.append(doc),
    )
    assert stored[0]["depth"] == 3
    assert "XIV" in stored[0]["path"]


def test_ingest_embeds_children():
    stored = []

    result = ingest_documents(
        [
            {
                "title": "Article I",
                "text": "Legislative powers.",
                "clauses": [
                    {"title": "Clause 1", "text": "The Senate..."},
                    {"title": "Clause 2", "text": "No person..."},
                ],
            }
        ],
        embed_fn=lambda t: [0.1],
        store_fn=lambda doc: stored.append(doc),
    )
    assert result.processed == 1
    clauses = stored[0]["clauses"]
    assert len(clauses) == 2
    assert "embedding" in clauses[0]
    assert "embedding" in clauses[1]


def test_ingest_skips_empty_docs():
    result = ingest_documents(
        [{"_id": "empty"}],
        embed_fn=lambda t: [0.1],
        store_fn=lambda doc: None,
    )
    assert result.skipped == 1
    assert result.processed == 0


def test_ingest_handles_errors():
    def failing_store(doc):
        raise RuntimeError("DB is down")

    result = ingest_documents(
        [{"title": "Test", "text": "Content"}],
        embed_fn=lambda t: [0.1],
        store_fn=failing_store,
    )
    assert result.errors == 1
    assert result.processed == 0
    assert len(result.error_details) == 1


def test_ingest_custom_config():
    stored = []
    cfg = DocumentConfig(
        title_field="name",
        text_fields=["body"],
        children_fields=["items"],
        hierarchy=["name", "category"],
    )

    result = ingest_documents(
        [{"name": "Guideline", "body": "Content", "category": "Medical"}],
        config=cfg,
        embed_fn=lambda t: [0.1],
        store_fn=lambda doc: stored.append(doc),
    )
    assert result.processed == 1
    assert stored[0]["depth"] == 2
    assert "Medical" in stored[0]["path"]


def test_ingest_wraps_flat_text_as_child():
    stored = []

    result = ingest_documents(
        [{"title": "Simple", "text": "Just flat text."}],
        embed_fn=lambda t: [0.1],
        store_fn=lambda doc: stored.append(doc),
    )
    assert result.processed == 1
    assert "clauses" in stored[0]
    assert len(stored[0]["clauses"]) == 1
