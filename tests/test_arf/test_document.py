"""Tests for arf.document — Document and DocumentConfig."""

from arf.document import Document, DocumentConfig


def test_default_config():
    cfg = DocumentConfig()
    assert cfg.title_field == "title"
    assert cfg.id_field == "_id"
    assert "text" in cfg.text_fields
    assert "clauses" in cfg.children_fields


def test_field_mapping_property():
    cfg = DocumentConfig(title_field="name", text_fields=["body"], children_fields=["items"])
    fm = cfg.field_mapping
    assert fm["title"] == "name"
    assert fm["text"] == ["body"]
    assert fm["nested_text"] == ["items"]


def test_document_from_dict_basic():
    raw = {"_id": "123", "title": "14th Amendment", "text": "All persons born..."}
    doc = Document.from_dict(raw)
    assert doc.id == "123"
    assert doc.title == "14th Amendment"
    assert doc.content == "All persons born..."
    assert doc.children is None
    assert doc.has_children is False


def test_document_from_dict_with_children():
    raw = {
        "_id": "456",
        "title": "Article I",
        "article": "I",
        "section": "1",
        "text": "All legislative powers...",
        "clauses": [
            {"title": "Clause 1", "text": "The Senate..."},
            {"title": "Clause 2", "text": "No person..."},
        ],
    }
    doc = Document.from_dict(raw)
    assert doc.has_children is True
    assert len(doc.children) == 2
    assert doc.children[0].title == "Clause 1"
    assert doc.children[0].content == "The Senate..."


def test_document_depth_and_path():
    raw = {"title": "14th Amendment", "article": "XIV", "section": "1"}
    doc = Document.from_dict(raw)
    assert doc.depth == 3
    assert "14th Amendment" in doc.path
    assert "XIV" in doc.path
    assert "1" in doc.path


def test_document_full_text():
    raw = {
        "title": "Test",
        "text": "Parent text.",
        "clauses": [{"text": "Child text."}],
    }
    doc = Document.from_dict(raw)
    assert "Parent text." in doc.full_text
    assert "Child text." in doc.full_text


def test_document_ancestors():
    doc = Document(path="Statute A / Part 2 / Chapter 3")
    assert doc.ancestors() == ["Statute A", "Part 2", "Chapter 3"]


def test_document_custom_config():
    cfg = DocumentConfig(
        id_field="pk",
        title_field="name",
        text_fields=["body"],
        children_fields=["items"],
        hierarchy=["name", "category"],
        domain_id=5,
    )
    raw = {"pk": "abc", "name": "Test Doc", "body": "Content here", "category": "Legal"}
    doc = Document.from_dict(raw, cfg)
    assert doc.id == "abc"
    assert doc.title == "Test Doc"
    assert doc.content == "Content here"
    assert doc.depth == 2
    assert "Legal" in doc.path


def test_document_no_text():
    raw = {"_id": "empty"}
    doc = Document.from_dict(raw)
    assert doc.content == ""
    assert doc.title == ""
