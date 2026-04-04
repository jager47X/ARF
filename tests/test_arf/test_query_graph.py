"""Tests for arf.query_graph — follow_rephrase_chain."""

from arf.query_graph import ChainResult, follow_rephrase_chain


def test_cache_hit_immediate():
    """Direct cache hit on the seed query."""
    nodes = {"what is due process": {"results": [{"title": "14th Amendment"}]}}
    result = follow_rephrase_chain("What is due process", lookup_fn=lambda q: nodes.get(q))
    assert result.hit is True
    assert result.hops == 0
    assert len(result.cached_results) == 1


def test_one_hop():
    """Follow one rephrase edge to find cached results."""
    nodes = {
        "due process clause": {"next": "what is due process"},
        "what is due process": {"results": [{"title": "14th Amendment"}]},
    }
    result = follow_rephrase_chain("due process clause", lookup_fn=lambda q: nodes.get(q))
    assert result.hit is True
    assert result.hops == 1
    assert result.chain == ["what is due process"]


def test_multi_hop():
    nodes = {
        "a": {"next": "b"},
        "b": {"next": "c"},
        "c": {"results": [{"title": "found"}]},
    }
    result = follow_rephrase_chain("a", lookup_fn=lambda q: nodes.get(q))
    assert result.hit is True
    assert result.hops == 2
    assert result.chain == ["b", "c"]


def test_loop_detection():
    nodes = {
        "a": {"next": "b"},
        "b": {"next": "c"},
        "c": {"next": "a"},
    }
    result = follow_rephrase_chain("a", lookup_fn=lambda q: nodes.get(q))
    assert result.hit is False
    assert result.loop_detected is True


def test_max_hops():
    nodes = {
        "a": {"next": "b"},
        "b": {"next": "c"},
        "c": {"next": "d"},
        "d": {"results": [{"title": "found"}]},
    }
    result = follow_rephrase_chain("a", lookup_fn=lambda q: nodes.get(q), max_hops=2)
    assert result.hit is False
    assert result.hit_max_hops is True
    assert result.hops == 2


def test_miss_no_node():
    result = follow_rephrase_chain("unknown", lookup_fn=lambda q: None)
    assert result.hit is False
    assert result.hops == 0
    assert result.cached_results is None


def test_miss_no_results_no_next():
    nodes = {"query": {"metadata": "something"}}
    result = follow_rephrase_chain("query", lookup_fn=lambda q: nodes.get(q))
    assert result.hit is False


def test_custom_normalize():
    nodes = {"UPPER": {"results": [1, 2, 3]}}
    result = follow_rephrase_chain("upper", lookup_fn=lambda q: nodes.get(q), normalize_fn=str.upper)
    assert result.hit is True


def test_chain_result_dataclass():
    r = ChainResult(final_text="x")
    assert r.hit is False
    assert r.hops == 0
    assert r.chain == []
