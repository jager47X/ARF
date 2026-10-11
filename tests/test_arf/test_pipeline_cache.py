"""Tests for the Pipeline cache: key normalisation, payload shape, validity
(TTL / version / top_k), JSON round-trips, cached summaries and the optional
semantic lookup."""

import json
import time

from arf.pipeline import CACHE_FORMAT_VERSION, Pipeline


def _embed(text):
    return [0.1] * 4


def _docs():
    return [
        ({"_id": "1", "title": "14th Amendment", "text": "All persons born..."}, 0.92),
        ({"_id": "2", "title": "5th Amendment", "text": "No person shall..."}, 0.78),
        ({"_id": "3", "title": "Random doc", "text": "Unrelated."}, 0.50),
    ]


class DictCache:
    """Minimal graph store: node = {"results": <payload>, "next": <text>}."""

    def __init__(self, json_roundtrip=False):
        self.nodes = {}
        self.json_roundtrip = json_roundtrip

    def lookup(self, text):
        return self.nodes.get(text)

    def store(self, text, payload):
        if self.json_roundtrip:
            payload = json.loads(json.dumps(payload))  # what Redis / Mongo would hand back
        self.nodes.setdefault(text, {})["results"] = payload

    def link(self, source, target):
        self.nodes.setdefault(source, {})["next"] = target


def _pipeline(cache, **kwargs):
    search_calls = []

    def search_fn(embedding, k):
        search_calls.append(k)
        return _docs()

    pipeline = Pipeline(
        search_fn=search_fn, embed_fn=_embed, cache_lookup=cache.lookup, cache_store=cache.store, **kwargs
    )
    return pipeline, search_calls


def _titles(results):
    return [r["document"].title for r in results]


def test_store_and_lookup_share_the_normalised_key():
    cache = DictCache()
    pipeline, search_calls = _pipeline(cache)
    first = pipeline.run("What is Due Process")
    second = pipeline.run("  what is   DUE process ")
    assert list(cache.nodes) == ["what is due process"]
    assert search_calls == [30]  # second run was a hit
    assert _titles(second) == _titles(first)


def test_payload_shape_is_versioned_and_json_friendly():
    cache = DictCache()
    pipeline, _ = _pipeline(cache)
    pipeline.run("what is due process", top_k=5)
    payload = cache.nodes["what is due process"]["results"]
    assert payload["version"] == CACHE_FORMAT_VERSION
    assert payload["cache_version"] is None
    assert payload["top_k"] == 5
    assert abs(payload["created_at"] - time.time()) < 5
    assert [e["item"]["_id"] for e in payload["results"]] == ["1", "2"]
    assert payload["results"][0] == {
        "item": _docs()[0][0],
        "score": 0.92,
        "llm_score": None,
        "summary": "",
    }
    json.dumps(payload)  # must not raise


def test_json_roundtripped_entry_is_a_hit():
    cache = DictCache(json_roundtrip=True)
    pipeline, search_calls = _pipeline(cache)
    first = pipeline.run("q")
    second = pipeline.run("q")
    assert search_calls == [30]
    assert _titles(second) == ["14th Amendment", "5th Amendment"]
    assert [r["score"] for r in second] == [r["score"] for r in first]
    assert second[0]["document"].id == "1"


def test_cache_hit_serves_summaries_and_llm_scores_without_calling_llms():
    cache = DictCache(json_roundtrip=True)
    summaries, verifications = [], []

    def summarize_fn(query, doc, context):
        summaries.append(doc.id)
        return f"Summary of {doc.title}"

    def llm_fn(query, item):
        verifications.append(item["_id"])
        return "Score: 8"

    pipeline, _ = _pipeline(cache, summarize_fn=summarize_fn, llm_fn=llm_fn)
    first = pipeline.run("q")
    assert summaries == ["1", "2"] and verifications == ["2"]

    second = pipeline.run("q")
    assert summaries == ["1", "2"] and verifications == ["2"]  # nothing re-run
    assert [r["summary"] for r in second] == [r["summary"] for r in first]
    assert second[1]["llm_score"] == 8


def test_ttl_expires_entries():
    cache = DictCache()
    pipeline, search_calls = _pipeline(cache, cache_ttl=60)
    pipeline.run("q")
    pipeline.run("q")
    assert search_calls == [30]  # fresh → hit

    cache.nodes["q"]["results"]["created_at"] = time.time() - 120
    pipeline.run("q")
    assert search_calls == [30, 30]  # expired → miss, re-stored
    pipeline.run("q")
    assert search_calls == [30, 30]


def test_cache_version_mismatch_is_a_miss():
    cache = DictCache()
    v1, calls_v1 = _pipeline(cache, cache_version="ingest-2026-04")
    v1.run("q")
    assert cache.nodes["q"]["results"]["cache_version"] == "ingest-2026-04"

    v2, calls_v2 = _pipeline(cache, cache_version="ingest-2026-10")
    v2.run("q")
    assert calls_v2 == [30]  # other version → miss (and overwrote the entry)

    v1.run("q")
    assert calls_v1 == [30, 30]  # the entry now belongs to v2


def test_legacy_bare_list_entries_still_hit_unless_validity_checks_are_on():
    cache = DictCache()
    cache.nodes["q"] = {"results": [({"_id": "c", "title": "Cached Result", "text": "..."}, 0.95)]}

    pipeline, search_calls = _pipeline(cache)
    assert _titles(pipeline.run("q")) == ["Cached Result"]
    assert search_calls == []

    strict, strict_calls = _pipeline(cache, cache_ttl=3600)
    strict.run("q")
    assert strict_calls == [30]  # no created_at → cannot be trusted → miss


def test_entry_stored_for_smaller_top_k_is_a_miss():
    cache = DictCache()
    pipeline, search_calls = _pipeline(cache)
    pipeline.run("q", top_k=2)
    assert search_calls == [6]

    assert len(pipeline.run("q", top_k=1)) == 1  # hit, sliced down
    assert search_calls == [6]

    pipeline.run("q", top_k=5)
    assert search_calls == [6, 15]  # needs more than was stored → miss


def test_rephrase_chain_walk_reaches_a_payload_node():
    cache = DictCache()
    pipeline, search_calls = _pipeline(cache)
    pipeline.run("due process clause")  # stores under the normalised key
    cache.link("what is due process", "due process clause")

    results = pipeline.run("What is due process")
    assert search_calls == [30]  # served via the rephrase edge, no new search
    assert _titles(results) == ["14th Amendment", "5th Amendment"]


def test_semantic_lookup_serves_a_near_duplicate_query():
    cache = DictCache()
    embed_calls = []

    def embed_fn(text):
        embed_calls.append(text)
        return [0.1] * 4

    def nearest_fn(embedding):
        return ("what is due process", 0.97)

    search_calls = []

    def search_fn(embedding, k):
        search_calls.append(k)
        return _docs()

    def make(threshold):
        return Pipeline(
            search_fn=search_fn,
            embed_fn=embed_fn,
            cache_lookup=cache.lookup,
            cache_store=cache.store,
            cache_nearest_fn=nearest_fn,
            cache_similarity_threshold=threshold,
        )

    make(0.95).run("what is due process")
    assert search_calls == [30]

    results = make(0.95).run("explain due process")
    assert search_calls == [30]  # semantic hit → no search
    assert _titles(results) == ["14th Amendment", "5th Amendment"]

    make(0.99).run("explain due process please")
    assert search_calls == [30, 30]  # below threshold → miss ...
    assert embed_calls.count("explain due process please") == 1  # ... and the embedding was reused


def test_link_fn_receives_normalised_queries():
    cache = DictCache()
    links = []
    attempts = []

    def search_fn(embedding, k):
        attempts.append(k)
        return [] if len(attempts) == 1 else _docs()

    pipeline = Pipeline(
        search_fn=search_fn,
        embed_fn=_embed,
        cache_lookup=cache.lookup,
        cache_store=cache.store,
        rephrase_fn=lambda q, previous: "Due Process  Clause",
        link_fn=lambda source, target: links.append((source, target)),
    )
    results = pipeline.run("What is Due Process")
    assert len(results) == 2
    assert links == [("what is due process", "due process clause")]
    assert "due process clause" in cache.nodes  # stored under the rephrased, normalised key


def test_failing_cache_backend_is_reported_and_bypassed():
    errors = []

    def broken_lookup(text):
        raise ConnectionError("redis down")

    pipeline = Pipeline(
        search_fn=lambda e, k: _docs(),
        embed_fn=_embed,
        cache_lookup=broken_lookup,
        on_error=lambda stage, exc: errors.append(stage),
    )
    assert len(pipeline.run("q")) == 2
    assert errors == ["cache_lookup"]
