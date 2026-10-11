# ARF — Advanced Retrieval Framework

[![CI](https://github.com/jager47X/ARF/actions/workflows/ci.yml/badge.svg)](https://github.com/jager47X/ARF/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/advanced-rag-framework.svg)](https://pypi.org/project/advanced-rag-framework/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **zero-dependency** retrieval pipeline toolkit. Plug in your own vector search, embedding model, LLM, ML model, and database — ARF provides the routing algorithms, feature engineering, rephrase-graph caching, and score blending.

```bash
pip install advanced-rag-framework
```

## What ARF Does

Most RAG pipelines send every candidate to an expensive LLM for reranking. ARF eliminates this waste with a multi-stage filtering pipeline called **R-Flow**:

```
Query
  → Cache graph walk (free — returns instantly if seen before)
  → Vector search (your provider; optionally multi-query via expand_fn)
  → Threshold + gap filter (free — drops obvious junk)
  → MLP triage (free, <5ms — accept/reject/uncertain)
  → Cross-encoder rerank (optional, ~ms — resolves most of the uncertain ones)
  → LLM verification ($$$ — only for what is still uncertain)
  → Answer with summaries
```

Each stage filters candidates so the next stage does less work; typically only the uncertain ~20%
ever reach the LLM — and `pipeline.stats.summary()["llm_share_of_candidates"]` tells you the real
number on your traffic (see [Stats & evaluation](#stats--evaluation)).

## Quick Start

```python
from arf import Pipeline, DocumentConfig, Triage

pipeline = Pipeline(
    doc_config=DocumentConfig(title_field="title", text_fields=["text"]),
    triage=Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70),
    search_fn=my_search,       # (embedding, top_k) → [(dict, float)]
    embed_fn=my_embed,         # (text) → [float]
)

results = pipeline.run("how does caching work?")
```

That's it. Two required functions. Everything else is optional.

## Full Pipeline

```python
from arf import Pipeline, DocumentConfig, Triage
from arf.trainer import load_reranker

pipeline = Pipeline(
    doc_config=DocumentConfig(
        title_field="title",
        text_fields=["text", "summary"],
        children_fields=["sections", "clauses"],
        hierarchy=["title", "chapter", "section"],
    ),
    triage=Triage(
        min_score=0.65,
        accept_threshold=0.85,
        verify_threshold=0.70,
        gap=0.20,
    ),

    # Required
    search_fn=my_search,           # any vector DB
    embed_fn=my_embed,             # any embedding model

    # Scoring (optional)
    predict_fn=load_reranker("model.joblib"),  # trained MLP
    rerank_fn=my_cross_encoder,   # (query, [dict]) → [prob]   e.g. arf.rerankers
    llm_fn=my_llm_verify,         # any LLM

    # Matching (optional)
    keyword_fn=my_keyword_search,  # (query) → [(dict, score)]  lexical hits
    alias_fn=my_alias_lookup,      # (query) → [(alias, title)]  synonym hits

    # Cache (optional)
    cache_lookup=my_cache_get,     # any cache backend
    cache_store=my_cache_set,

    # Preprocessing (optional)
    preprocess_fn=my_clean,        # translate, normalize, etc.
    moderate_fn=my_moderate,       # content safety
    expand_fn=my_expand,           # (query) → [variants]  multi-query retrieval
    rephrase_fn=my_rephrase,       # retry with rephrased query when nothing is found

    # Hierarchy (optional)
    resolve_fn=my_get_parent,      # walk up document tree
    summarize_fn=my_summarize,     # generate answer
)

results = pipeline.run("what is due process?", top_k=5)
# [{"document": Document, "score": 0.94, "decided_by": "llm", "llm_score": 8, "rerank_score": None,
#   "context": [...], "summary": "..."}, ...]
```

## Components

ARF is 6 independent modules. Use them together or individually.

### Document — DB-agnostic data model

```python
from arf import Document, DocumentConfig

config = DocumentConfig(
    title_field="name",
    text_fields=["body", "content"],
    children_fields=["subsections"],
    hierarchy=["category", "name"],
)

doc = Document.from_dict({"name": "Guide", "body": "...", "category": "Medical"}, config)
# doc.depth = 2, doc.path = "Medical / Guide"
```

Works with any database. MongoDB, PostgreSQL, DynamoDB, Pinecone, FAISS — just map your fields.

### Features — 16-feature extraction

```python
from arf import CorpusStats, FeatureExtractor

stats = CorpusStats.from_documents(all_docs, config)   # once, at ingest time (optional)
extractor = FeatureExtractor(config, corpus_stats=stats)
features = extractor.extract_features(query="...", document={...}, semantic_score=0.85)
vector = extractor.to_vector(features)  # [0.85, 4.2, 0, 0, ...]
```

| Feature | Description |
|---------|-------------|
| `semantic_score` | Raw cosine similarity from vector search |
| `bm25_score` | BM25 of the query against the text — real IDF weights and corpus average length when `CorpusStats` are given, otherwise a saturated term-frequency overlap |
| `alias_match` | Whether query matches a document alias |
| `keyword_match` | Whether query matches via keyword pattern |
| `domain_type` | Encoded domain identifier |
| `document_length` | Log-scaled character count |
| `query_length` | Log-scaled query token count |
| `section_depth` | Depth in document hierarchy |
| `embedding_cosine_similarity` | Cosine between query and stored document embedding (equals `semantic_score` when no document embedding is available) |
| `match_type` | 0=none, 1=partial, 2=exact |
| `score_gap_from_top` | Gap from the query's highest-scored document |
| `query_term_coverage` | Fraction of query terms in document |
| `title_similarity` | Jaccard similarity between query and title |
| `exact_phrase_match` | Whether the query occurs verbatim (contiguous tokens) in the title or text |
| `has_nested_content` | Whether document has children |
| `bias_adjustment` | Configurable per-document bias |

`FEATURE_VERSION` (currently 2) is recorded in every model bundle; `load_reranker` warns (or
raises with `strict=True`) when a model was trained on another version. Persist `CorpusStats`
with `to_dict()` / `from_dict()` and pass the same stats to `Pipeline(corpus_stats=...)` so
training and inference agree.

All title, hierarchy, text and embedding lookups go through your `DocumentConfig`
(`title_field`, `hierarchy`, `text_fields`, `children_fields`, `embedding_field`), so a schema
with `title_field="name"` produces the same features as the default one.

### Triage — threshold + gap + zone routing

```python
from arf import Triage

triage = Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70, gap=0.20)
result = triage.classify(candidates)
# result.accepted, result.needs_review, result.rejected
```

| Threshold | Meaning |
|-----------|---------|
| `min_score` | Below this → rejected outright |
| `accept_threshold` | At or above this → accepted without further checks |
| `verify_threshold` | Between this and `accept_threshold` → "needs review" (MLP → LLM) |
| `gap` | Candidates more than this far below the top result are dropped |

**Candidate pool.** `pipeline.run(query, top_k=10)` asks `search_fn` for `top_k × candidate_multiplier`
candidates (default 3×, so 30) and slices the final list back to `top_k`; the triage, MLP and LLM
stages need a pool larger than the result count to have anything to re-rank. Tune with
`Pipeline(candidate_multiplier=..., max_candidates=...)`.

**Calibrate the thresholds to your embedding model.** The defaults above assume cosine
similarities where a strong match scores ≳0.85. Many models (e.g. OpenAI `text-embedding-3-*`)
score relevant documents in the 0.3–0.6 range; with those, keep the *ordering* of the thresholds
but shift them down (the sample project uses `0.30 / 0.60 / 0.40`). If every candidate is
rejected, the pipeline logs a one-time warning with the top observed score to help you re-tune.

**No reviewer configured?** When neither `predict_fn` nor `llm_fn` is provided (or the MLP is
uncertain and there is no `llm_fn`), "needs review" candidates are returned at their retrieval
score instead of being discarded. Pass `Pipeline(..., unreviewed_policy="drop")` to discard them.

### Query expansion — `expand_fn`

```python
def expand(query: str) -> list[str]:
    # any source of alternative phrasings: an LLM, a synonym table, translations,
    # or a hypothetical answer (HyDE) that is embedded instead of the question
    return llm(f"Write 3 alternative search queries for: {query}").splitlines()

pipeline = Pipeline(..., expand_fn=expand, max_expansions=3)
```

Each variant is embedded and searched like the original, and a document keeps its **best** score
across all of them (every variant uses the same embedding model, so the scores are comparable).
Expansion runs *before* triage and only ever adds candidates: a failing `expand_fn` or a failing
search for one variant is reported through `on_error` and skipped. `keyword_fn` still runs once,
on the original query. This is different from `rephrase_fn`, which *replaces* the query and is
only tried when a search comes back empty.

### Hybrid retrieval — `keyword_fn`

```python
pipeline = Pipeline(
    ...,
    keyword_fn=my_bm25_search,   # (query) → [(dict, score)]  — any lexical/full-text search
    keyword_boost=0.05,          # bonus for every keyword hit (default)
)
```

Keyword scores (BM25, text-index ranks) are not comparable with cosine similarities, so ARF
puts every keyword hit on the semantic scale before merging:

1. A hit that is **also a vector candidate** keeps its cosine score.
2. A **keyword-only** hit with a stored embedding (`DocumentConfig.embedding_field`, default
   `"embedding"`) is scored by cosine against the query embedding.
3. A keyword-only hit **without** an embedding gets `keyword_floor` (default: `verify_threshold`),
   so it is sent for review rather than accepted or rejected blindly.

Every hit then receives `keyword_boost`. Set `keyword_mode="max"` to merge raw keyword scores
as-is (only sensible if they are already cosine-like).

### Cross-encoder reranking — `rerank_fn`

```python
from arf.rerankers import sentence_transformers_reranker   # pip install advanced-rag-framework[rerank]

pipeline = Pipeline(
    ...,
    rerank_fn=sentence_transformers_reranker(config=my_config),  # default: cross-encoder/ms-marco-MiniLM-L-6-v2
    rerank_zones=(0.4, 0.6),   # ≥0.6 accept, <0.4 reject, between → LLM
)
results[0]["rerank_score"]    # probability, or None if the candidate never reached the re-ranker
```

A cross-encoder reads the query and document together and resolves most "uncertain" candidates
for a few milliseconds each, so far fewer reach the LLM. It runs on whatever the MLP left uncertain
(or on the whole review band if you have no MLP). Any `(query, [doc_dict, ...]) → [prob, ...]`
callable works — Cohere Rerank, Voyage rerank, a local model — and
`arf.rerankers.cross_encoder_rerank_fn` wraps anything with a `predict(pairs)` method.

### LLM verification — `llm_fn`

```python
pipeline = Pipeline(
    ...,
    llm_fn=my_llm_verify,   # (query, doc_dict) → text containing a 0-9 score, e.g. "Score: 7"
    llm_pass_score=6,       # default
    llm_workers=4,          # verify uncertain candidates concurrently (threads; default 1)
)
results = pipeline.run("...")
results[0]["llm_score"]     # 0-9, or None if the candidate never reached the LLM
```

Only *uncertain* candidates reach the LLM. By default (`llm_mode="gate"`) its 0-9 score is a
**pass/fail gate**: a candidate scoring at least `llm_pass_score` is accepted and keeps its
retrieval score, so ranking and the gap filter stay on a single scale. The gap filter is always
measured on retrieval scores. `llm_mode="blend"` restores the pre-0.3.0 multiplicative blend
(`parser_range`, default 0.5×–1.5×).

### Failure handling — `on_error`

Optional stages never take the query down: if the cache, the LLM, the MLP or the summarizer
fails, the pipeline degrades and carries on. Every such failure is logged at `WARNING` and
passed to an optional hook, so an outage shows up in your logs and metrics rather than as
quietly thinner results:

```python
pipeline = Pipeline(
    ...,
    on_error=lambda stage, exc: metrics.increment(f"arf.errors.{stage}"),
    llm_error_policy="keep",   # default: an unverifiable candidate keeps its retrieval score
)                              # "drop" discards it instead
```

`stage` is one of `store_query_fn`, `link_fn`, `cache_store`, `alias_fn`, `predict_fn`,
`llm_fn`, `llm_fn output parsing`, `summarize_fn`.

### QueryGraph — rephrase chain walk

```python
from arf import follow_rephrase_chain

result = follow_rephrase_chain("due process clause", lookup_fn=my_db_lookup, max_hops=3)
# result.hit, result.cached_results, result.chain, result.loop_detected
```

Walks a directed graph of query→rephrase edges with loop detection and early exit on cache hit. Storage-agnostic — you provide the `lookup_fn`.

### Cache — what is stored, and when it is a hit

```python
cache = {}  # any store: Redis, MongoDB, SQLite, a dict ...

pipeline = Pipeline(
    ...,
    cache_lookup=lambda key: cache.get(key),                        # → {"results": payload, "next": key} | None
    cache_store=lambda key, payload: cache.setdefault(key, {}).update(results=payload),
    link_fn=lambda src, dst: cache.setdefault(src, {}).update(next=dst),   # optional rephrase edge
    cache_ttl=24 * 3600,            # optional: seconds an entry stays valid
    cache_version="ingest-2026-10", # optional: change it after re-ingesting to invalidate everything
)
```

* **Keys are normalised** (lower-case, trimmed, whitespace collapsed — `cache_normalize_fn`) for
  lookups, stores and links alike, so `"What is  due process"` and `"what is due process"` share one entry.
* **The payload is a JSON-friendly dict** — `{"version", "cache_version", "top_k", "created_at",
  "results": [{"item", "score", "llm_score", "summary"}, ...]}`. Store it as-is; it survives a
  round-trip through Redis or MongoDB (tuples that come back as lists are fine).
* **A hit is free.** Cached summaries and LLM scores are served directly; `llm_fn` and
  `summarize_fn` are not called again.
* **A hit must fit the request.** Expired entries (`cache_ttl`), entries written under another
  `cache_version`, and entries stored for a smaller `top_k` than requested are misses, and the
  rephrase walk keeps following `next` edges past them.
* **Semantic lookup (optional).** Give `cache_nearest_fn=(embedding) → (cached_query_text, similarity)`
  (e.g. a vector index over your cached queries) and near-duplicate queries above
  `cache_similarity_threshold` (default 0.95) are served from the neighbour's entry. On a miss
  the embedding is reused for the vector search, so it costs nothing extra.

### ScoreParser — LLM output parsing + multiplier blending

```python
from arf import extract_score, multiplier, adjust_score

extract_score('{"score": 7}')                        # → 7
extract_score("Reasoning: on point.\nScore: 8 (high)")  # → 8
extract_score("**Score:** 7/10")                      # → 6   (other scales are rescaled to 0-9)
extract_score('```json\n{"score": 6}\n```')            # → 6   (code fences, embedded JSON, "score is N" ...)
multiplier(8)                                         # → 1.39
adjust_score(0.72, "Score: 8")                        # → min(0.72 * 1.39, 1.0)
```

Parses messy LLM output into a 0-9 score (booleans are rejected, fractions like `7/10` are
rescaled, the last `Score:` line wins), converts it to a multiplier, and blends it with the
retrieval score.

### Trainer — MLP training

```python
from arf.trainer import train_reranker, load_reranker

# Train
metrics = train_reranker(X, y, architecture=(64, 32, 16), save_path="model.joblib")

# Load as a predict_fn for Pipeline
predict_fn = load_reranker("model.joblib")
```

Requires `pip install advanced-rag-framework[ml]` (numpy + scikit-learn).

## Ingest — document ingestion helper

```python
from arf import ingest_documents, DocumentConfig

result = ingest_documents(
    documents,
    config=DocumentConfig(title_field="title", text_fields=["text"]),
    embed_fn=my_embed,     # your embedding function
    store_fn=my_store,     # your DB write function
)
# result.processed, result.skipped, result.errors
```

Validates documents, computes hierarchy metadata (depth, path), generates embeddings for parent and children, and stores via your function.

## Stats & evaluation

Every run records what each stage did; nothing is estimated.

```python
results = pipeline.run("what is due process?")

s = pipeline.last_stats          # arf.RunStats for this query
s.fetched, s.accepted_direct, s.needs_review, s.rejected      # triage funnel
s.mlp_evaluated, s.rerank_evaluated, s.llm_evaluated           # who did the work
s.cache_hit, s.errors, s.timings_ms                            # {"embed": 12.1, "search": 30.4, "llm": 410.2, "total": ...}

pipeline.stats.summary()         # aggregate over all runs of this instance
# {"runs": 1203, "cache_hit_rate": 0.31, "llm_share_of_candidates": 0.18,
#  "llm_calls_per_query": 1.4, "llm_calls_avoided": 5120, "avg_latency_ms": 212.0,
#  "funnel": {...}, "avg_stage_ms": {...}, "errors_by_stage": {...}}
pipeline.stats.recent(20)        # the last 20 RunStats
Pipeline(..., stats_fn=export)   # (RunStats) -> None, called after every run
```

To measure quality **and** cost on a labelled set in one go:

```python
from arf.evaluation import evaluate

dataset = [
    {"query": "what is due process?", "relevant": ["usc-14-1", "usc-5"]},
    {"query": "quartering soldiers",   "relevant": {"usc-3": 3, "usc-4": 1}},   # graded
]
report = evaluate(pipeline, dataset, ks=(1, 5, 10))
report.metrics          # {"mrr": 0.83, "recall@5": 0.91, "ndcg@10": 0.88, ...}
report.stats            # pipeline.stats-style summary over the evaluation runs
print(report.markdown())
```

`arf.evaluation` also exposes `recall_at_k`, `precision_at_k`, `hit_rate_at_k`, `reciprocal_rank`
and `ndcg_at_k` for your own harness. Pure Python, no dependencies.

## Admin console & search UI

```bash
pip install advanced-rag-framework[ui]
python -m arf.ui --demo                     # no keys, no database: a built-in demo corpus
#   http://127.0.0.1:8000/        search page
#   http://127.0.0.1:8000/admin   admin console
```

Serve your own pipeline with a factory (`module:callable` returning a `Pipeline`), or mount the app
yourself:

```bash
python -m arf.ui --app mypackage.search:build_pipeline --admin-token "$ARF_ADMIN_TOKEN"
```

```python
from arf.ui import create_app
app = create_app(pipeline, admin_token="...", cache_clear_fn=my_cache.flushall)   # uvicorn mymodule:app
```

* **Search page** (`/`, `POST /api/query`): results with score, the stage that accepted each one
  (`threshold` / `mlp` / `rerank` / `llm` / `unreviewed`), LLM and re-ranker scores, summaries and
  context, plus a per-query strip (fetched → review → LLM, cache hit, latency, errors).
* **Admin console** (`/admin`, `/admin/api/*`): live stat tiles (cache hit rate, LLM share of
  candidates, LLM calls per query, latency, errors), the candidate funnel and per-stage latency
  with table views, recent runs, a **live editor for thresholds, zones and policies** (applied to
  the running pipeline, not persisted), an **evaluation runner** for labelled queries, and cache
  tools.
* **Access control**: the admin API requires the `X-ARF-Admin-Token` header (`admin_token=` or
  `ARF_ADMIN_TOKEN`). Without a token it only answers requests from localhost. The search page is
  public — put it behind your own auth if the corpus is not.

## Bring Your Own Everything

| Slot | What you provide | Examples |
|------|-----------------|----------|
| `search_fn` | Vector search | FAISS, Pinecone, Weaviate, Qdrant, MongoDB Atlas, pgvector |
| `embed_fn` | Embeddings | OpenAI, Voyage AI, Cohere, sentence-transformers, Ollama |
| `predict_fn` | ML model | scikit-learn, XGBoost, PyTorch, any callable |
| `rerank_fn` | Cross-encoder / reranker | sentence-transformers, Cohere Rerank, Voyage rerank |
| `llm_fn` | LLM verification | OpenAI, Anthropic, Ollama, Llama.cpp, any API |
| `expand_fn` | Query expansion | LLM paraphrases, HyDE, synonym tables, translation |
| `keyword_fn` | Lexical / full-text search | BM25, Elasticsearch, MongoDB text index, SQL `LIKE` |
| `alias_fn` | Alias / synonym lookup | Dict, alias table, any search |
| `cache_lookup/store` | Cache | Redis, MongoDB, SQLite, in-memory dict |
| `cache_nearest_fn` | Semantic cache lookup | Any vector index over cached queries |
| `resolve_fn` | Parent lookup | Any database query |
| `summarize_fn` | Answer generation | Any LLM |
| `store_fn` (ingest) | Document storage | Any database write |

## Installation

```bash
# Core (zero dependencies)
pip install advanced-rag-framework

# With MLP training support (numpy + scikit-learn)
pip install advanced-rag-framework[ml]

# With the built-in cross-encoder re-ranker (sentence-transformers)
pip install advanced-rag-framework[rerank]

# With the search page + admin console (fastapi + uvicorn)
pip install advanced-rag-framework[ui]
```

## Sample Project

See [`sample-project/`](sample-project/) for a complete working example using:
- **FAISS** for vector search
- **Voyage AI** for embeddings
- **OpenAI** for LLM verification
- A **cooking recipe** dataset (non-legal, 46 recipes from 15 cuisines)

```bash
python sample-project/ingest.py                          # Embed recipes into FAISS
python sample-project/train.py                           # Train MLP reranker
python sample-project/query.py "spicy noodle soup"       # Full pipeline query
```

## R-Flow Pipeline

The core innovation — each stage filters candidates so the next stage does less work:

```
                    ┌──────────────────────┐
                    │   Vector Search      │
                    │  (your provider)     │
                    └──────────┬───────────┘
                               │ candidates with scores
                    ┌──────────▼───────────┐
                    │  Threshold + Gap     │
                    │  Filter (~60% cut)   │
                    └──────────┬───────────┘
                               │ survivors
                    ┌──────────▼───────────┐
                    │  Feature Extraction  │
                    │  (16 features)       │
                    └──────────┬───────────┘
                               │ feature vectors
                    ┌──────────▼───────────┐
                    │   MLP Reranker       │
                    │  (<5ms, $0.00)       │
                    └──────────┬───────────┘
                        ┌──────┼──────┐
                   p≥0.6│ 0.4≤p<0.6   │p<0.4
                        │      │      │
                   Accept   ┌──▼──────────┐  Reject
                   (free)   │Cross-encoder│  (free)
                            │ (optional)  │
                            └──┬──────────┘
                        ┌──────┼──────┐
                   p≥0.6│ 0.4≤p<0.6   │p<0.4
                        │      │      │
                   Accept   ┌──▼──┐  Reject
                   (free)   │ LLM │  (free)
                            └──┬──┘
                          Accept/Reject
```

## Development

```bash
git clone https://github.com/jager47X/ARF.git
cd ARF
pip install -e ".[dev]"

# Run library tests
pytest tests/test_arf/ -v

# Lint
ruff check arf/ tests/test_arf/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.
