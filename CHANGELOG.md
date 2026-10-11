# Changelog

All notable changes to ARF will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Changed
- **Unreviewed candidates are kept** — when no `predict_fn`/`llm_fn` is configured (or the MLP is
  uncertain and there is no `llm_fn`), "needs review" candidates are returned at their retrieval
  score instead of being silently dropped. The previous behaviour is available with
  `Pipeline(unreviewed_policy="drop")` (`arf/pipeline.py`)
- **Keyword hits are merged on the semantic scale** — `keyword_fn` results are no longer
  max-merged with cosine scores. A hit also found by vector search keeps its cosine score, a
  keyword-only hit is re-scored by cosine against its stored embedding, and one without an
  embedding gets `keyword_floor` (default `verify_threshold`); all hits receive `keyword_boost`
  (default 0.05). `Pipeline(keyword_mode="max")` restores the old merge (`arf/pipeline.py`)
- **LLM verification is a gate, not a score multiplier** — an uncertain candidate whose LLM score
  reaches `llm_pass_score` (default 6) is accepted and keeps its retrieval score; the LLM score is
  reported as `llm_score` on each result. The multiplicative blend (0.5×–1.5×, capped at 1.0) let a
  0.70 document become 1.0 and evict everything else through the gap filter, and demanded a higher
  LLM score from lower-ranked candidates. `Pipeline(llm_mode="blend")` restores it
- **Gap filter always uses retrieval scores**, in both LLM modes (`arf/pipeline.py`)
- **Search fetches a candidate pool, not just `top_k`** — `search_fn` is now asked for
  `top_k × candidate_multiplier` candidates (default 3×, capped by `max_candidates`) so the
  triage/MLP/LLM stages have something to re-rank; results are still sliced to `top_k`.
  `candidate_multiplier=1.0` restores the old behaviour (`arf/pipeline.py`)
- **`cache_store` receives a versioned payload dict instead of a bare results list** —
  `{"version", "cache_version", "top_k", "created_at", "results": [{"item", "score", "llm_score",
  "summary"}]}`. It is JSON-friendly, and a hit serves the cached summaries and LLM scores
  without calling `llm_fn` / `summarize_fn` again. Pre-0.3.0 bare-list entries are still read
  (unless `cache_ttl` / `cache_version` is set). `cache_store` and `link_fn` now receive
  normalised keys (`arf/pipeline.py`)
- **Features: version 2 — retrain your MLP models.** `bm25_score` is real BM25 (IDF + corpus
  average length) when `CorpusStats` are supplied, `query_length` is a log-scaled token count
  instead of a raw character count, `embedding_cosine_similarity` falls back to `semantic_score`
  instead of 0 when the document has no stored embedding, and a 16th feature
  `exact_phrase_match` was added. Model bundles now record `feature_version`; `load_reranker`
  warns (or raises with `strict=True`) when a model was trained on another version
  (`arf/features.py`, `arf/trainer.py`)
- **Swallowed errors are logged at WARNING** (they were `debug` or silent) and forwarded to the new
  `on_error(stage, exc)` hook; `predict_fn`, `llm_fn`, `cache_store`, `store_query_fn`, `link_fn`,
  `alias_fn` and `summarize_fn` failures no longer disappear (`arf/pipeline.py`)

### Added
- **Per-run and aggregate statistics** — `pipeline.last_stats` (`arf.RunStats`: funnel counts per
  stage, cache hit, errors, per-stage timings) and `pipeline.stats` (`arf.PipelineStats`:
  `summary()` with `llm_share_of_candidates`, `llm_calls_per_query`, `cache_hit_rate`,
  `avg_stage_ms`, `errors_by_stage`; `recent()`; `reset()`), plus `Pipeline(stats_fn=...)`
  (`arf/stats.py`, `arf/pipeline.py`)
- **Evaluation module** — `arf.evaluation.evaluate(pipeline, dataset)` reports recall@k,
  precision@k, hit-rate@k, MRR and nDCG@k together with the pipeline's cost statistics;
  the metric functions are public and pure Python (`arf/evaluation.py`)
- Coverage gate raised from 10% (over `arf` + `rag_dependencies`) to 85% over `arf` alone
  (`pyproject.toml`)
- **Query expansion / multi-query retrieval** — `Pipeline(expand_fn=(query) -> [variants],
  max_expansions=3)` embeds and searches each variant (LLM paraphrases, HyDE, synonyms ...) and
  keeps a document's best score across them, before triage. Failures are reported via `on_error`
  and only ever cost the extra candidates (`arf/pipeline.py`)
- **Cross-encoder rerank slot** — `Pipeline(rerank_fn=(query, [doc]) -> [prob], rerank_zones=(0.4, 0.6))`
  runs between the MLP and the LLM on whatever is still uncertain; accepted candidates keep their
  retrieval score and every result reports `rerank_score`. `arf.rerankers` provides
  `sentence_transformers_reranker()` (lazy import, `pip install advanced-rag-framework[rerank]`),
  `cross_encoder_rerank_fn()` for any `predict(pairs)` model, and `document_text_fn()`
  (`arf/pipeline.py`, `arf/rerankers.py`)
- `Pipeline(llm_error_policy="keep"|"drop", on_error=...)` — what happens to a candidate when the
  LLM call fails or its answer has no parseable score, and an observer hook for every swallowed
  error (`arf/pipeline.py`)
- `arf.CorpusStats` (`from_documents`, `idf`, `to_dict`/`from_dict`) and
  `Pipeline(corpus_stats=...)` / `FeatureExtractor(corpus_stats=...)`; `sample-project/train.py`
  writes `corpus_stats.json` and `query.py` loads it (`arf/features.py`)
- `Pipeline(llm_workers=N)` — verify a query's uncertain candidates on a thread pool instead of one
  call at a time; order and error reporting are unchanged (`arf/pipeline.py`)
- Cache controls: `cache_ttl` (expiry), `cache_version` (bulk invalidation after re-ingesting),
  `cache_normalize_fn` (key normalisation, default `arf.query_graph.normalize_query`), and an
  optional semantic lookup via `cache_nearest_fn` + `cache_similarity_threshold`
  (`arf/pipeline.py`, `arf/query_graph.py`)
- `DocumentConfig.embedding_field` (default `"embedding"`) — the field holding a document's stored
  embedding, used by `Document.from_dict` and keyword re-scoring (`arf/document.py`)
- `arf.features.cosine_similarity` public helper
- `Pipeline(alias_fn=...)` — `(query) -> [(alias, title), ...]` alias/synonym hits fed to the MLP
  features (`arf/pipeline.py`)
- `FeatureExtractor.extract_batch(top_score=...)` to measure `score_gap_from_top` against the
  query's real top candidate when scoring a subset (`arf/features.py`)

### Fixed
- **MLP saw different features at inference than at training** — the pipeline never passed
  keyword/alias matches (so `keyword_match`, `alias_match` and `match_type` were always 0) and
  computed `score_gap_from_top` against the review band's own top instead of the query's top.
  Both now flow through `_mlp_triage`; `sample-project/train.py` passes the same global top
  (`arf/pipeline.py`, `sample-project/train.py`)
- **`FeatureExtractor` ignored `DocumentConfig`** — it read `document["title"]` directly (so
  `title_similarity`, `keyword_match`, `alias_match`, `match_type` and `bias_adjustment` were 0 for
  any schema with a custom `title_field`), counted depth against a fixed legal hierarchy instead
  of `config.hierarchy`, and read embeddings from a hard-coded `"embedding"` key. All lookups now
  honour the config (`arf/features.py`)
- **Pipeline dedupe ignored `DocumentConfig.id_field`** — documents were keyed by `_id`/`id`/title,
  so with a custom id field two different documents sharing a title collapsed into one
  (`arf/pipeline.py`)
- **Cache lookups and stores used different keys** — lookups lower-cased the query but stores did
  not, so a plain dict/Redis cache never hit on mixed-case queries (`arf/pipeline.py`)
- **Cache entries broke after a JSON round-trip** — `(item, score)` tuples came back from
  Redis/Mongo as lists and were served as a "document" that was a list with score 1.0
- **A cache hit re-ran `summarize_fn`** (an LLM call per result) and ignored `top_k` — a stored
  top-5 answer was returned for a top-10 request. A failing `cache_lookup` also took the whole
  query down; it is now reported and bypassed (`arf/pipeline.py`)
- **Score parser rejected common LLM answers** — `Score: 8 (high)`, `**Score:** 7`, `Score: 7/10`,
  `7.5`, JSON inside ` ``` ` fences or prose, and `the score is 7` all raised `ValueError` (and
  therefore dropped the document); `"true"` parsed as 1. All are handled now: fractions on other
  scales are rescaled to 0-9, values round half-up, the last `Score:` line wins and booleans are
  rejected. `adjust_scores` raises on mismatched list lengths instead of silently truncating
  (`arf/score_parser.py`)
- **LLM-failure fallback could never trigger** — it kept a candidate only if its score was already
  at or above `accept_threshold`, which no LLM-bound candidate can satisfy, so every failed or
  unparseable LLM call silently dropped the document. Replaced by `llm_error_policy`
  (`arf/pipeline.py`)
- **Threshold-calibration warning** — the pipeline logs a one-time warning (with the top observed
  score) when triage rejects every candidate, which usually means the thresholds don't match the
  embedding model's score range (`arf/pipeline.py`)
- README: threshold table and "calibrate to your embedding model" guidance

## [0.2.2] - 2026-04-06

### Added
- MLP hyperparameter optimization (`alpha`, `learning_rate_init`, `activation`) and threshold tuning
  (`arf/trainer.py`, `benchmarks/train_reranker.py`, `config_schema.py`)

## [0.2.1] - 2026-04-04

### Added
- `arf` pip-installable library package with a zero-dependency core (`arf/`)
- `sample-project/`: cooking-recipe search with FAISS, Voyage AI and OpenAI
- GitHub Actions workflow publishing to PyPI on release (`.github/workflows/publish.yml`)
- README rewritten for the library package

## [0.1.0] - 2026-03-16

### Added
- **Benchmark evaluation framework** with Precision@k, Recall@k, MRR, NDCG metrics (`benchmarks/`)
- **100-query US Constitution test set** across 5 difficulty levels (`benchmarks/us_constitution_test_set.csv`)
- **Hallucination evaluation** using LLM-as-judge faithfulness scoring (`benchmarks/hallucination_eval.py`)
- **Cost tracking** instrumentation for token usage, LLM call frequency, and cache hit rates (`benchmarks/cost_tracker.py`)
- **Pydantic config validation** with range checks and threshold ordering constraints (`config_schema.py`)
- **Mermaid architecture diagram** in README showing full query processing pipeline
- **Benchmark template** with comparison tables for retrieval strategies
- **Docker support** with `Dockerfile` and `docker-compose.yml` for containerized development
- **GitHub Actions CI** with Ruff linting, format checks, pytest on Python 3.10/3.12, and README badge
- **Integration test framework** with 13 offline tests and live test scaffolding (`tests/test_integration.py`)
- **Unit tests** for KeywordMatcher (37 tests), metrics (16 tests), cost tracker (4 tests), config schema (7 tests)
- **Sample fixture dataset** with 10 US Constitution documents for offline testing (`fixtures/`)
- `pyproject.toml` for dependency management and installability
- `.env.example` documenting required environment variables
- MIT `LICENSE` file

### Fixed
- "hullsination" typo in README corrected to "hallucination"

## [0.0.1] - 2025-02-11

### Added
- Initial RAG framework with multi-strategy retrieval
- MongoDB Atlas Vector Search integration with Voyage AI embeddings
- Query processing pipeline with rephrasing, threshold gates, and LLM reranking
- Support for US Constitution, US Code, CFR, USCIS Policy, Supreme Court cases
- Bilingual support (English/Spanish)
- Data ingestion scripts for all legal document sources
