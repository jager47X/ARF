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

### Added
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
