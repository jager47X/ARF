# Changelog

All notable changes to ARF will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
