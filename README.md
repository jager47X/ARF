# ARF - Advanced Retrieval Framework

[![CI](https://github.com/jager47X/ARF/actions/workflows/ci.yml/badge.svg)](https://github.com/jager47X/ARF/actions/workflows/ci.yml)

**ARF**  (Advanced Retrieval Framework) is a sophisticated Retrieval-Augmented Generation (RAG) system that designed to minimize the cost and hallucination based on R-Flow. I optimized for legal document search and analysis in this use. It provides intelligent semantic search, multi-strategy retrieval, and context-aware document summarization across multiple legal domains.

## 🚀 Live Demo

**Experience ARF in action:** [KnowYourRights.ai](https://knowyourrights-ai.com)

![KnowYourRights.ai Demo](media/demo_en.png)

*KnowYourRights.ai - AI-powered legal rights search and case intake platform powered by ARF*

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Components](#components)
- [Development](#development)
- [Contributing](#contributing)

## Overview

ARF is a production-ready RAG framework that enables:

- **Multi-domain legal document retrieval** across US Constitution, US Code, Code of Federal Regulations, USCIS Policy Manual, Supreme Court cases, and client cases
- **Intelligent semantic search** using MongoDB Atlas Vector Search with Voyage AI embeddings
- **Hybrid search strategies** combining semantic, keyword, alias, and exact matching
- **LLM-powered reranking** to improve result relevance and ordering
- **Bilingual support** (English/Spanish) for queries and responses
- **Automatic document ingestion** and embedding generation
- **Domain-specific threshold tuning** for optimal retrieval performance

## Features

### Core Capabilities

- **Semantic Vector Search**: MongoDB Atlas Vector Search with Voyage AI embeddings (voyage-3-large, 1024 dimensions)
- **Multi-Strategy Retrieval**:
  - Semantic similarity search
  - Keyword/BM25 matching (configurable per domain)
  - Alias-based search (for US Constitution)
  - Exact pattern matching
  - Hybrid search combining multiple strategies
- **Query Processing Pipeline**:
  - Query rephrasing and expansion
  - Multi-stage filtering with configurable thresholds
  - LLM reranking for borderline results
  - Result ranking and gap filtering
- **Intelligent Caching**: Query result caching and summary reuse
- **Bilingual Support**: English and Spanish query processing and response generation
- **Case-to-Document Mapping**: Automatic linking of Supreme Court cases to relevant constitutional provisions

### Domain-Specific Optimizations

- **US Constitution**: Alias search, keyword matching, structured article/section navigation
- **US Code**: Large-scale document handling with efficient indexing
- **Code of Federal Regulations (CFR)**: Hierarchical part/chapter/section organization
- **USCIS Policy Manual**: Automatic weekly updates, reference tracking
- **Supreme Court Cases**: Case-to-constitutional provision mapping
- **Client Cases**: SQL-based search for private case databases

## Architecture

### System Components

```
ARF/
├── RAG_interface.py          # Main orchestrator class
├── config.py                 # Configuration and collection definitions
├── rag_dependencies/         # Core RAG components
│   ├── mongo_manager.py      # MongoDB connection and query management
│   ├── vector_search.py      # MongoDB Atlas Vector Search implementation
│   ├── query_manager.py      # Query processing and normalization
│   ├── query_processor.py    # End-to-end query pipeline
│   ├── alias_manager.py      # Alias/keyword search for US Constitution
│   ├── keyword_matcher.py    # Structured keyword matching
│   ├── llm_verifier.py       # LLM-based result reranking
│   ├── openai_service.py     # OpenAI API integration
│   └── ai_service.py         # AI service abstraction
└── preprocess/               # Data ingestion scripts
    ├── us_constitution/      # US Constitution ingestion
    ├── us_code/              # US Code ingestion
    ├── cfr/                  # CFR ingestion
    ├── uscis_policy_manual/  # USCIS Policy Manual ingestion
    ├── supreme_court_cases/  # Supreme Court cases ingestion
    └── [other sources]/      # Additional data sources
```

### Query Processing Flow

```mermaid
flowchart TD
    A["User Query\n(en / es)"] --> B{"Language?"}
    B -- es --> C["Translate to English"]
    B -- en --> D["Normalize Query"]
    C --> D

    D --> E["Get / Create Embedding\n(Voyage-3-large, 1024d)"]
    E --> F{"Cached results\nexist?"}
    F -- yes --> G["Return cached results\n(skip all LLM calls)"]

    F -- no --> H["OpenAI Omni\nModeration Check"]
    H -- flagged --> I["Reject query"]
    H -- pass --> J["fix_query via LLM"]

    J --> K["Multi-Strategy Search"]

    K --> K1["Semantic Vector Search\n(MongoDB Atlas)"]
    K --> K2{"Alias search\nenabled?"}
    K --> K3{"Keyword matcher\nenabled?"}
    K2 -- yes --> K2a["Alias Embedding\nCosine Similarity"]
    K3 -- yes --> K3a["Article / Section\nPattern Match"]

    K1 --> L["Merge & Score"]
    K2a --> L
    K3a --> L

    L --> M{"score >= RAG_SEARCH\n(0.85)?"}
    M -- yes --> N["Accept result"]
    M -- no --> O{"score >= LLM_VERIF\n(0.70)?"}
    O -- yes --> P["LLM Reranking\n(borderline verification)"]
    O -- no --> Q{"Rephrase\nattempts left?"}
    P -- verified --> N
    P -- rejected --> Q
    Q -- yes --> R["Rephrase query\nvia LLM"] --> E
    Q -- no --> S["Return empty"]

    N --> T{"score >= confident\n(0.85)?"}
    T -- yes --> U["Cache results +\nGenerate & cache summary"]
    T -- no --> V["Return results\n(no cache)"]
    U --> W["Gap Filter\n(remove outliers)"]
    V --> W
    W --> X["Return ranked results\n+ bilingual summary"]
```

#### Flow summary

1. **Query Input** — user query with optional filters (jurisdiction, language, case filters)
2. **Query Normalization** — text normalization, pattern matching, domain detection
3. **Multi-Strategy Search** — semantic vector search (primary), alias search (if enabled), keyword matching (if enabled), exact pattern matching
4. **Result Filtering** — threshold-based filtering (domain-specific), LLM reranking for borderline results, gap filtering to remove outliers
5. **Result Ranking** — score-based ranking with bias adjustments
6. **Summary Generation** — LLM-powered document summaries (cached for reuse)
7. **Response Formatting** — bilingual response generation

## Installation

### Prerequisites

- Python 3.8+
- MongoDB Atlas account with vector search enabled
- OpenAI API key
- Voyage AI API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd arf
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Configure environment variables**:
   Create a `.env` file (or `.env.local`, `.env.dev`, `.env.production`) with:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   MONGO_URI=your_mongodb_atlas_connection_string
   ```

4. **Set up MongoDB Atlas**:
   - Create vector search indexes on your collections
   - Index name: `vector_index` (default)
   - Vector field: `embedding`
   - Dimensions: 1024

## Configuration

### Collection Configuration

Collections are defined in `config.py` with domain-specific settings:

```python
COLLECTION = {
    "US_CONSTITUTION_SET": {
        "db_name": "public",
        "main_collection_name": "us_constitution",
        "document_type": "US Constitution",
        "use_alias_search": True,
        "use_keyword_matcher": True,
        "thresholds": DOMAIN_THRESHOLDS["us_constitution"],
        # ... additional settings
    },
    # ... other collections
}
```

### Domain-Specific Thresholds

Each domain has optimized thresholds for:
- `query_search`: Initial semantic search threshold
- `alias_search`: Alias matching threshold
- `RAG_SEARCH_min`: Minimum score to continue processing
- `LLM_VERIFication`: Threshold for LLM reranking
- `RAG_SEARCH`: High-confidence result threshold
- `confident`: Threshold for saving summaries
- `FILTER_GAP`: Maximum score gap between results
- `LLM_SCORE`: LLM reranking score adjustment

### Environment Selection

The framework supports multiple environments:
- `--production`: Uses `.env.production`
- `--dev`: Uses `.env.dev`
- `--local`: Uses `.env.local`
- Auto-detection: Based on Docker environment and file existence

## Usage

### Basic Usage

```python
from RAG_interface import RAG
from config import COLLECTION

# Initialize RAG for a specific collection
rag = RAG(COLLECTION["US_CONSTITUTION_SET"], debug_mode=False)

# Process a query
results, query = rag.process_query(
    query="What does the 14th Amendment say about equal protection?",
    language="en"
)

# Get summary for a specific result
summary = rag.process_summary(
    query=query,
    result_list=results,
    index=0,
    language="en"
)
```

### Advanced Usage

```python
# With jurisdiction filtering
results, query = rag.process_query(
    query="immigration policy",
    jurisdiction="federal",
    language="en"
)

# Bilingual summary
insight_en, insight_es = rag.process_summary_bilingual(
    query=query,
    result_list=results,
    index=0,
    language="es"  # Returns both English and Spanish
)

# SQL-based client case search
rag_sql = RAG(COLLECTION["CLIENT_CASES"], debug_mode=False)
results = rag_sql.process_query(
    query="asylum case",
    filtered_cases=["case_id_1", "case_id_2"]
)
```

### Query Processing Options

- `skip_pre_checks`: Skip initial query validation
- `skip_cases_search`: Skip Supreme Court case search
- `filtered_cases`: Filter results to specific case IDs (SQL path)
- `jurisdiction`: Filter by jurisdiction
- `language`: Query language ("en" or "es")

## Data Sources

ARF supports ingestion from multiple legal document sources:

### Supported Sources

1. **US Constitution** (`preprocess/us_constitution/`)
   - Main constitutional text
   - Alias mappings for articles/sections
   - Supreme Court case references

2. **US Code** (`preprocess/us_code/`)
   - All 54 titles of the United States Code
   - XML to JSON conversion
   - Hierarchical clause organization

3. **Code of Federal Regulations** (`preprocess/cfr/`)
   - All CFR titles
   - Part/chapter/section structure
   - XML parsing and normalization

4. **USCIS Policy Manual** (`preprocess/uscis_policy_manual/`)
   - HTML to JSON conversion
   - Automatic weekly updates
   - Reference tracking to CFR

5. **Supreme Court Cases** (`preprocess/supreme_court_cases/`)
   - Public case database
   - Case-to-constitutional provision mapping

6. **California Codes** (`preprocess/ca_codes/`)
   - California State Codes
   - Multiple fetch strategies

7. **California Constitution** (`preprocess/ca_constitution/`)
   - State constitutional text

8. **Federal Register** (`preprocess/federal_register/`)
   - Federal Register documents

9. **Agency Guidance** (`preprocess/agency_guidance/`)
   - USCIS, DHS, ICE guidance documents

### Ingesting Data

See `preprocess/README.md` for detailed ingestion instructions. Example:

```bash
# Ingest US Constitution with embeddings
python preprocess/us_constitution/ingest_con_law.py --production --from-scratch --with-embeddings

# Ingest Supreme Court cases
python preprocess/supreme_court_cases/ingest_supreme_court_cases.py --production --with-embeddings
```

## Components

### RAG Interface (`RAG_interface.py`)

Main orchestrator class that wires all subsystems together:
- Collection configuration management
- Domain-specific threshold selection
- Component initialization
- Public API for query processing

### Query Processor (`query_processor.py`)

End-to-end query processing pipeline:
- Query normalization and expansion
- Multi-stage search execution
- Result filtering and ranking
- Summary generation and caching
- Case-to-document mapping

### Vector Search (`vector_search.py`)

MongoDB Atlas Vector Search implementation:
- Native `$vectorSearch` aggregation
- Score bias adjustments
- Efficient similarity search
- Error handling and retries

### Query Manager (`query_manager.py`)

Query processing utilities:
- Text normalization
- Pattern matching
- Query rephrasing
- Domain detection

### Alias Manager (`alias_manager.py`)

Alias-based search for US Constitution:
- Keyword/alias embeddings
- Fast alias matching
- Score boosting for exact matches

### Keyword Matcher (`keyword_matcher.py`)

Structured keyword matching:
- Article/section pattern matching
- Hierarchical document navigation
- Exact match detection

### LLM Verifier (`llm_verifier.py`)

LLM-based result reranking:
- Relevance scoring and reranking
- Borderline result reranking
- Confidence adjustment and score refinement

### Mongo Manager (`mongo_manager.py`)

MongoDB connection and query management:
- Database connections
- Collection access
- Query caching
- User query history

## Development

### Project Structure

```
arf/
├── RAG_interface.py          # Main entry point
├── config.py                 # Configuration
├── rag_dependencies/         # Core RAG modules
├── preprocess/               # Data ingestion
│   ├── [source]/            # Source-specific scripts
│   └── README.md            # Ingestion documentation
└── Data/                    # Knowledge base data
    └── Knowledge/           # Processed JSON files
```

### Running Tests

```bash
# Unit + integration tests (no API keys needed)
pytest tests/ -v

# Live integration tests (requires API keys + MongoDB)
ARF_LIVE_TESTS=1 pytest tests/test_integration.py -v

# Validate config schemas
python config_schema.py
```

### Adding New Data Sources

1. Create a new directory in `preprocess/`
2. Implement fetch and ingest scripts
3. Add collection configuration to `config.py`
4. Define domain-specific thresholds
5. Create vector search indexes in MongoDB Atlas

### Debugging

Enable debug mode for detailed logging:

```python
rag = RAG(COLLECTION["US_CONSTITUTION_SET"], debug_mode=True)
```

## Evaluation & Benchmarks

ARF includes a full evaluation framework. Run benchmarks with:

```bash
# Dry run (validate queries, no API calls)
python benchmarks/run_eval.py --dry-run

# Full evaluation against live system
python benchmarks/run_eval.py --production

# With hallucination measurement
python benchmarks/run_eval.py --production --eval-faithfulness

# Specific domain
python benchmarks/run_eval.py --production --domain us_constitution
```

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of top-k results that are relevant |
| Recall@k | Fraction of relevant documents found in top-k |
| MRR | Mean Reciprocal Rank — average of 1/rank of first relevant result |
| NDCG@k | Normalized Discounted Cumulative Gain |

### Benchmark: Retrieval Strategy Comparison

Measured on 15 US Constitution benchmark queries. Each strategy runs in its **own isolated RAG instance** — MongoDB Atlas strategies use direct `$vectorSearch` with no caching. Similar queries (50% random pick, ~1 word changed, ~90% embedding similarity) test whether ARF's cache improves accuracy over time.

| Strategy | N | MRR | P@1 | P@5 | R@5 | NDCG@5 | Avg Latency |
|----------|---|-----|-----|-----|-----|--------|-------------|
| MongoDB Atlas (Semantic Only) | 15 | 0.665 | 0.600 | 0.147 | 0.613 | 0.603 | 428 ms |
| MongoDB Atlas (Hybrid) | 15 | 0.665 | 0.600 | 0.147 | 0.613 | 0.603 | 410 ms |
| **Full ARF Pipeline** | 15 | 0.489 | 0.400 | 0.133 | 0.580 | 0.503 | 1,130 ms |
| **Full ARF Pipeline (similar queries)** | 7 | **0.679** | **0.571** | **0.171** | **0.743** | **0.682** | **1,065 ms** |

> **Key findings:**
> - **MongoDB Atlas `$vectorSearch`** provides a strong raw baseline (MRR 0.665) at ~400ms — but returns every result without quality filtering, including low-confidence matches.
> - **Full ARF Pipeline** deliberately filters out borderline results via threshold gates (`RAG_SEARCH ≥ 0.85`), trading recall for precision. This lowers MRR on initial queries (0.489).
> - **ARF improves over time**: when similar queries arrive (~1 word changed, ~90% embedding similarity), ARF's cache returns previously verified high-confidence results, boosting MRR to **0.679** and R@5 to **0.743** — surpassing raw Atlas on precision while maintaining quality guarantees.
> - **Latency**: similar queries run at 1,065ms vs 1,130ms cold. The cache skips vector search, moderation, and LLM reranking, but embedding lookup and document enrichment still add ~700ms.
>
> Run `python benchmarks/run_ablation.py --production` to reproduce.

### Hallucination & Faithfulness

Evaluated using an LLM-as-judge approach (`benchmarks/hallucination_eval.py`) — compares generated summaries against source documents to detect unsupported claims.

| Metric | Value |
|--------|-------|
| Faithfulness rate | **60%** (3/5 summaries fully faithful) |
| Hallucination rate | **40%** (2/5 contained unsupported claims) |
| Avg faithfulness score | **0.82** |

**Detected hallucinations:**
- *"the right to keep and bear arms for all citizens, **with no restrictions whatsoever**"* — the 2nd Amendment text does not say "no restrictions"
- *"Congress has the power to **declare war**"* — not present in the Article I Section 8 excerpt tested

> The evaluator correctly identifies when summaries add claims not supported by the source text. This validates ARF's threshold gates — by only returning high-confidence results (`score ≥ 0.85`), the system reduces the surface area for hallucination in downstream LLM summarization.

### Cost Analysis

| Metric | MongoDB Atlas (no pipeline) | Full ARF Pipeline |
|--------|---------------------------|-------------------|
| Embedding calls/query | 1 (always fresh) | 1 (cached in MongoDB after first call) |
| LLM calls/query | 0 | 0 on this benchmark; ~0.2 in production |
| LLM rerank frequency | N/A | 0% here; ~15-25% on ambiguous queries |
| Cache hit rate | N/A (no cache) | Grows with query volume |
| Avg latency (cold) | **410-428 ms** | 1,130 ms |
| Avg latency (similar query) | Same as cold | **1,065 ms** (cache hit) |
| Quality guarantee | None (returns all) | Only results with score ≥ 0.85 |

> **Cost thesis:** MongoDB Atlas alone is faster per-query but provides no quality filtering, caching, or hallucination reduction. ARF adds ~700ms overhead for threshold gating, embedding caching, and document enrichment — but **improves accuracy over time** as the cache accumulates verified results. In production with repeated/similar queries, the effective cost per high-quality result decreases as cache hit rate grows.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Add docstrings to public functions
- Include logging for important operations

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- MongoDB Atlas for vector search capabilities
- Voyage AI for embedding models
- OpenAI for LLM services

---

For detailed information on data ingestion, see [preprocess/README.md](preprocess/README.md).
