# ARF - Advanced Retrieval Framework

**ARF**  (Advanced Retrieval Framework) is a sophisticated Retrieval-Augmented Generation (RAG) system that designed to minimize the cost and hullsination based on R-Flow. I optimized for legal document search and analysis in this use. It provides intelligent semantic search, multi-strategy retrieval, and context-aware document summarization across multiple legal domains.

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

1. **Query Input**: User query with optional filters (jurisdiction, language, case filters)
2. **Query Normalization**: Text normalization, pattern matching, domain detection
3. **Multi-Strategy Search**:
   - Semantic vector search (primary)
   - Alias search (if enabled)
   - Keyword matching (if enabled)
   - Exact pattern matching
4. **Result Filtering**:
   - Threshold-based filtering (domain-specific)
   - LLM reranking for borderline results
   - Gap filtering to remove outliers
5. **Result Ranking**: Score-based ranking with bias adjustments
6. **Summary Generation**: LLM-powered document summaries (cached for reuse)
7. **Response Formatting**: Bilingual response generation

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
   pip install -r requirements.txt
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
# Run preprocessing verification scripts
python preprocess/cfr/check_cfr_structure.py
python preprocess/us_code/verify_clause_numbers.py
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

[Specify your license here]

## Acknowledgments

- MongoDB Atlas for vector search capabilities
- Voyage AI for embedding models
- OpenAI for LLM services

---

For detailed information on data ingestion, see [preprocess/README.md](preprocess/README.md).
