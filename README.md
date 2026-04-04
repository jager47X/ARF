# ARF — Advanced Retrieval Framework

[![CI](https://github.com/jager47X/ARF/actions/workflows/ci.yml/badge.svg)](https://github.com/jager47X/ARF/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/advanced-rag-framework.svg)](https://pypi.org/project/advanced-rag-framework/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **zero-dependency** retrieval pipeline toolkit. Plug in your own vector search, embedding model, LLM, ML model, and database — ARF provides the routing algorithms, feature engineering, rephrase-graph caching, and score blending.

```bash
pip install advanced-rag-framework
```

## What ARF Does

Most RAG pipelines send every candidate to an expensive LLM for reranking. ARF eliminates this waste with a multi-stage filtering pipeline called **R-Flow**:

```
Query
  → Cache graph walk (free — returns instantly if seen before)
  → Vector search (your provider)
  → Threshold + gap filter (free — drops obvious junk)
  → MLP triage (free, <5ms — accept/reject/uncertain)
  → LLM verification ($$$ — only for the ~20% uncertain candidates)
  → Answer with summaries
```

Each stage filters candidates so the next stage does less work. Only the uncertain ~20% ever reach the LLM.

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
    llm_fn=my_llm_verify,         # any LLM

    # Cache (optional)
    cache_lookup=my_cache_get,     # any cache backend
    cache_store=my_cache_set,

    # Preprocessing (optional)
    preprocess_fn=my_clean,        # translate, normalize, etc.
    moderate_fn=my_moderate,       # content safety
    rephrase_fn=my_rephrase,       # retry with rephrased query

    # Hierarchy (optional)
    resolve_fn=my_get_parent,      # walk up document tree
    summarize_fn=my_summarize,     # generate answer
)

results = pipeline.run("what is due process?", top_k=5)
# [{"document": Document, "score": 0.94, "context": [...], "summary": "..."}, ...]
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

### Features — 15-feature extraction

```python
from arf import FeatureExtractor

extractor = FeatureExtractor(config)
features = extractor.extract_features(query="...", document={...}, semantic_score=0.85)
vector = extractor.to_vector(features)  # [0.85, 4.2, 0, 0, ...]
```

| Feature | Description |
|---------|-------------|
| `semantic_score` | Raw cosine similarity from vector search |
| `bm25_score` | Term-frequency relevance approximation |
| `alias_match` | Whether query matches a document alias |
| `keyword_match` | Whether query matches via keyword pattern |
| `domain_type` | Encoded domain identifier |
| `document_length` | Log-scaled character count |
| `query_length` | Query character count |
| `section_depth` | Depth in document hierarchy |
| `embedding_cosine_similarity` | Direct embedding cosine similarity |
| `match_type` | 0=none, 1=partial, 2=exact |
| `score_gap_from_top` | Gap from highest-scored document |
| `query_term_coverage` | Fraction of query terms in document |
| `title_similarity` | Jaccard similarity between query and title |
| `has_nested_content` | Whether document has children |
| `bias_adjustment` | Configurable per-document bias |

### Triage — threshold + gap + zone routing

```python
from arf import Triage

triage = Triage(min_score=0.65, accept_threshold=0.85, verify_threshold=0.70, gap=0.20)
result = triage.classify(candidates)
# result.accepted, result.needs_review, result.rejected
```

### QueryGraph — rephrase chain walk

```python
from arf import follow_rephrase_chain

result = follow_rephrase_chain("due process clause", lookup_fn=my_db_lookup, max_hops=3)
# result.hit, result.cached_results, result.path, result.loop_detected
```

Walks a directed graph of query→rephrase edges with loop detection and early exit on cache hit. Storage-agnostic — you provide the `lookup_fn`.

### ScoreParser — LLM output parsing + multiplier blending

```python
from arf import extract_score, multiplier, adjust_score

extract_score('{"score": 7}')           # → 7
extract_score("Score: 8")               # → 8
multiplier(8)                           # → 1.39
adjust_score(0.72, "Score: 8")          # → min(0.72 * 1.39, 1.0)
```

Parses messy LLM output (JSON, bare numbers, "Score: N" lines) into a 0-9 score, converts to a multiplier, and blends with the retrieval score.

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

## Bring Your Own Everything

| Slot | What you provide | Examples |
|------|-----------------|----------|
| `search_fn` | Vector search | FAISS, Pinecone, Weaviate, Qdrant, MongoDB Atlas, pgvector |
| `embed_fn` | Embeddings | OpenAI, Voyage AI, Cohere, sentence-transformers, Ollama |
| `predict_fn` | ML model | scikit-learn, XGBoost, PyTorch, any callable |
| `llm_fn` | LLM verification | OpenAI, Anthropic, Ollama, Llama.cpp, any API |
| `cache_lookup/store` | Cache | Redis, MongoDB, SQLite, in-memory dict |
| `resolve_fn` | Parent lookup | Any database query |
| `summarize_fn` | Answer generation | Any LLM |
| `store_fn` (ingest) | Document storage | Any database write |

## Installation

```bash
# Core (zero dependencies)
pip install advanced-rag-framework

# With MLP training support (numpy + scikit-learn)
pip install advanced-rag-framework[ml]
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
                    │  (15 features)       │
                    └──────────┬───────────┘
                               │ feature vectors
                    ┌──────────▼───────────┐
                    │   MLP Reranker       │
                    │  (<5ms, $0.00)       │
                    └──────────┬───────────┘
                        ┌──────┼──────┐
                   p≥0.6│  0.4<p<0.6  │p≤0.4
                        │      │      │
                   Accept   ┌──▼──┐  Reject
                   (free)   │ LLM │  (free)
                            │(20%)│
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
