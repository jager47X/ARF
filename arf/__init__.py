"""ARF — Advanced Retrieval Framework.

A dependency-free retrieval pipeline toolkit.  Plug in your own vector
search, embedding model, LLM, ML model, and storage backend — ARF
provides the routing algorithms, feature engineering, rephrase-graph
caching, and score blending.

Quick start::

    from arf import Pipeline, DocumentConfig, Triage

    pipeline = Pipeline(
        doc_config=DocumentConfig(title_field="title", text_fields=["text"]),
        triage=Triage(min_score=0.65, accept_threshold=0.85),
        search_fn=my_search,
        embed_fn=my_embed,
    )
    results = pipeline.run("what is due process")
"""

__version__ = "0.2.1"

from arf.document import Document, DocumentConfig
from arf.features import FeatureExtractor
from arf.ingest import IngestResult, ingest_documents
from arf.pipeline import Pipeline
from arf.query_graph import ChainResult, follow_rephrase_chain
from arf.score_parser import adjust_score, adjust_scores, extract_score, multiplier
from arf.triage import Triage, TriageResult

# trainer imports are lazy — they require numpy + sklearn
# Use: from arf.trainer import train_reranker, load_reranker

__all__ = [
    # Document model
    "Document",
    "DocumentConfig",
    # Features
    "FeatureExtractor",
    # Triage
    "Triage",
    "TriageResult",
    # Query graph
    "follow_rephrase_chain",
    "ChainResult",
    # Score parser
    "extract_score",
    "multiplier",
    "adjust_score",
    "adjust_scores",
    # Pipeline
    "Pipeline",
    # Ingest
    "ingest_documents",
    "IngestResult",
]
