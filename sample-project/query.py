"""Query the cooking recipe knowledge base using ARF pipeline.

Usage:
    python sample-project/query.py "how do I make a creamy curry?"
    python sample-project/query.py "best dumpling recipe"
    python sample-project/query.py "french pastry with butter"

Requires: ingest.py and train.py to have been run first.
Reads API keys from ../.env
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import voyageai
from dotenv import load_dotenv
from openai import OpenAI

from arf import DocumentConfig, Pipeline, Triage
from arf.trainer import load_reranker

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "voyage-3-large"
EMBED_DIM = 1024

OUT_DIR = Path(__file__).resolve().parent
INDEX_PATH = OUT_DIR / "faiss.index"
DOCS_PATH = OUT_DIR / "docs.json"
MODEL_PATH = OUT_DIR / "recipe_reranker.joblib"

# Load stored docs and FAISS index
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)
STORED_DOCS = _data["docs"]
INDEX_MAP = _data["index_map"]
FAISS_INDEX = faiss.read_index(str(INDEX_PATH))

# Document config
recipe_config = DocumentConfig(
    id_field="title",
    title_field="title",
    text_fields=["text"],
    children_fields=["steps"],
    hierarchy=["cuisine", "category", "title"],
    domain_id=0,
)


# ── User-provided functions ──────────────────────────────────────

def embed_fn(text: str) -> list[float]:
    result = voyage.embed([text], model=EMBED_MODEL)
    return result.embeddings[0]


def search_fn(embedding: list[float], top_k: int) -> list[tuple[dict, float]]:
    """Search FAISS index and return (doc_dict, score) pairs."""
    q_vec = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = FAISS_INDEX.search(q_vec, top_k)

    results = []
    for score_val, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(INDEX_MAP):
            doc = STORED_DOCS[INDEX_MAP[idx]]
            results.append((doc, float(score_val)))
    return results


def llm_fn(query: str, document: dict) -> str:
    """Ask OpenAI to rate relevance 0-9."""
    title = document.get("title", "Unknown")
    text = document.get("text", "")[:500]
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Rate how relevant this recipe is to the query on a scale of 0-9.\n"
                f"Query: {query}\n"
                f"Recipe: {title} — {text}\n"
                f"Respond with only: Score: N"
            ),
        }],
        max_tokens=20,
        temperature=0.0,
    )
    return response.choices[0].message.content


def summarize_fn(query: str, doc, context: list) -> str:
    """Generate a short summary of why this recipe matches the query."""
    title = doc.title if hasattr(doc, "title") else str(doc)
    content = doc.content if hasattr(doc, "content") else ""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"In 1-2 sentences, explain why this recipe answers the query.\n"
                f"Query: {query}\n"
                f"Recipe: {title}\n"
                f"Description: {content[:500]}\n"
            ),
        }],
        max_tokens=100,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ── In-memory cache ──────────────────────────────────────────────

_cache: dict[str, dict] = {}


def cache_lookup(query: str) -> dict | None:
    return _cache.get(query.lower())


def cache_store(query: str, results: list) -> None:
    _cache[query.lower()] = {"results": results}


# ── Build pipeline ───────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    # Load MLP reranker if available
    predict_fn = None
    if MODEL_PATH.exists():
        predict_fn = load_reranker(str(MODEL_PATH))
        print("  MLP reranker loaded.")
    else:
        print("  No MLP model found — using threshold + LLM only.")

    return Pipeline(
        doc_config=recipe_config,
        triage=Triage(
            min_score=0.30,          # FAISS cosine similarity is typically lower range
            accept_threshold=0.60,
            verify_threshold=0.40,
            gap=0.25,
            top_k=5,
        ),
        search_fn=search_fn,
        embed_fn=embed_fn,
        predict_fn=predict_fn,
        llm_fn=llm_fn,
        summarize_fn=summarize_fn,
        cache_lookup=cache_lookup,
        cache_store=cache_store,
        predict_zones=(0.4, 0.6),
        parser_range=(0.50, 1.50),
        max_rephrase=0,  # no rephrase for this demo
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python sample-project/query.py \"your query here\"")
        print("\nExamples:")
        print('  python sample-project/query.py "how do I make a creamy curry?"')
        print('  python sample-project/query.py "best dumpling recipe"')
        print('  python sample-project/query.py "french pastry with butter"')
        print('  python sample-project/query.py "spicy noodle soup"')
        print('  python sample-project/query.py "grilled meat with chimichurri"')
        return

    query = " ".join(sys.argv[1:])
    print(f"\nQuery: {query}")
    print("=" * 60)

    pipeline = build_pipeline()
    results = pipeline.run(query, top_k=5)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        doc = result["document"]
        score = result["score"]
        summary = result.get("summary", "")
        title = doc.title if hasattr(doc, "title") else "Unknown"
        cuisine = doc.metadata.get("cuisine", "") if hasattr(doc, "metadata") else ""
        category = doc.metadata.get("category", "") if hasattr(doc, "metadata") else ""

        print(f"\n#{i}  {title}")
        print(f"    Cuisine: {cuisine} | Category: {category}")
        print(f"    Score: {score:.4f}")
        if summary:
            print(f"    Summary: {summary}")
        print()


if __name__ == "__main__":
    main()
