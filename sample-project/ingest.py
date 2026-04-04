"""Ingest cooking recipes into a FAISS index using ARF + Voyage AI embeddings.

Usage:
    python sample-project/ingest.py

Reads API keys from ../.env
Outputs:  sample-project/faiss.index  (FAISS index)
          sample-project/docs.json    (stored documents)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root so `arf` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import voyageai
from dotenv import load_dotenv

from arf import DocumentConfig, ingest_documents

# Load API keys from project root .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise RuntimeError("VOYAGE_API_KEY not set in .env")

# Voyage client
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
EMBED_MODEL = "voyage-3-large"
EMBED_DIM = 1024

# Where to store outputs
OUT_DIR = Path(__file__).resolve().parent
INDEX_PATH = OUT_DIR / "faiss.index"
DOCS_PATH = OUT_DIR / "docs.json"

# Document config for recipes
recipe_config = DocumentConfig(
    id_field="title",          # use title as unique ID
    title_field="title",
    text_fields=["text"],
    children_fields=["steps"],
    hierarchy=["cuisine", "category", "title"],
    domain_id=0,
)


def embed_fn(text: str) -> list[float]:
    """Embed a single text using Voyage AI."""
    result = voyage.embed([text], model=EMBED_MODEL)
    return result.embeddings[0]


def main():
    from data import RECIPES

    # Collect all documents that store_fn will receive
    stored_docs: list[dict] = []

    def store_fn(doc: dict):
        stored_docs.append(doc)

    print(f"Ingesting {len(RECIPES)} recipes...")
    result = ingest_documents(
        RECIPES,
        config=recipe_config,
        embed_fn=embed_fn,
        store_fn=store_fn,
    )
    print(f"  Processed: {result.processed}, Skipped: {result.skipped}, Errors: {result.errors}")

    if result.errors:
        for idx, err in result.error_details:
            print(f"  Error at index {idx}: {err}")

    # Build FAISS index from parent embeddings
    embeddings = []
    doc_index_map = []  # maps FAISS row -> doc index in stored_docs

    for i, doc in enumerate(stored_docs):
        emb = doc.get("embedding")
        if emb:
            embeddings.append(emb)
            doc_index_map.append(i)

    if not embeddings:
        print("No embeddings generated. Check your API key.")
        return

    X = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(X)  # normalize for cosine similarity via inner product

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(X)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"  FAISS index saved: {INDEX_PATH} ({index.ntotal} vectors)")

    # Save documents (strip embeddings to save space)
    docs_for_storage = []
    for doc in stored_docs:
        clean = {k: v for k, v in doc.items() if k != "embedding"}
        # Also strip child embeddings
        for field in recipe_config.children_fields:
            children = clean.get(field)
            if children and isinstance(children, list):
                clean[field] = [
                    {k: v for k, v in child.items() if k != "embedding"}
                    if isinstance(child, dict) else child
                    for child in children
                ]
        docs_for_storage.append(clean)

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump({"docs": docs_for_storage, "index_map": doc_index_map}, f, indent=2, ensure_ascii=False)
    print(f"  Documents saved: {DOCS_PATH} ({len(docs_for_storage)} docs)")
    print("Done!")


if __name__ == "__main__":
    main()
