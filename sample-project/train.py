"""Train an MLP reranker on the cooking recipe dataset using ARF.

Usage:
    python sample-project/train.py

Requires: the FAISS index from ingest.py to be built first.
Reads API keys from ../.env
Outputs:  sample-project/recipe_reranker.joblib
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

from arf import DocumentConfig, FeatureExtractor
from arf.trainer import train_reranker

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
EMBED_MODEL = "voyage-3-large"
EMBED_DIM = 1024

OUT_DIR = Path(__file__).resolve().parent
INDEX_PATH = OUT_DIR / "faiss.index"
DOCS_PATH = OUT_DIR / "docs.json"
MODEL_PATH = OUT_DIR / "recipe_reranker.joblib"

recipe_config = DocumentConfig(
    id_field="title",
    title_field="title",
    text_fields=["text"],
    children_fields=["steps"],
    hierarchy=["cuisine", "category", "title"],
    domain_id=0,
)


def embed_fn(text: str) -> list[float]:
    result = voyage.embed([text], model=EMBED_MODEL)
    return result.embeddings[0]


def main():
    from data import TRAINING_DATA

    # Load stored docs and FAISS index
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    stored_docs = data["docs"]
    index_map = data["index_map"]

    index = faiss.read_index(str(INDEX_PATH))

    # Build title -> doc lookup
    title_to_doc = {doc["title"]: doc for doc in stored_docs}

    # Feature extractor
    extractor = FeatureExtractor(recipe_config)

    # Generate feature vectors from training data
    print(f"Generating features for {len(TRAINING_DATA)} training pairs...")
    X_list = []
    y_list = []
    skipped = 0

    # Cache query embeddings to avoid redundant API calls
    query_cache: dict[str, list[float]] = {}

    for query, recipe_title, relevant in TRAINING_DATA:
        doc = title_to_doc.get(recipe_title)
        if not doc:
            skipped += 1
            continue

        # Get query embedding (cached)
        if query not in query_cache:
            query_cache[query] = embed_fn(query)
        q_emb = query_cache[query]

        # Search FAISS to get the score for this doc
        q_vec = np.array([q_emb], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = index.search(q_vec, index.ntotal)

        # Find this doc's score in search results
        doc_score = 0.0
        for score_val, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(index_map):
                result_doc = stored_docs[index_map[idx]]
                if result_doc["title"] == recipe_title:
                    doc_score = float(score_val)
                    break

        features = extractor.extract_features(
            query=query,
            document=doc,
            semantic_score=doc_score,
            query_embedding=q_emb,
        )
        vector = extractor.to_vector(features)
        X_list.append(vector)
        y_list.append(1 if relevant else 0)

    if skipped:
        print(f"  Skipped {skipped} pairs (recipe not found)")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    print(f"Training MLP on {X.shape[0]} samples ({sum(y)} positive, {sum(1-y)} negative)...")
    result = train_reranker(
        X, y,
        architecture=(64, 32, 16),
        max_iter=500,
        calibrate=True,
        feature_names=FeatureExtractor.feature_names(),
        save_path=str(MODEL_PATH),
    )

    print("\nTraining metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
