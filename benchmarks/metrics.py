"""
Retrieval evaluation metrics: Precision@k, Recall@k, MRR, NDCG.

All functions take:
    retrieved  – ordered list of document titles returned by the system
    relevant   – set/list of ground-truth relevant titles
"""

from __future__ import annotations

import math
from typing import List, Set, Union


def precision_at_k(retrieved: List[str], relevant: Union[Set[str], List[str]], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    relevant = set(relevant)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc in top_k if doc in relevant) / k


def recall_at_k(retrieved: List[str], relevant: Union[Set[str], List[str]], k: int) -> float:
    """Fraction of relevant documents found in top-k."""
    relevant = set(relevant)
    if not relevant:
        return 1.0  # vacuously true
    top_k = retrieved[:k]
    return sum(1 for doc in top_k if doc in relevant) / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Union[Set[str], List[str]]) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    relevant = set(relevant)
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def mrr(results: List[tuple]) -> float:
    """Mean Reciprocal Rank over a list of (retrieved, relevant) pairs."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)


def ndcg_at_k(retrieved: List[str], relevant: Union[Set[str], List[str]], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    relevant = set(relevant)
    top_k = retrieved[:k]

    # DCG
    dcg = sum(
        (1.0 if doc in relevant else 0.0) / math.log2(i + 2)
        for i, doc in enumerate(top_k)
    )

    # Ideal DCG: all relevant docs at the top
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_metrics(
    retrieved: List[str],
    relevant: Union[Set[str], List[str]],
    k_values: List[int] = None,
) -> dict:
    """Compute all metrics for a single query. Returns a flat dict."""
    if k_values is None:
        k_values = [1, 3, 5, 10]
    relevant = set(relevant)
    result = {"rr": reciprocal_rank(retrieved, relevant)}
    for k in k_values:
        result[f"p@{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"r@{k}"] = recall_at_k(retrieved, relevant, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return result
