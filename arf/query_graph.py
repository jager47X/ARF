"""Graph-based semantic query cache with rephrase chain traversal.

Walks a directed graph of query nodes linked by rephrase edges.  At each
node, checks for cached results (early exit) and follows rephrase links
until results are found, a loop is detected, or *max_hops* is reached.

The algorithm is storage-agnostic: the caller provides a *lookup_fn* that
retrieves a node dict from whatever backend they use (MongoDB, Redis,
SQLite, in-memory dict, etc.).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChainResult:
    """Result of walking a rephrase chain.

    Attributes:
        final_text: The query text at the point where walking stopped.
        chain: Ordered list of intermediate query texts traversed.
        hops: Number of rephrase edges followed.
        loop_detected: ``True`` if a cycle was found in the graph.
        hit_max_hops: ``True`` if *max_hops* was reached before resolution.
        cached_results: The cached results if found, otherwise ``None``.
    """

    final_text: str
    chain: List[str] = field(default_factory=list)
    hops: int = 0
    loop_detected: bool = False
    hit_max_hops: bool = False
    cached_results: Optional[List[Any]] = None

    @property
    def hit(self) -> bool:
        """Whether cached results were found."""
        return self.cached_results is not None


def follow_rephrase_chain(
    seed: str,
    lookup_fn: Callable[[str], Optional[dict]],
    *,
    normalize_fn: Callable[[str], str] = str.lower,
    max_hops: int = 3,
) -> ChainResult:
    """Walk a rephrase graph starting from *seed*.

    At each node the *lookup_fn* is called with the normalised query text.
    It should return a dict with optional keys:

    * ``"results"`` — cached results (any truthy value triggers early exit).
    * ``"next"`` — the normalised text of the next node in the chain.

    If ``"results"`` is present and truthy, walking stops immediately and
    the results are returned (cache hit).  If ``"next"`` is present the
    walk continues to that node.  Walking also stops on loop detection or
    when *max_hops* is exhausted.

    Args:
        seed: The original query text.
        lookup_fn: ``(normalised_text) -> {"results": ..., "next": ...}``
            or ``None``.
        normalize_fn: Text normalisation applied before lookup (default:
            ``str.lower``).
        max_hops: Maximum number of rephrase edges to follow.

    Returns:
        A :class:`ChainResult` describing the outcome.
    """
    final = seed
    chain: List[str] = []
    hops = 0
    loop_detected = False
    hit_max_hops = False
    seen_norm = {normalize_fn(seed)}

    while True:
        norm_final = normalize_fn(final)
        node = lookup_fn(norm_final)

        if node is None:
            break

        # Cache hit?
        results = node.get("results")
        if results:
            logger.debug("Cache hit for %r at hop %d -> %d results", final, hops, len(results))
            return ChainResult(
                final_text=final,
                chain=chain,
                hops=hops,
                loop_detected=loop_detected,
                hit_max_hops=hit_max_hops,
                cached_results=results,
            )

        # Follow rephrase edge
        nxt = node.get("next")
        if not nxt:
            break

        nxt_norm = normalize_fn(nxt)
        if nxt_norm in seen_norm:
            loop_detected = True
            logger.debug("Loop detected: %r -> %r; stopping.", final, nxt)
            break

        if hops >= max_hops:
            hit_max_hops = True
            logger.debug("Reached max_hops=%d; stopping.", max_hops)
            break

        seen_norm.add(nxt_norm)
        chain.append(nxt)
        final = nxt
        hops += 1

    return ChainResult(
        final_text=final,
        chain=chain,
        hops=hops,
        loop_detected=loop_detected,
        hit_max_hops=hit_max_hops,
        cached_results=None,
    )
