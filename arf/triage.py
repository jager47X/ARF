"""Candidate triage — threshold gates, gap filtering, and zone routing.

Splits retrieval candidates into *accepted*, *needs_review* (uncertain),
and *rejected* buckets using configurable score thresholds.  Includes
deduplication and gap filtering utilities.

All functions are pure — no I/O, no external dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Scored = Tuple[Any, float]


@dataclass
class TriageResult:
    """Outcome of candidate triage.

    Attributes:
        accepted: Candidates above *accept_threshold*.
        needs_review: Candidates between *verify_threshold* and *accept_threshold*.
        rejected: Candidates below *verify_threshold* (or below *min_score*,
            or outside the gap window).
    """

    accepted: List[Scored] = field(default_factory=list)
    needs_review: List[Scored] = field(default_factory=list)
    rejected: List[Scored] = field(default_factory=list)


class Triage:
    """Score-based candidate router.

    Args:
        min_score: Minimum score to consider at all (below → reject).
        accept_threshold: Score at or above which a candidate is accepted
            outright (no further verification needed).
        verify_threshold: Score at or above which a candidate should be
            reviewed (e.g. by MLP or LLM).  Between *verify_threshold*
            and *accept_threshold* is the "grey zone".
        gap: Maximum allowed score gap from the top result.  Candidates
            further away are rejected.
        top_k: Maximum number of results to return.
    """

    def __init__(
        self,
        *,
        min_score: float = 0.65,
        accept_threshold: float = 0.85,
        verify_threshold: float = 0.70,
        gap: float = 0.20,
        top_k: int = 20,
    ):
        self.min_score = min_score
        self.accept_threshold = accept_threshold
        self.verify_threshold = verify_threshold
        self.gap = gap
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Zone routing
    # ------------------------------------------------------------------

    def classify(
        self,
        candidates: List[Scored],
        *,
        key: Optional[Callable[[Scored], float]] = None,
    ) -> TriageResult:
        """Split candidates into accept / needs_review / reject.

        Args:
            candidates: List of ``(item, score)`` tuples.
            key: Optional callable to extract the score from a candidate.
                Defaults to ``lambda c: c[1]``.
        """
        if key is None:
            key = lambda c: c[1]  # noqa: E731

        accepted: List[Scored] = []
        needs_review: List[Scored] = []
        rejected: List[Scored] = []

        for candidate in candidates:
            score = key(candidate)
            if score < self.min_score:
                rejected.append(candidate)
            elif score >= self.accept_threshold:
                accepted.append(candidate)
            elif score >= self.verify_threshold:
                needs_review.append(candidate)
            else:
                rejected.append(candidate)

        return TriageResult(accepted=accepted, needs_review=needs_review, rejected=rejected)

    def by_zones(
        self,
        items: list,
        probabilities: List[float],
        *,
        zones: Tuple[float, float] = (0.4, 0.6),
    ) -> TriageResult:
        """Route items based on model probability zones.

        Args:
            items: Parallel list of items (any type).
            probabilities: Parallel list of probabilities in [0, 1].
            zones: ``(low, high)`` thresholds.  Below *low* → reject,
                above *high* → accept, between → needs_review.
        """
        low, high = zones
        accepted: List[Any] = []
        needs_review: List[Any] = []
        rejected: List[Any] = []

        for item, prob in zip(items, probabilities):
            if prob >= high:
                accepted.append(item)
            elif prob < low:
                rejected.append(item)
            else:
                needs_review.append(item)

        return TriageResult(accepted=accepted, needs_review=needs_review, rejected=rejected)

    # ------------------------------------------------------------------
    # Gap filter
    # ------------------------------------------------------------------

    def gap_filter(
        self,
        items: List[Scored],
        *,
        key: Optional[Callable[[Scored], float]] = None,
    ) -> List[Scored]:
        """Keep items within *self.gap* of the top score.

        Args:
            items: Sorted list of ``(item, score)`` tuples (highest first).
            key: Optional callable to extract score.
        """
        if not items:
            return []
        if key is None:
            key = lambda c: c[1]  # noqa: E731

        top_score = key(items[0])
        return [c for c in items if (top_score - key(c)) <= self.gap]

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def dedupe(
        *lists: List[Scored],
        key_fn: Optional[Callable[[Any], Any]] = None,
    ) -> List[Scored]:
        """Merge multiple scored lists, keeping the best score per key.

        Args:
            *lists: One or more lists of ``(item, score)`` tuples.
            key_fn: Callable that extracts a hashable dedup key from an item.
                Defaults to extracting ``item.get("id")`` or ``item.get("title")``
                or ``id(item)``.

        Returns:
            Merged list sorted by score descending.
        """
        if key_fn is None:
            key_fn = _default_key_fn

        best: Dict[Any, Scored] = {}
        for lst in lists:
            for item, score in (lst or []):
                k = key_fn(item)
                prev = best.get(k)
                if prev is None or score > prev[1]:
                    best[k] = (item, score)

        return sorted(best.values(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Full gate logic
    # ------------------------------------------------------------------

    def apply(
        self,
        candidates: List[Scored],
        *,
        review_fn: Optional[Callable[[List[Scored]], List[Scored]]] = None,
        key: Optional[Callable[[Scored], float]] = None,
        apply_gap: bool = True,
    ) -> List[Scored]:
        """Full triage pipeline: classify → review → merge → gap → top_k.

        Args:
            candidates: Raw ``(item, score)`` tuples from retrieval.
            review_fn: Optional function that takes ``needs_review`` candidates
                and returns them with adjusted scores.  This is where you'd
                plug in MLP scoring or LLM verification.
            key: Score extraction callable.
            apply_gap: Whether to apply gap filtering.

        Returns:
            Final list of accepted candidates, sorted by score, capped at
            *top_k*.
        """
        result = self.classify(candidates, key=key)

        reviewed: List[Scored] = []
        if result.needs_review and review_fn is not None:
            reviewed = review_fn(result.needs_review)
        elif result.needs_review:
            reviewed = result.needs_review

        merged = self.dedupe(result.accepted, reviewed)

        if apply_gap and merged:
            merged = self.gap_filter(merged, key=key)

        return merged[: self.top_k]


def _default_key_fn(item: Any) -> Any:
    if isinstance(item, dict):
        k = item.get("_id") or item.get("id") or item.get("title")
        if k is not None:
            return k
    return id(item)
