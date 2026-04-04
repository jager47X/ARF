"""Parse LLM output into numeric scores and blend with retrieval scores.

Extracts a 0-9 relevance score from messy LLM text (JSON, bare numbers,
"Score: N" lines) and converts it to a multiplicative adjustment that is
applied to the original retrieval score.

All functions are pure — no LLM calls, no I/O.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_SCORE_LINE_RX = re.compile(r"(?im)^\s*score\s*:\s*(-?\d{1,2})\s*$")
_JSON_SCORE_KEYS = ("score", "rating", "relevance")


def extract_score(text: str) -> int:
    """Extract a 0-9 integer score from raw LLM output.

    Supports three formats (tried in order):
      1. JSON object with a score key: ``{"score": 7}``
      2. Bare numeric string: ``"3"``
      3. A line matching ``Score: N``

    Args:
        text: Raw string output from an LLM.

    Returns:
        An integer clamped to [0, 9].

    Raises:
        ValueError: If no parseable score is found.
    """
    s = unicodedata.normalize("NFKC", str(text or "")).strip()

    # 1) JSON (object OR scalar)
    try:
        obj = json.loads(s)
        if isinstance(obj, (int, float)):
            v = int(obj)
            return max(0, min(9, v))
        if isinstance(obj, dict):
            for key in _JSON_SCORE_KEYS:
                if key in obj:
                    v = int(float(obj[key]))
                    return max(0, min(9, v))
    except Exception:
        pass

    # 2) Bare numeric string: "2"
    if re.fullmatch(r"-?\d{1,2}", s):
        v = int(s)
        return max(0, min(9, v))

    # 3) Anchored "Score: N" line
    m = _SCORE_LINE_RX.search(s)
    if m:
        v = int(m.group(1))
        return max(0, min(9, v))

    raise ValueError(
        "No parseable score (JSON scalar/object, numeric string, or 'Score:' line) found."
    )


def multiplier(score: float, *, min_mult: float = 0.50, max_mult: float = 1.50) -> float:
    """Convert a 0-9 score to a multiplicative factor.

    Linear interpolation::

        score 0 -> min_mult  (penalise)
        score 9 -> max_mult  (boost)

    Args:
        score: Numeric score in [0, 9].
        min_mult: Multiplier for score 0.
        max_mult: Multiplier for score 9.
    """
    s = max(0.0, min(9.0, float(score)))
    t = s / 9.0
    return min_mult + (max_mult - min_mult) * t


def adjust_score(
    base_score: float,
    llm_output: str,
    *,
    min_mult: float = 0.50,
    max_mult: float = 1.50,
) -> float:
    """Parse an LLM output, compute the multiplier, and blend with *base_score*.

    Returns ``min(base_score * mult, 1.0)``.

    Args:
        base_score: Original retrieval score (0-1).
        llm_output: Raw string from LLM containing a score.
        min_mult: Multiplier floor.
        max_mult: Multiplier ceiling.

    Raises:
        ValueError: If the LLM output contains no parseable score.
    """
    raw_score = extract_score(llm_output)
    mult = multiplier(raw_score, min_mult=min_mult, max_mult=max_mult)
    return min(base_score * mult, 1.0)


def adjust_scores(
    candidates: list,
    llm_outputs: list,
    *,
    key: object = None,
    min_mult: float = 0.50,
    max_mult: float = 1.50,
) -> list:
    """Batch-adjust scores for a list of ``(item, base_score)`` tuples.

    Args:
        candidates: List of ``(item, base_score)`` tuples.
        llm_outputs: Parallel list of raw LLM output strings.
        key: Unused, reserved for future use.
        min_mult: Multiplier floor.
        max_mult: Multiplier ceiling.

    Returns:
        List of ``(item, adjusted_score)`` tuples.  If parsing fails for an
        entry the original score is kept.
    """
    results = []
    for (item, base_score), llm_text in zip(candidates, llm_outputs):
        try:
            adj = adjust_score(base_score, llm_text, min_mult=min_mult, max_mult=max_mult)
        except ValueError:
            logger.warning("Could not parse LLM output; keeping base score %.4f", base_score)
            adj = base_score
        results.append((item, adj))
    return results
