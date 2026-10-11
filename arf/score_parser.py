"""Parse LLM output into numeric scores and blend with retrieval scores.

Extracts a 0-9 relevance score from messy LLM text and converts it to a
multiplicative adjustment that is applied to the original retrieval score.

Accepted formats (see :func:`extract_score`): JSON objects or scalars, also
inside markdown code fences or embedded in prose; bare numbers; ``Score: N``
lines with trailing commentary; ``N/10``-style fractions (rescaled to 0-9);
and ``score is N`` / ``rating: N`` phrases.

All functions are pure — no LLM calls, no I/O.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FENCE_RX = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_BOLD_RX = re.compile(r"\*\*|__")
_FRACTION = r"(-?\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?"
_BARE_RX = re.compile(rf"^\s*{_FRACTION}\s*$")
# "Score: 8", "**Score:** 7/10", "Final score = 7.5 (strong match)"
_SCORE_LINE_RX = re.compile(rf"(?im)^\s*(?:final\s+)?score\s*[:=]\s*{_FRACTION}\b.*$")
# "the score is 7", "rating: 8/10", "relevance = 6" anywhere in prose
_SCORE_ANYWHERE_RX = re.compile(rf"(?i)\b(?:score|rating|relevance)\b\s*(?:is|:|=|of)?\s*{_FRACTION}\b")
_JSON_OBJECT_RX = re.compile(r"\{.*\}", re.DOTALL)
_JSON_SCORE_KEYS = ("score", "rating", "relevance")


def _clean(text: Any) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).strip()
    fenced = _FENCE_RX.match(s)
    if fenced:
        s = fenced.group(1).strip()
    return _BOLD_RX.sub("", s)


def _to_score(value: Any, denominator: Any = None) -> int:
    """Round a (possibly fractional, possibly on another scale) value into 0-9."""
    v = float(value)
    if denominator not in (None, "", 0, 0.0) and float(denominator) != 9.0:
        v = v / float(denominator) * 9.0
    return max(0, min(9, int(math.floor(v + 0.5))))


def _score_from_json(obj: Any) -> Optional[int]:
    if isinstance(obj, bool):  # bool is an int subclass — "true" is not a score
        return None
    if isinstance(obj, (int, float)):
        return _to_score(obj)
    if isinstance(obj, dict):
        for key in _JSON_SCORE_KEYS:
            val = obj.get(key)
            if isinstance(val, bool) or val is None:
                continue
            if isinstance(val, (int, float)):
                return _to_score(val)
            if isinstance(val, str):
                m = _BARE_RX.match(val)
                if m:
                    return _to_score(m.group(1), m.group(2))
    return None


def extract_score(text: str) -> int:
    """Extract a 0-9 integer score from raw LLM output.

    Formats are tried in order of confidence:

      1. The whole text is JSON — ``{"score": 7}``, ``{"rating": "8/10"}``,
         ``7`` — optionally wrapped in a markdown code fence.
      2. A bare number or fraction — ``"3"``, ``"7.5"``, ``"8/10"``.
      3. A ``Score: N`` line (``**Score:** 7``, ``Score = 8/10 (strong)`` ...);
         the last such line wins, since it is the final answer.
      4. A JSON object embedded in prose.
      5. ``score is N`` / ``rating: N`` / ``relevance = N`` anywhere.

    Fractions on another scale (``7/10``) are rescaled to 0-9; fractional
    values are rounded half-up; results are clamped to [0, 9].  Booleans
    are never treated as scores.

    Args:
        text: Raw string output from an LLM.

    Returns:
        An integer clamped to [0, 9].

    Raises:
        ValueError: If no parseable score is found.
    """
    s = _clean(text)

    # 1) The whole text is JSON (object or scalar)
    try:
        found = _score_from_json(json.loads(s))
    except (ValueError, TypeError):
        found = None
    if found is not None:
        return found

    # 2) Bare number / fraction
    m = _BARE_RX.match(s)
    if m:
        return _to_score(m.group(1), m.group(2))

    # 3) Anchored "Score: N" lines — the last one is the final answer
    lines = _SCORE_LINE_RX.findall(s)
    if lines:
        value, denominator = lines[-1]
        return _to_score(value, denominator)

    # 4) A JSON object embedded in prose
    embedded = _JSON_OBJECT_RX.search(s)
    if embedded:
        try:
            found = _score_from_json(json.loads(embedded.group(0)))
        except (ValueError, TypeError):
            found = None
        if found is not None:
            return found

    # 5) "score is N" anywhere in prose — last mention wins
    anywhere = _SCORE_ANYWHERE_RX.findall(s)
    if anywhere:
        value, denominator = anywhere[-1]
        return _to_score(value, denominator)

    raise ValueError(
        "No parseable score found (expected JSON, a bare number, a 'Score: N' line, or 'score is N')."
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

    Raises:
        ValueError: If the two lists differ in length.
    """
    if len(candidates) != len(llm_outputs):
        raise ValueError(f"candidates ({len(candidates)}) and llm_outputs ({len(llm_outputs)}) differ in length")
    results = []
    for (item, base_score), llm_text in zip(candidates, llm_outputs):
        try:
            adj = adjust_score(base_score, llm_text, min_mult=min_mult, max_mult=max_mult)
        except ValueError:
            logger.warning("Could not parse LLM output; keeping base score %.4f", base_score)
            adj = base_score
        results.append((item, adj))
    return results
