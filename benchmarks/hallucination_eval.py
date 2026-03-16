"""
Hallucination / faithfulness evaluation using LLM-as-judge.

Compares generated summaries against source documents to detect
unsupported claims. Can be used with any OpenAI-compatible LLM.

Usage:
    evaluator = FaithfulnessEvaluator(api_key="sk-...")
    result = evaluator.evaluate(
        source_text="The 14th Amendment guarantees equal protection...",
        generated_summary="The 14th Amendment requires states to provide equal protection...",
        query="What does the 14th Amendment say?"
    )
    print(result)
    # {"faithful": True, "score": 0.95, "unsupported_claims": [], ...}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a faithfulness evaluator for a legal RAG system.
Your job is to determine whether a generated summary is faithful to the source document.

A summary is FAITHFUL if every claim it makes is supported by the source document.
A summary is UNFAITHFUL if it contains claims not present in or contradicted by the source.

Respond with a JSON object:
{
    "faithful": true/false,
    "score": 0.0-1.0 (1.0 = perfectly faithful),
    "unsupported_claims": ["list of claims not supported by source"],
    "contradictions": ["list of claims that contradict the source"],
    "reasoning": "brief explanation"
}"""

JUDGE_USER_TEMPLATE = """Source document:
{source_text}

User query:
{query}

Generated summary:
{generated_summary}

Evaluate the faithfulness of the generated summary against the source document."""


@dataclass
class FaithfulnessResult:
    faithful: bool
    score: float
    unsupported_claims: List[str]
    contradictions: List[str]
    reasoning: str
    query: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "faithful": self.faithful,
            "score": self.score,
            "unsupported_claims": self.unsupported_claims,
            "contradictions": self.contradictions,
            "reasoning": self.reasoning,
            "query": self.query,
        }


class FaithfulnessEvaluator:
    """LLM-as-judge faithfulness evaluator."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        self._api_key = api_key

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                key = self._api_key or os.getenv("OPENAI_API_KEY")
                self._client = OpenAI(api_key=key)
            except ImportError:
                raise RuntimeError("openai package required: pip install openai")
        return self._client

    def evaluate(
        self,
        source_text: str,
        generated_summary: str,
        query: str = "",
    ) -> FaithfulnessResult:
        """Evaluate faithfulness of a single summary against its source."""
        prompt = JUDGE_USER_TEMPLATE.format(
            source_text=source_text[:4000],  # truncate for token limits
            query=query,
            generated_summary=generated_summary[:2000],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            return FaithfulnessResult(
                faithful=data.get("faithful", False),
                score=float(data.get("score", 0.0)),
                unsupported_claims=data.get("unsupported_claims", []),
                contradictions=data.get("contradictions", []),
                reasoning=data.get("reasoning", ""),
                query=query,
                raw_response=raw,
            )
        except Exception as e:
            logger.exception("Faithfulness evaluation failed: %s", e)
            return FaithfulnessResult(
                faithful=False, score=0.0,
                unsupported_claims=[], contradictions=[],
                reasoning=f"Evaluation error: {e}",
                query=query,
            )

    def evaluate_batch(
        self,
        items: List[Dict],
    ) -> Dict:
        """
        Evaluate a batch of (source, summary, query) triples.

        Args:
            items: list of dicts with keys: source_text, generated_summary, query

        Returns:
            Aggregate results with per-item details.
        """
        results = []
        for item in items:
            result = self.evaluate(
                source_text=item["source_text"],
                generated_summary=item["generated_summary"],
                query=item.get("query", ""),
            )
            results.append(result)

        faithful_count = sum(1 for r in results if r.faithful)
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        return {
            "total": len(results),
            "faithful_count": faithful_count,
            "hallucination_count": len(results) - faithful_count,
            "faithfulness_rate": round(faithful_count / len(results), 3) if results else 0.0,
            "hallucination_rate": round(1 - faithful_count / len(results), 3) if results else 0.0,
            "avg_faithfulness_score": round(avg_score, 3),
            "details": [r.to_dict() for r in results],
        }
