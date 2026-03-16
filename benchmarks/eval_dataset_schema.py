"""
Pydantic schema for the ARF evaluation dataset.

Validates the structure of eval_dataset.json entries, including
graded relevance labels, difficulty levels, and query type tags.

Usage:
    from benchmarks.eval_dataset_schema import EvalDataset, validate_eval_dataset
    dataset = validate_eval_dataset("benchmarks/eval_dataset.json")
    print(f"Loaded {len(dataset.queries)} queries")
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, Field, model_validator
except ImportError:
    print(
        "ERROR: pydantic>=2.0 is required. Install with: pip install pydantic>=2.0",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Enums for constrained fields
# ---------------------------------------------------------------------------

class Domain(str, Enum):
    US_CONSTITUTION = "us_constitution"
    CODE_OF_FEDERAL_REGULATIONS = "code_of_federal_regulations"
    US_CODE = "us_code"
    USCIS_POLICY = "uscis_policy"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QueryTag(str, Enum):
    """Recognized query-type tags. Additional freeform tags are allowed."""
    KEYWORD_EXACT = "keyword-exact"
    SEMANTIC = "semantic"
    MULTI_HOP = "multi-hop"
    AMBIGUOUS = "ambiguous"
    CROSS_DOMAIN = "cross-domain"
    NEGATION = "negation"
    ABBREVIATION = "abbreviation"
    ORDINAL = "ordinal"
    MISSPELLING = "misspelling"
    ADVERSARIAL = "adversarial"


# ---------------------------------------------------------------------------
# Document relevance models
# ---------------------------------------------------------------------------

class RelevanceDoc(BaseModel):
    """A document with a graded relevance label."""
    title: str = Field(..., min_length=1, description="Document title as stored in MongoDB")
    relevance: int = Field(
        ...,
        ge=0,
        le=3,
        description="Graded relevance: 3=perfect, 2=highly relevant, 1=marginally relevant, 0=not relevant",
    )


# ---------------------------------------------------------------------------
# Single evaluation query
# ---------------------------------------------------------------------------

class EvalQuery(BaseModel):
    """One query in the evaluation dataset."""
    id: str = Field(..., min_length=1, description="Unique query ID, e.g. USC-001")
    domain: Domain
    query: str = Field(..., min_length=1, description="The user query text")
    expected_docs: List[RelevanceDoc] = Field(
        ...,
        min_length=1,
        description="Documents expected to be relevant (relevance >= 1)",
    )
    negative_docs: List[RelevanceDoc] = Field(
        default_factory=list,
        description="Documents that should NOT match (relevance = 0)",
    )
    difficulty: Difficulty
    tags: List[str] = Field(default_factory=list, description="Query-type tags")
    notes: Optional[str] = Field(default=None, description="Human-readable notes about this query")

    @model_validator(mode="after")
    def check_expected_positive(self) -> "EvalQuery":
        """Ensure at least one expected_doc has relevance > 0."""
        if not any(d.relevance > 0 for d in self.expected_docs):
            raise ValueError(
                f"Query {self.id}: expected_docs must contain at least one doc with relevance > 0"
            )
        return self

    @model_validator(mode="after")
    def check_negatives_are_zero(self) -> "EvalQuery":
        """Ensure all negative_docs have relevance = 0."""
        bad = [d.title for d in self.negative_docs if d.relevance != 0]
        if bad:
            raise ValueError(
                f"Query {self.id}: negative_docs must have relevance=0, but got non-zero for: {bad}"
            )
        return self


# ---------------------------------------------------------------------------
# Full dataset
# ---------------------------------------------------------------------------

class EvalDataset(BaseModel):
    """The complete evaluation dataset."""
    model_config = {"populate_by_name": True}
    description: Optional[str] = Field(
        default=None,
        alias="_description",
        description="Human-readable description of the dataset",
    )
    version: str = Field(default="1.0", description="Dataset schema version")
    queries: List[EvalQuery]

    @model_validator(mode="after")
    def check_unique_ids(self) -> "EvalDataset":
        ids = [q.id for q in self.queries]
        dupes = [qid for qid in ids if ids.count(qid) > 1]
        if dupes:
            raise ValueError(f"Duplicate query IDs found: {set(dupes)}")
        return self


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def validate_eval_dataset(path: str | Path) -> EvalDataset:
    """Load and validate eval_dataset.json. Returns the parsed EvalDataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return EvalDataset(**data)


def dataset_statistics(dataset: EvalDataset) -> Dict:
    """Compute summary statistics for the dataset."""
    from collections import Counter

    stats: Dict = {
        "total_queries": len(dataset.queries),
        "by_domain": dict(Counter(q.domain.value for q in dataset.queries)),
        "by_difficulty": dict(Counter(q.difficulty.value for q in dataset.queries)),
        "by_tag": dict(Counter(tag for q in dataset.queries for tag in q.tags)),
        "total_expected_docs": sum(len(q.expected_docs) for q in dataset.queries),
        "total_negative_docs": sum(len(q.negative_docs) for q in dataset.queries),
        "relevance_distribution": dict(
            Counter(d.relevance for q in dataset.queries for d in q.expected_docs)
        ),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])

    print(f"Validating {dataset_path} ...")
    ds = validate_eval_dataset(dataset_path)
    stats = dataset_statistics(ds)

    print(f"\nDataset valid: {stats['total_queries']} queries")
    print(f"  By domain:     {stats['by_domain']}")
    print(f"  By difficulty:  {stats['by_difficulty']}")
    print(f"  Expected docs:  {stats['total_expected_docs']}")
    print(f"  Negative docs:  {stats['total_negative_docs']}")
    print(f"  Relevance dist: {stats['relevance_distribution']}")
    from collections import Counter as _Counter
    print(f"  Top tags:       {dict(_Counter(stats['by_tag']).most_common(10))}")
    print("\nAll entries valid.")
