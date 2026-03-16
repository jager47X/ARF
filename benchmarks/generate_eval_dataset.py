#!/usr/bin/env python3
"""
Evaluation Dataset Generator and Validator for ARF.

Validates the evaluation dataset structure, expands it with negative examples
from MongoDB, and reports statistics.

Usage:
    python benchmarks/generate_eval_dataset.py --validate
    python benchmarks/generate_eval_dataset.py --expand --production
    python benchmarks/generate_eval_dataset.py --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import standalone_setup  # noqa: F401

logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

DOMAIN_TO_COLLECTION = {
    "us_constitution": "US_CONSTITUTION_SET",
    "code_of_federal_regulations": "CFR_SET",
    "us_code": "US_CODE_SET",
    "uscis_policy": "USCIS_POLICY_SET",
}


def load_dataset(path: Path = EVAL_DATASET_PATH) -> dict:
    """Load the evaluation dataset JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_dataset(path: Path = EVAL_DATASET_PATH) -> bool:
    """Validate dataset against Pydantic schema."""
    try:
        from benchmarks.eval_dataset_schema import dataset_statistics, validate_eval_dataset
        ds = validate_eval_dataset(path)
        stats = dataset_statistics(ds)
        print("\n  Dataset is VALID")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  By domain: {stats['by_domain']}")
        print(f"  By difficulty: {stats['by_difficulty']}")
        print(f"  Relevance distribution: {stats['relevance_distribution']}")
        return True
    except Exception as e:
        print(f"\n  VALIDATION FAILED: {e}")
        return False


def validate_titles_in_mongodb(path: Path = EVAL_DATASET_PATH, env: str = "production") -> None:
    """Check that expected_titles exist in actual MongoDB collections."""
    from config import COLLECTION, load_environment
    load_environment(env)
    import os

    from pymongo import MongoClient

    data = load_dataset(path)
    queries = data.get("queries", [])

    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["public"]

    missing_by_domain: Dict[str, List[str]] = {}
    checked = 0
    found = 0

    for q in queries:
        domain = q["domain"]
        collection_key = DOMAIN_TO_COLLECTION.get(domain)
        if not collection_key or collection_key not in COLLECTION:
            continue

        collection_name = COLLECTION[collection_key]["main_collection_name"]
        collection = db[collection_name]

        for doc in q.get("expected_docs", []):
            title = doc["title"]
            checked += 1
            exists = collection.find_one({"title": title}, {"_id": 1})
            if exists:
                found += 1
            else:
                missing_by_domain.setdefault(domain, []).append(title)

    client.close()

    print(f"\n  Title Validation: {found}/{checked} found in MongoDB")
    if missing_by_domain:
        print("  Missing titles by domain:")
        for domain, titles in sorted(missing_by_domain.items()):
            print(f"    {domain}: {len(titles)} missing")
            for t in titles[:5]:
                print(f"      - {t}")
            if len(titles) > 5:
                print(f"      ... and {len(titles) - 5} more")
    else:
        print("  All expected titles found in MongoDB!")


def expand_with_negatives(
    path: Path = EVAL_DATASET_PATH,
    env: str = "production",
    max_negatives_per_query: int = 3,
    seed: int = 42,
) -> None:
    """Expand dataset with random negative examples from MongoDB."""
    from config import COLLECTION, load_environment
    load_environment(env)
    import os

    from pymongo import MongoClient

    data = load_dataset(path)
    queries = data.get("queries", [])

    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["public"]
    rng = random.Random(seed)

    # Cache all titles per domain
    domain_titles: Dict[str, List[str]] = {}
    for domain, collection_key in DOMAIN_TO_COLLECTION.items():
        if collection_key not in COLLECTION:
            continue
        collection_name = COLLECTION[collection_key]["main_collection_name"]
        collection = db[collection_name]
        titles = collection.distinct("title")
        domain_titles[domain] = titles
        print(f"  {domain}: {len(titles)} documents available")

    added = 0
    for q in queries:
        domain = q["domain"]
        if domain not in domain_titles:
            continue

        # Get existing titles (positive + negative)
        existing_titles: Set[str] = set()
        for doc in q.get("expected_docs", []):
            existing_titles.add(doc["title"])
        for doc in q.get("negative_docs", []):
            existing_titles.add(doc["title"])

        # Pick random negatives that aren't already listed
        available = [t for t in domain_titles[domain] if t not in existing_titles]
        if not available:
            continue

        n_to_add = max_negatives_per_query - len(q.get("negative_docs", []))
        if n_to_add <= 0:
            continue

        negatives = rng.sample(available, min(n_to_add, len(available)))
        if "negative_docs" not in q:
            q["negative_docs"] = []
        for title in negatives:
            q["negative_docs"].append({"title": title, "relevance": 0})
            added += 1

    client.close()

    # Save updated dataset
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  Added {added} negative examples across {len(queries)} queries")
    print(f"  Dataset saved to {path}")


def print_statistics(path: Path = EVAL_DATASET_PATH) -> None:
    """Print comprehensive dataset statistics."""
    data = load_dataset(path)
    queries = data.get("queries", [])

    if not queries:
        print("  No queries in dataset.")
        return

    domains = Counter(q["domain"] for q in queries)
    difficulties = Counter(q["difficulty"] for q in queries)
    tags = Counter(tag for q in queries for tag in q.get("tags", []))

    n_with_negatives = sum(1 for q in queries if q.get("negative_docs"))
    total_expected = sum(len(q.get("expected_docs", [])) for q in queries)
    total_negatives = sum(len(q.get("negative_docs", [])) for q in queries)

    relevance_dist = Counter()
    for q in queries:
        for doc in q.get("expected_docs", []):
            relevance_dist[doc["relevance"]] += 1

    print(f"\n{'='*60}")
    print("EVALUATION DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Total queries:          {len(queries)}")
    print(f"  Total expected docs:    {total_expected}")
    print(f"  Total negative docs:    {total_negatives}")
    print(f"  Queries with negatives: {n_with_negatives} ({n_with_negatives/len(queries)*100:.0f}%)")

    print("\n  By Domain:")
    for domain, count in sorted(domains.items()):
        print(f"    {domain:<40} {count:>4}")

    print("\n  By Difficulty:")
    for diff, count in sorted(difficulties.items()):
        pct = count / len(queries) * 100
        print(f"    {diff:<15} {count:>4} ({pct:.0f}%)")

    print("\n  Relevance Distribution (expected_docs):")
    for rel in sorted(relevance_dist.keys()):
        labels = {3: "perfect", 2: "highly relevant", 1: "marginally relevant", 0: "not relevant"}
        print(f"    {rel} ({labels.get(rel, '?'):<20}): {relevance_dist[rel]:>4}")

    print("\n  Top Tags:")
    for tag, count in tags.most_common(15):
        print(f"    {tag:<25} {count:>4}")

    print(f"{'='*60}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ARF Evaluation Dataset Manager")
    parser.add_argument("--validate", action="store_true", help="Validate dataset schema")
    parser.add_argument("--validate-titles", action="store_true", help="Check titles exist in MongoDB")
    parser.add_argument("--expand", action="store_true", help="Expand with negative examples from MongoDB")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--dataset", type=str, default=str(EVAL_DATASET_PATH), help="Dataset path")
    parser.add_argument("--production", action="store_const", const="production", dest="env")
    parser.add_argument("--dev", action="store_const", const="dev", dest="env")
    parser.add_argument("--local", action="store_const", const="local", dest="env")
    args = parser.parse_args()

    env = args.env or "production"
    path = Path(args.dataset)

    if not any([args.validate, args.validate_titles, args.expand, args.stats]):
        args.stats = True  # Default action

    if args.validate:
        print("\n  Validating dataset schema...")
        validate_dataset(path)

    if args.validate_titles:
        print("\n  Validating titles in MongoDB...")
        validate_titles_in_mongodb(path, env=env)

    if args.expand:
        print("\n  Expanding dataset with negative examples...")
        expand_with_negatives(path, env=env)

    if args.stats:
        print_statistics(path)


if __name__ == "__main__":
    main()
