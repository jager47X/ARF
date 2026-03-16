#!/usr/bin/env python3
"""
Identify duplicate titles in US Code MongoDB collection with detailed information.

This script shows which specific documents have duplicate titles, their IDs,
article+chapter+section information, and content previews to help identify
why they're duplicates.
"""

import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

from pymongo import MongoClient

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from services.rag.config import COLLECTION, MONGO_URI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Get US Code collection config
US_CODE_CONF = COLLECTION.get("US_CODE_SET")
if not US_CODE_CONF:
    raise ValueError("US_CODE_SET not found in COLLECTION config")

DB_NAME: str = US_CODE_CONF["db_name"]
COLL_NAME: str = US_CODE_CONF["main_collection_name"]


def get_document_preview(doc: Dict[str, Any], max_length: int = 100) -> str:
    """Get a preview of document content."""
    # Try to get text from clauses
    clauses = doc.get("clauses", [])
    if clauses and isinstance(clauses, list):
        for clause in clauses[:1]:  # Just first clause
            if isinstance(clause, dict):
                clause_text = clause.get("text") or clause.get("content") or clause.get("body")
                if clause_text:
                    text = str(clause_text).strip()
                    if len(text) > max_length:
                        return text[:max_length] + "..."
                    return text

    # Fallback to root-level text
    root_text = doc.get("text") or doc.get("summary") or doc.get("content") or doc.get("body")
    if root_text:
        text = str(root_text).strip()
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    return "(no text)"


def identify_duplicate_titles(client: MongoClient, limit: int = None) -> Dict[str, List[Dict[str, Any]]]:
    """Identify documents with duplicate titles and show details."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]

    logger.info("Finding documents with duplicate titles...")

    # Use aggregation to find duplicate titles
    pipeline = [
        {"$group": {
            "_id": "$title",
            "count": {"$sum": 1},
            "doc_ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}}
    ]

    if limit:
        pipeline.append({"$limit": limit})

    duplicate_groups = list(coll.aggregate(pipeline))

    logger.info(f"Found {len(duplicate_groups)} duplicate title groups")

    # Build detailed information for each duplicate group
    title_details: Dict[str, List[Dict[str, Any]]] = {}

    for group in duplicate_groups:
        title = group["_id"]
        doc_ids = group["doc_ids"]
        count = group["count"]

        # Fetch full documents for these IDs
        docs = list(coll.find({"_id": {"$in": doc_ids}}, {
            "_id": 1,
            "title": 1,
            "article": 1,
            "chapter": 1,
            "section": 1,
            "clauses": 1,
            "text": 1,
            "summary": 1
        }))

        # Add preview and key info to each doc
        detailed_docs = []
        for doc in docs:
            detailed_docs.append({
                "_id": str(doc.get("_id")),
                "title": doc.get("title"),
                "article": doc.get("article", ""),
                "chapter": doc.get("chapter", ""),
                "section": doc.get("section", ""),
                "clauses_count": len(doc.get("clauses", [])),
                "preview": get_document_preview(doc),
                "full_key": f"{doc.get('article', '')} | {doc.get('chapter', '')} | {doc.get('section', '')}"
            })

        title_details[title] = {
            "count": count,
            "documents": detailed_docs
        }

    return title_details


def print_duplicate_details(title_details: Dict[str, List[Dict[str, Any]]], max_titles: int = 20):
    """Print detailed information about duplicate titles."""
    logger.info("\n" + "=" * 100)
    logger.info("DUPLICATE TITLE DETAILS")
    logger.info("=" * 100)

    sorted_titles = sorted(title_details.items(), key=lambda x: x[1]["count"], reverse=True)

    for idx, (title, info) in enumerate(sorted_titles[:max_titles], 1):
        count = info["count"]
        docs = info["documents"]

        logger.info(f"\n[{idx}] Title: '{title}'")
        logger.info(f"    Appears {count} times")
        logger.info(f"    {'-' * 96}")

        # Group by article+chapter+section to see if they're truly duplicates
        key_groups = defaultdict(list)
        for doc in docs:
            key = doc["full_key"]
            key_groups[key].append(doc)

        if len(key_groups) == 1:
            logger.info(f"    ⚠️  ALL {count} DOCUMENTS HAVE THE SAME article+chapter+section!")
            logger.info(f"    Key: {list(key_groups.keys())[0]}")
        else:
            logger.info(f"    Found {len(key_groups)} different article+chapter+section combinations")

        # Show first few documents
        for i, doc in enumerate(docs[:10], 1):
            logger.info(f"    [{i}] ID: {doc['_id']}")
            logger.info(f"        Article: {doc['article']}")
            logger.info(f"        Chapter: {doc['chapter']}")
            logger.info(f"        Section: {doc['section']}")
            logger.info(f"        Clauses: {doc['clauses_count']}")
            logger.info(f"        Preview: {doc['preview']}")

        if len(docs) > 10:
            logger.info(f"    ... and {len(docs) - 10} more documents with this title")

        logger.info("")

    if len(sorted_titles) > max_titles:
        logger.info(f"\n... and {len(sorted_titles) - max_titles} more duplicate title groups")


def analyze_duplicate_patterns(title_details: Dict[str, List[Dict[str, Any]]]):
    """Analyze patterns in duplicate titles."""
    logger.info("\n" + "=" * 100)
    logger.info("DUPLICATE PATTERN ANALYSIS")
    logger.info("=" * 100)

    total_duplicates = 0
    same_key_duplicates = 0
    different_key_duplicates = 0

    for title, info in title_details.items():
        count = info["count"]
        docs = info["documents"]
        total_duplicates += count

        # Group by article+chapter+section
        key_groups = defaultdict(list)
        for doc in docs:
            key = doc["full_key"]
            key_groups[key].append(doc)

        if len(key_groups) == 1:
            same_key_duplicates += count
        else:
            different_key_duplicates += count

    logger.info(f"\nTotal duplicate documents: {total_duplicates}")
    logger.info(f"Documents with same article+chapter+section: {same_key_duplicates}")
    logger.info(f"Documents with different article+chapter+section: {different_key_duplicates}")

    # Find titles where all documents have the same key (true duplicates)
    true_duplicates = []
    for title, info in title_details.items():
        docs = info["documents"]
        key_groups = defaultdict(list)
        for doc in docs:
            key = doc["full_key"]
            key_groups[key].append(doc)

        if len(key_groups) == 1:
            true_duplicates.append({
                "title": title,
                "count": info["count"],
                "key": list(key_groups.keys())[0]
            })

    if true_duplicates:
        logger.info(f"\n⚠️  Found {len(true_duplicates)} titles where ALL documents have identical article+chapter+section:")
        for dup in sorted(true_duplicates, key=lambda x: x["count"], reverse=True)[:10]:
            logger.info(f"  - '{dup['title']}' ({dup['count']} documents) - Key: {dup['key']}")


def main():
    """Main function to identify duplicate titles."""
    import argparse

    parser = argparse.ArgumentParser(description="Identify duplicate titles in US Code collection")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of duplicate title groups to analyze (for testing)"
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum number of duplicate titles to display in detail (default: 20)"
    )

    args = parser.parse_args()

    client = None
    try:
        # Configure TLS for MongoDB Atlas connections
        tls_config = {}
        if MONGO_URI and "mongodb+srv://" in MONGO_URI:
            tls_config = {"tls": True}
        elif MONGO_URI and ("mongodb.net" in MONGO_URI or "mongodb.com" in MONGO_URI):
            tls_config = {"tls": True}

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, **tls_config)

        try:
            client.admin.command('ping')
            logger.info("MongoDB connection test successful.")
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            raise

        logger.info(f"Identifying duplicate titles in US Code collection: {DB_NAME}.{COLL_NAME}")

        # Identify duplicates
        title_details = identify_duplicate_titles(client, limit=args.limit)

        if not title_details:
            logger.info("No duplicate titles found")
            return 0

        # Print detailed information
        print_duplicate_details(title_details, max_titles=args.max_display)

        # Analyze patterns
        analyze_duplicate_patterns(title_details)

        return 0

    except Exception as e:
        logger.error(f"Error identifying duplicate titles: {e}", exc_info=True)
        return 1
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    sys.exit(main())





































