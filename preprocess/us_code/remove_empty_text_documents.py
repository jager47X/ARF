#!/usr/bin/env python3
"""
Remove US Code documents with no text content from MongoDB.

This script identifies and removes documents that have no text in their clauses
or root-level text fields.
"""

import logging
import os
import sys
from typing import Any, Dict, List

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

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


def find_empty_text_documents(client: MongoClient) -> List[Dict[str, Any]]:
    """Find documents with no text content."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]

    logger.info("Finding documents with no text content...")

    empty_text_docs = []

    # Check documents with clauses
    docs_with_clauses = coll.find({"clauses": {"$exists": True, "$ne": []}})

    for doc in docs_with_clauses:
        clauses = doc.get("clauses", [])
        has_text = False

        # Check if any clause has text
        for clause in clauses:
            if isinstance(clause, dict):
                clause_text = clause.get("text") or clause.get("content") or clause.get("body")
                if clause_text and str(clause_text).strip():
                    has_text = True
                    break

        if not has_text:
            # Check root-level text as fallback
            root_text = doc.get("text") or doc.get("summary") or doc.get("content") or doc.get("body")
            if not root_text or not str(root_text).strip():
                empty_text_docs.append(doc.get("_id"))

    # Also check documents without clauses
    docs_without_clauses = coll.find({
        "clauses": {"$exists": False}
    })

    for doc in docs_without_clauses:
        root_text = doc.get("text") or doc.get("summary") or doc.get("content") or doc.get("body")
        if not root_text or not str(root_text).strip():
            empty_text_docs.append(doc.get("_id"))

    logger.info(f"Found {len(empty_text_docs)} documents with no text content")
    return empty_text_docs


def remove_empty_documents(client: MongoClient, doc_ids: List[Any], dry_run: bool = True) -> int:
    """Remove documents with no text content."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]

    if not doc_ids:
        logger.info("No documents to remove")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would remove {len(doc_ids)} documents")
        # Show first 10 examples
        for i, doc_id in enumerate(doc_ids[:10]):
            doc = coll.find_one({"_id": doc_id}, {"title": 1, "article": 1, "chapter": 1, "section": 1})
            if doc:
                logger.info(f"  Would remove: {doc_id} - '{doc.get('title', 'N/A')}' "
                          f"(Article: {doc.get('article', 'N/A')}, Section: {doc.get('section', 'N/A')})")
        if len(doc_ids) > 10:
            logger.info(f"  ... and {len(doc_ids) - 10} more")
        return 0

    logger.info(f"Removing {len(doc_ids)} documents with no text content...")
    result = coll.delete_many({"_id": {"$in": doc_ids}})
    logger.info(f"Removed {result.deleted_count} documents")
    return result.deleted_count


def main():
    """Main function to remove empty text documents."""
    import argparse

    parser = argparse.ArgumentParser(description="Remove US Code documents with no text content")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't actually delete, just show what would be deleted"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion (required for actual deletion)"
    )

    args = parser.parse_args()

    dry_run = args.dry_run or not args.confirm

    if not dry_run:
        response = input("Are you sure you want to delete documents with no text? This cannot be undone! (yes/no): ")
        if response.lower() != "yes":
            logger.info("Deletion cancelled")
            return 0

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

        logger.info(f"Checking US Code collection: {DB_NAME}.{COLL_NAME}")

        # Find empty text documents
        empty_doc_ids = find_empty_text_documents(client)

        if not empty_doc_ids:
            logger.info("No documents with empty text found")
            return 0

        # Remove them
        removed_count = remove_empty_documents(client, empty_doc_ids, dry_run=dry_run)

        if dry_run:
            logger.info("\nThis was a dry run. To actually delete, run with --confirm flag")
        else:
            logger.info(f"\nSuccessfully removed {removed_count} documents")

        return 0

    except Exception as e:
        logger.error(f"Error removing empty documents: {e}", exc_info=True)
        return 1
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    sys.exit(main())





































