#!/usr/bin/env python3
"""
Update duplicate titles in US Code MongoDB collection by prepending article + chapter + section + clause number.

For documents with duplicate titles (like "In general", "Definitions", etc.), this script
prepends the full citation path: "Title# Chapter# Section#.#" to make titles unique.

Example:
  Before: "In general"
  After:  "Title 42 Chapter 6A Section 300f.1 In general"
"""

import os
import sys
import logging
import re
from typing import Dict, List, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from collections import defaultdict

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


def format_citation_prefix(article: str, chapter: Optional[str], section: Optional[str], clause_num: Optional[str] = None) -> str:
    """Format citation prefix as 'Title# Chapter# Section#.#'"""
    parts = []
    
    # Article (e.g., "Title 42")
    if article:
        parts.append(article)
    
    # Chapter (e.g., "Chapter 6A")
    if chapter:
        if not chapter.startswith("Chapter "):
            parts.append(f"Chapter {chapter}")
        else:
            parts.append(chapter)
    
    # Section (e.g., "Section 300f")
    section_str = ""
    if section:
        # Remove "Section " prefix if present, we'll add it back
        section_clean = section.replace("Section ", "").strip()
        section_str = f"Section {section_clean}"
        parts.append(section_str)
    
    # Clause number (e.g., ".1", ".2") - append directly to section without space
    if clause_num:
        # Append clause number directly to the last part (section) without space
        if parts:
            parts[-1] = parts[-1] + f".{clause_num}"
        else:
            parts.append(f".{clause_num}")
    
    return " ".join(parts)


def find_duplicate_titles(client: MongoClient, specific_titles: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Find documents with duplicate titles, or specific titles if provided."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    # If specific titles are provided, find all documents with those titles
    if specific_titles:
        logger.info(f"Finding documents with specific titles: {specific_titles}")
        title_to_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for title in specific_titles:
            # First try exact match (case-sensitive)
            docs = list(coll.find({"title": title}, {
                "_id": 1,
                "title": 1,
                "article": 1,
                "chapter": 1,
                "section": 1,
                "clauses": 1
            }))
            
            # If no exact match, try case-insensitive exact match
            if not docs:
                docs = list(coll.find({"title": {"$regex": f"^{re.escape(title)}$", "$options": "i"}}, {
                    "_id": 1,
                    "title": 1,
                    "article": 1,
                    "chapter": 1,
                    "section": 1,
                    "clauses": 1
                }))
            
            # If still no match, try titles that start with the word (for "Definitions" or "In general")
            # This handles cases like "Definitions." or "Definitions and..." or "In general;"
            if not docs:
                # Match titles that start with the word, optionally followed by punctuation or whitespace
                pattern = f"^{re.escape(title)}([.;,]|$|\\s)"
                docs = list(coll.find({"title": {"$regex": pattern, "$options": "i"}}, {
                    "_id": 1,
                    "title": 1,
                    "article": 1,
                    "chapter": 1,
                    "section": 1,
                    "clauses": 1
                }))
            
            if docs:
                title_to_docs[title] = docs
                logger.info(f"Found {len(docs)} documents with title matching '{title}'")
        
        return title_to_docs
    
    # Otherwise, find all duplicate titles
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
    
    duplicate_groups = list(coll.aggregate(pipeline))
    
    # Build a map of title -> list of document IDs
    title_to_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for group in duplicate_groups:
        title = group["_id"]
        doc_ids = group["doc_ids"]
        
        # Fetch full documents for these IDs
        docs = list(coll.find({"_id": {"$in": doc_ids}}, {
            "_id": 1,
            "title": 1,
            "article": 1,
            "chapter": 1,
            "section": 1,
            "clauses": 1
        }))
        
        title_to_docs[title] = docs
    
    logger.info(f"Found {len(duplicate_groups)} duplicate title groups")
    return title_to_docs


def update_titles_with_citation(client: MongoClient, title_to_docs: Dict[str, List[Dict[str, Any]]], dry_run: bool = True, skip_if_has_prefix: bool = True) -> int:
    """Update titles by prepending citation prefix."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    updated_count = 0
    skipped_count = 0
    
    for title, docs in title_to_docs.items():
        logger.info(f"Processing {len(docs)} documents with title '{title}'...")
        
        for doc in docs:
            doc_id = doc.get("_id")
            current_title = doc.get("title", "")
            article = doc.get("article", "")
            chapter = doc.get("chapter")
            section = doc.get("section")
            clauses = doc.get("clauses", [])
            
            # Skip if title already has a citation prefix (starts with "Title" or has citation-like structure)
            if skip_if_has_prefix:
                if current_title.startswith("Title ") or re.match(r'^Title \d+', current_title):
                    skipped_count += 1
                    if dry_run:
                        logger.debug(f"  Skipping (already has prefix): {doc_id} - '{current_title}'")
                    continue
            
            # Determine clause number
            # If there's only one clause, use "1", otherwise use the clause number
            clause_num = None
            if clauses:
                if len(clauses) == 1:
                    clause_num = "1"
                else:
                    # For documents with multiple clauses, we need to handle each clause separately
                    # But since we're updating the document title, we'll use the first clause's number
                    first_clause = clauses[0] if isinstance(clauses[0], dict) else None
                    if first_clause:
                        clause_num = first_clause.get("number", "1")
            
            # Format citation prefix
            citation_prefix = format_citation_prefix(article, chapter, section, clause_num)
            
            # New title: citation prefix + original title (use current_title, not the search term)
            new_title = f"{citation_prefix} {current_title}"
            
            if dry_run:
                logger.info(f"  Would update: {doc_id}")
                logger.info(f"    Old: '{current_title}'")
                logger.info(f"    New: '{new_title}'")
            else:
                # Update document
                result = coll.update_one(
                    {"_id": doc_id},
                    {"$set": {"title": new_title}}
                )
                if result.modified_count > 0:
                    updated_count += 1
                    logger.debug(f"  Updated: {doc_id} -> '{new_title}'")
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} documents that already have citation prefixes")
    
    return updated_count


def main():
    """Main function to update duplicate titles."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update duplicate titles in US Code collection")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't actually update, just show what would be updated"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm update (required for actual updates)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of duplicate title groups to process (for testing)"
    )
    parser.add_argument(
        "--specific-titles",
        nargs="+",
        default=None,
        help="Specific titles to update (e.g., 'Definitions' 'In general'). If provided, only these titles will be processed."
    )
    parser.add_argument(
        "--skip-prefixed",
        action="store_true",
        default=True,
        help="Skip documents that already have citation prefixes (default: True)"
    )
    
    args = parser.parse_args()
    
    dry_run = args.dry_run or not args.confirm
    
    if not dry_run:
        # Only prompt for confirmation if running interactively (not in automated environment)
        import sys
        if sys.stdin.isatty():
            response = input(f"Are you sure you want to update duplicate titles? This will modify the database! (yes/no): ")
            if response.lower() != "yes":
                logger.info("Update cancelled")
                return 0
        else:
            logger.info("Running in non-interactive mode. Proceeding with update...")
    
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
        
        logger.info(f"Updating duplicate titles in US Code collection: {DB_NAME}.{COLL_NAME}")
        
        # Find duplicate titles or specific titles
        if args.specific_titles:
            title_to_docs = find_duplicate_titles(client, specific_titles=args.specific_titles)
        else:
            title_to_docs = find_duplicate_titles(client)
        
        if not title_to_docs:
            logger.info("No titles found to update")
            return 0
        
        # Limit if specified
        if args.limit:
            title_to_docs = dict(list(title_to_docs.items())[:args.limit])
            logger.info(f"Processing limited to {len(title_to_docs)} title groups")
        
        # Update titles
        updated_count = update_titles_with_citation(client, title_to_docs, dry_run=dry_run, skip_if_has_prefix=args.skip_prefixed)
        
        if dry_run:
            logger.info(f"\nThis was a dry run. {updated_count} documents would be updated.")
            logger.info("To actually update, run with --confirm flag")
        else:
            logger.info(f"\nSuccessfully updated {updated_count} documents")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error updating duplicate titles: {e}", exc_info=True)
        return 1
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    sys.exit(main())

