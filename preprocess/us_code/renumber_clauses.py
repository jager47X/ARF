#!/usr/bin/env python3
"""
Renumber clauses for duplicate article+chapter+section combinations in US Code MongoDB collection.

For documents that share the same article+chapter+section, this script assigns unique
sequential clause numbers (.1, .2, .3, etc.) and updates both the title and the clause
number in the clauses array.

Example:
  Before: 27 documents all with "Title 42 Chapter 6A Section 300f.1 In general"
  After:  "Title 42 Chapter 6A Section 300f.1 In general"
          "Title 42 Chapter 6A Section 300f.2 In general"
          "Title 42 Chapter 6A Section 300f.3 In general"
          etc.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
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


def format_citation_prefix(article: str, chapter: str, section: str, clause_num: str) -> str:
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


def extract_original_title(title: str, doc: Dict[str, Any]) -> str:
    """Extract the original title from a title that may already have citation prefix."""
    if not title:
        # Try to get title from first clause
        clauses = doc.get("clauses", [])
        if clauses and isinstance(clauses, list) and len(clauses) > 0:
            first_clause = clauses[0]
            if isinstance(first_clause, dict):
                return first_clause.get("title", "")
        return ""
    
    # If title starts with "Title", it likely has a citation prefix
    # Try to extract the part after the citation
    parts = title.split()
    if len(parts) > 0 and parts[0] == "Title":
        # Find where the actual title starts (after Section X.Y)
        # Look for pattern: Title X Chapter Y Section Z.N <actual title>
        for i, part in enumerate(parts):
            if part == "Section" and i + 1 < len(parts):
                # Check if next part has clause number (e.g., "300f.1")
                next_part = parts[i + 1]
                if "." in next_part:
                    # Title starts after the clause number
                    if i + 2 < len(parts):
                        return " ".join(parts[i + 2:])
                    else:
                        # No title after clause number, try to get from clause
                        clauses = doc.get("clauses", [])
                        if clauses and isinstance(clauses, list) and len(clauses) > 0:
                            first_clause = clauses[0]
                            if isinstance(first_clause, dict):
                                return first_clause.get("title", "")
                        return ""
    
    # No citation prefix found, return as-is
    return title


def find_duplicate_sections(client: MongoClient) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Find documents grouped by article+chapter+section that have duplicates."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    logger.info("Finding documents with duplicate article+chapter+section combinations...")
    
    # Use aggregation to find duplicates
    pipeline = [
        {"$match": {
            "article": {"$exists": True, "$ne": None, "$ne": ""},
            "section": {"$exists": True, "$ne": None, "$ne": ""}
        }},
        {"$group": {
            "_id": {
                "article": "$article",
                "chapter": "$chapter",
                "section": "$section"
            },
            "count": {"$sum": 1},
            "doc_ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    duplicate_groups = list(coll.aggregate(pipeline))
    
    logger.info(f"Found {len(duplicate_groups)} duplicate article+chapter+section combinations")
    
    # Build detailed information for each duplicate group
    section_to_docs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    
    for group in duplicate_groups:
        key = group["_id"]
        article = key.get("article", "")
        chapter = key.get("chapter", "")
        section = key.get("section", "")
        doc_ids = group["doc_ids"]
        count = group["count"]
        
        # Fetch full documents for these IDs
        docs = list(coll.find({"_id": {"$in": doc_ids}}, {
            "_id": 1,
            "title": 1,
            "article": 1,
            "chapter": 1,
            "section": 1,
            "clauses": 1
        }))
        
        section_key = (article, chapter or "", section)
        section_to_docs[section_key] = docs
    
    return section_to_docs


def renumber_clauses_for_section(
    client: MongoClient,
    section_key: Tuple[str, str, str],
    docs: List[Dict[str, Any]],
    dry_run: bool = True
) -> int:
    """Renumber clauses for documents in the same section."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    article, chapter, section = section_key
    updated_count = 0
    
    # Sort documents by _id to ensure consistent ordering
    docs_sorted = sorted(docs, key=lambda d: str(d.get("_id")))
    
    logger.info(f"Processing {len(docs_sorted)} documents for {article} | {chapter} | {section}")
    
    for idx, doc in enumerate(docs_sorted, 1):
        doc_id = doc.get("_id")
        current_title = doc.get("title", "")
        
        # Extract original title (without citation prefix if present)
        original_title = extract_original_title(current_title, doc)
        if not original_title:
            original_title = current_title
        
        # Format new citation with sequential clause number
        new_clause_num = str(idx)
        new_citation = format_citation_prefix(article, chapter, section, new_clause_num)
        new_title = f"{new_citation} {original_title}".strip()
        
        if dry_run:
            logger.info(f"  Would update: {doc_id}")
            logger.info(f"    Old title: '{current_title}'")
            logger.info(f"    New title: '{new_title}'")
            logger.info(f"    Clause number: {new_clause_num}")
        else:
            # Update document
            update_fields = {"title": new_title}
            
            # Also update clause number in clauses array if it exists
            clauses = doc.get("clauses", [])
            if clauses and isinstance(clauses, list) and len(clauses) > 0:
                # Update first clause number
                updated_clauses = []
                for clause_idx, clause in enumerate(clauses):
                    if isinstance(clause, dict):
                        updated_clause = clause.copy()
                        # Update clause number - use sequential numbering starting from 1
                        updated_clause["number"] = str(clause_idx + 1)
                        updated_clauses.append(updated_clause)
                    else:
                        updated_clauses.append(clause)
                update_fields["clauses"] = updated_clauses
            
            try:
                # Log what we're trying to update
                logger.info(f"  Updating {doc_id}:")
                logger.info(f"    Old: '{current_title}'")
                logger.info(f"    New: '{new_title}'")
                
                result = coll.update_one(
                    {"_id": doc_id},
                    {"$set": update_fields}
                )
                logger.info(f"    Result: matched={result.matched_count}, modified={result.modified_count}")
                
                if result.modified_count > 0:
                    updated_count += 1
                    logger.info(f"  ✓ Updated: {doc_id}")
                elif result.matched_count > 0:
                    logger.info(f"  - No change: {doc_id} (title already matches)")
                else:
                    logger.warning(f"  ✗ Not found: {doc_id}")
            except Exception as e:
                logger.error(f"  ✗ Error updating {doc_id}: {e}")
    
    return updated_count


def main():
    """Main function to renumber clauses."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Renumber clauses for duplicate sections in US Code collection")
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
        help="Limit number of duplicate section groups to process (for testing)"
    )
    
    args = parser.parse_args()
    
    dry_run = args.dry_run or not args.confirm
    
    if not dry_run and not args.confirm:
        # Only ask for confirmation if --confirm was not explicitly passed
        try:
            response = input(f"Are you sure you want to renumber clauses? This will modify the database! (yes/no): ")
            if response.lower() != "yes":
                logger.info("Update cancelled")
                return 0
        except EOFError:
            # Non-interactive environment - require --confirm flag
            logger.error("Cannot prompt for confirmation in non-interactive environment. Use --confirm flag to proceed.")
            return 1
    
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
        
        logger.info(f"Renumbering clauses in US Code collection: {DB_NAME}.{COLL_NAME}")
        
        # Find duplicate sections
        section_to_docs = find_duplicate_sections(client)
        
        if not section_to_docs:
            logger.info("No duplicate sections found")
            return 0
        
        # Limit if specified
        if args.limit:
            section_to_docs = dict(list(section_to_docs.items())[:args.limit])
            logger.info(f"Processing limited to {len(section_to_docs)} duplicate section groups")
        
        total_updated = 0
        total_sections = len(section_to_docs)
        
        for idx, (section_key, docs) in enumerate(section_to_docs.items(), 1):
            article, chapter, section = section_key
            logger.info(f"\n[{idx}/{total_sections}] Processing: {article} | {chapter} | {section} ({len(docs)} documents)")
            
            updated = renumber_clauses_for_section(client, section_key, docs, dry_run=dry_run)
            total_updated += updated
        
        if dry_run:
            logger.info(f"\nThis was a dry run. {total_updated} documents would be updated.")
            logger.info("To actually update, run with --confirm flag")
        else:
            logger.info(f"\nSuccessfully updated {total_updated} documents across {total_sections} sections")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error renumbering clauses: {e}", exc_info=True)
        return 1
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    sys.exit(main())

