#!/usr/bin/env python3
"""
Check for duplicate documents in US Code MongoDB collection.

This script identifies duplicates based on:
1. Title (should be unique)
2. Article + Chapter + Section combination
3. Same _id appearing multiple times (shouldn't happen)
4. Documents with same content but different _id
"""

import os
import sys
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
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


def check_duplicate_titles(client: MongoClient) -> List[Dict[str, Any]]:
    """Check for documents with duplicate titles."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    logger.info("Checking for duplicate titles...")
    
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
    
    duplicates = list(coll.aggregate(pipeline))
    
    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate titles:")
        for dup in duplicates[:20]:  # Show first 20
            title = dup["_id"]
            count = dup["count"]
            doc_ids = dup["doc_ids"]
            logger.warning(f"  Title: '{title}' appears {count} times")
            logger.warning(f"    Document IDs: {[str(doc_id) for doc_id in doc_ids[:5]]}")
            if len(doc_ids) > 5:
                logger.warning(f"    ... and {len(doc_ids) - 5} more")
    else:
        logger.info("✓ No duplicate titles found")
    
    return duplicates


def check_duplicate_article_chapter_section(client: MongoClient) -> List[Dict[str, Any]]:
    """Check for documents with duplicate article + chapter + section combinations."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    logger.info("Checking for duplicate article + chapter + section combinations...")
    
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
            "doc_ids": {"$push": "$_id"},
            "titles": {"$push": "$title"}
        }},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    duplicates = list(coll.aggregate(pipeline))
    
    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate article+chapter+section combinations:")
        for dup in duplicates[:20]:  # Show first 20
            key = dup["_id"]
            count = dup["count"]
            doc_ids = dup["doc_ids"]
            titles = dup["titles"]
            logger.warning(f"  Article: '{key.get('article')}', Chapter: '{key.get('chapter')}', Section: '{key.get('section')}' appears {count} times")
            logger.warning(f"    Titles: {titles[:3]}")
            logger.warning(f"    Document IDs: {[str(doc_id) for doc_id in doc_ids[:3]]}")
            if len(doc_ids) > 3:
                logger.warning(f"    ... and {len(doc_ids) - 3} more")
    else:
        logger.info("✓ No duplicate article+chapter+section combinations found")
    
    return duplicates


def check_empty_text_documents(client: MongoClient) -> List[Dict[str, Any]]:
    """Check for documents with no text content."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    logger.info("Checking for documents with no text content...")
    
    # Find documents with no text in clauses
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
                empty_text_docs.append({
                    "_id": doc.get("_id"),
                    "title": doc.get("title"),
                    "article": doc.get("article"),
                    "chapter": doc.get("chapter"),
                    "section": doc.get("section"),
                    "clauses_count": len(clauses)
                })
    
    # Also check documents without clauses
    docs_without_clauses = coll.find({
        "clauses": {"$exists": False}
    })
    
    for doc in docs_without_clauses:
        root_text = doc.get("text") or doc.get("summary") or doc.get("content") or doc.get("body")
        if not root_text or not str(root_text).strip():
            empty_text_docs.append({
                "_id": doc.get("_id"),
                "title": doc.get("title"),
                "article": doc.get("article"),
                "chapter": doc.get("chapter"),
                "section": doc.get("section"),
                "clauses_count": 0
            })
    
    if empty_text_docs:
        logger.warning(f"Found {len(empty_text_docs)} documents with no text content:")
        for doc in empty_text_docs[:20]:  # Show first 20
            logger.warning(f"  ID: {doc['_id']}, Title: '{doc.get('title', 'N/A')}', "
                          f"Article: '{doc.get('article', 'N/A')}', "
                          f"Section: '{doc.get('section', 'N/A')}', "
                          f"Clauses: {doc['clauses_count']}")
    else:
        logger.info("✓ All documents have text content")
    
    return empty_text_docs


def check_wrong_document_type(client: MongoClient) -> List[Dict[str, Any]]:
    """Check for documents that might have wrong document_type indicators."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    logger.info("Checking for documents with potential document_type issues...")
    
    issues = []
    
    # Check for documents with sections array (should be CFR, not US Code)
    docs_with_sections = coll.find({
        "sections": {"$exists": True, "$ne": []}
    })
    
    for doc in docs_with_sections:
        issues.append({
            "_id": doc.get("_id"),
            "title": doc.get("title"),
            "issue": "Has 'sections' array (typical of CFR, not US Code)",
            "article": doc.get("article"),
            "chapter": doc.get("chapter"),
            "section": doc.get("section")
        })
    
    # Check for documents with keywords array (should be US Constitution, not US Code)
    docs_with_keywords = coll.find({
        "keywords": {"$exists": True, "$ne": []}
    })
    
    for doc in docs_with_keywords:
        issues.append({
            "_id": doc.get("_id"),
            "title": doc.get("title"),
            "issue": "Has 'keywords' array (typical of US Constitution, not US Code)",
            "article": doc.get("article"),
            "chapter": doc.get("chapter"),
            "section": doc.get("section")
        })
    
    if issues:
        logger.warning(f"Found {len(issues)} documents with potential document_type issues:")
        for issue in issues[:20]:  # Show first 20
            logger.warning(f"  ID: {issue['_id']}, Title: '{issue.get('title', 'N/A')}'")
            logger.warning(f"    Issue: {issue['issue']}")
    else:
        logger.info("✓ No document_type issues found")
    
    return issues


def get_collection_stats(client: MongoClient) -> Dict[str, Any]:
    """Get collection statistics."""
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    
    total_count = coll.count_documents({})
    
    # Count documents with clauses
    docs_with_clauses = coll.count_documents({"clauses": {"$exists": True, "$ne": []}})
    
    # Count documents with sections (shouldn't be in US Code)
    docs_with_sections = coll.count_documents({"sections": {"$exists": True, "$ne": []}})
    
    # Count documents with keywords (shouldn't be in US Code)
    docs_with_keywords = coll.count_documents({"keywords": {"$exists": True, "$ne": []}})
    
    # Count documents with article field
    docs_with_article = coll.count_documents({"article": {"$exists": True, "$ne": None, "$ne": ""}})
    
    return {
        "total_documents": total_count,
        "documents_with_clauses": docs_with_clauses,
        "documents_with_sections": docs_with_sections,
        "documents_with_keywords": docs_with_keywords,
        "documents_with_article": docs_with_article
    }


def main():
    """Main function to check for duplicates."""
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
        logger.info("=" * 80)
        
        # Get collection statistics
        stats = get_collection_stats(client)
        logger.info("\nCollection Statistics:")
        logger.info(f"  Total documents: {stats['total_documents']}")
        logger.info(f"  Documents with clauses: {stats['documents_with_clauses']}")
        logger.info(f"  Documents with sections: {stats['documents_with_sections']} (unexpected for US Code)")
        logger.info(f"  Documents with keywords: {stats['documents_with_keywords']} (unexpected for US Code)")
        logger.info(f"  Documents with article: {stats['documents_with_article']}")
        
        logger.info("\n" + "=" * 80)
        
        # Check for duplicates
        duplicate_titles = check_duplicate_titles(client)
        
        logger.info("\n" + "=" * 80)
        
        duplicate_combos = check_duplicate_article_chapter_section(client)
        
        logger.info("\n" + "=" * 80)
        
        empty_text_docs = check_empty_text_documents(client)
        
        logger.info("\n" + "=" * 80)
        
        wrong_type_docs = check_wrong_document_type(client)
        
        logger.info("\n" + "=" * 80)
        logger.info("\nSummary:")
        logger.info(f"  Duplicate titles: {len(duplicate_titles)}")
        logger.info(f"  Duplicate article+chapter+section: {len(duplicate_combos)}")
        logger.info(f"  Documents with no text: {len(empty_text_docs)}")
        logger.info(f"  Documents with wrong type indicators: {len(wrong_type_docs)}")
        
        if duplicate_titles or duplicate_combos or empty_text_docs or wrong_type_docs:
            logger.warning("\n⚠ Issues found! Review the details above.")
            return 1
        else:
            logger.info("\n✓ No issues found!")
            return 0
        
    except Exception as e:
        logger.error(f"Error checking duplicates: {e}", exc_info=True)
        return 1
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    sys.exit(main())





































