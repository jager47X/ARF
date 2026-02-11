# normalize_existing_docs.py
"""
Standalone script to normalize chapter and section fields for all existing US Code documents in MongoDB.
This applies normalization: "Chapter CHAPTER 1—" -> "Chapter 1" and "Section § 4." -> "Section 4"
"""
import os
import sys
import logging
import argparse
import re
from pathlib import Path
from pymongo import MongoClient
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create 'backend' -> 'kyr-backend' mapping for imports
import importlib.util
import types

# Set up backend module structure
if 'backend' not in sys.modules:
    backend_mod = types.ModuleType('backend')
    sys.modules['backend'] = backend_mod
    
    services_init = backend_dir / 'services' / '__init__.py'
    if services_init.exists():
        spec = importlib.util.spec_from_file_location('backend.services', services_init)
        if spec and spec.loader:
            services_mod = importlib.util.module_from_spec(spec)
            sys.modules['backend.services'] = services_mod
            spec.loader.exec_module(services_mod)
            setattr(backend_mod, 'services', services_mod)
            
            if 'services' not in sys.modules:
                sys.modules['services'] = services_mod
            
            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
            if rag_init.exists():
                spec = importlib.util.spec_from_file_location('backend.services.rag', rag_init)
                if spec and spec.loader:
                    rag_mod = importlib.util.module_from_spec(spec)
                    sys.modules['backend.services.rag'] = rag_mod
                    spec.loader.exec_module(rag_mod)
                    setattr(services_mod, 'rag', rag_mod)
                    sys.modules['services.rag'] = rag_mod
                    
                    config_file = backend_dir / 'services' / 'rag' / 'config.py'
                    if config_file.exists():
                        spec = importlib.util.spec_from_file_location('backend.services.rag.config', config_file)
                        if spec and spec.loader:
                            config_mod = importlib.util.module_from_spec(spec)
                            sys.modules['backend.services.rag.config'] = config_mod
                            spec.loader.exec_module(config_mod)
                            setattr(rag_mod, 'config', config_mod)
                            sys.modules['services.rag.config'] = config_mod

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Normalize chapter and section fields for US Code documents")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (.env.production)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev environment (.env.dev)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local environment (.env.local)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually updating"
    )
    return parser.parse_args()

# Load environment based on args
args = parse_args()
env_override = None
if args:
    if args.production:
        env_override = "production"
    elif args.dev:
        env_override = "dev"
    elif args.local:
        env_override = "local"

# Import config module and load environment
import backend.services.rag.config as config_module

if env_override:
    config_module.load_environment(env_override)
    import os
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found in {config_module._env_file_used}")
else:
    MONGO_URI = config_module.MONGO_URI

COLLECTION = config_module.COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("normalize_us_code")

US_CODE_CONF = COLLECTION.get("US_CODE_SET")
if not US_CODE_CONF:
    raise ValueError("US_CODE_SET not found in COLLECTION config")

DB_NAME: str = US_CODE_CONF["db_name"]
COLL_NAME: str = US_CODE_CONF["main_collection_name"]

def normalize_chapter(chapter_str: str) -> str:
    """Normalize chapter: 'Chapter CHAPTER 1—' -> 'Chapter 1'"""
    if not chapter_str:
        return ""
    
    chapter_str = str(chapter_str).strip()
    
    # Remove all "Chapter", "CHAPTER" prefixes (case insensitive, may appear multiple times)
    chapter_clean = chapter_str
    while True:
        new_clean = re.sub(r'^(chapter|CHAPTER)\s*', '', chapter_clean, flags=re.IGNORECASE).strip()
        if new_clean == chapter_clean:
            break
        chapter_clean = new_clean
    
    # Remove everything after and including em-dash, en-dash, or regular dash
    chapter_clean = re.sub(r'[—–\-]+.*$', '', chapter_clean).strip()
    
    # Extract just digits and optional letters (for cases like "1A")
    match = re.search(r'([0-9]+[A-Za-z]?)', chapter_clean)
    if match:
        chapter_num = match.group(1)
        return f"Chapter {chapter_num}"
    else:
        # If no match, try to extract any number
        num_match = re.search(r'(\d+)', chapter_clean)
        if num_match:
            return f"Chapter {num_match.group(1)}"
        else:
            return chapter_str if chapter_str else ""

def normalize_section(section_str: str) -> str:
    """Normalize section: 'Section § 1.' -> 'Section 1'"""
    if not section_str:
        return ""
    
    section_str = str(section_str).strip()
    
    # Remove "Section" prefix (case insensitive, may appear multiple times)
    section_clean = section_str
    while True:
        new_clean = re.sub(r'^(section|Section|SECTION)\s+', '', section_clean, flags=re.IGNORECASE).strip()
        if new_clean == section_clean:
            break
        section_clean = new_clean
    
    # Remove § symbols
    section_clean = re.sub(r'§+\s*', '', section_clean)
    # Remove trailing periods
    section_clean = section_clean.rstrip('.')
    section_clean = section_clean.strip()
    
    # Extract just the number
    match = re.search(r'([0-9]+[A-Za-z]?)', section_clean)
    if match:
        section_num = match.group(1)
        return f"Section {section_num}"
    else:
        # If it's already just a number, add "Section" prefix
        if section_clean and section_clean.isdigit():
            return f"Section {section_clean}"
        return section_str if section_str else ""

def normalize_all_documents():
    """Normalize chapter and section fields for all documents in the collection"""
    client = None
    try:
        # Configure TLS for MongoDB Atlas connections
        tls_config = {}
        if MONGO_URI and "mongodb+srv://" in MONGO_URI:
            tls_config = {"tls": True}
        elif MONGO_URI and ("mongodb.net" in MONGO_URI or "mongodb.com" in MONGO_URI):
            tls_config = {"tls": True}
        
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, **tls_config)
        
        # Test connection
        try:
            client.admin.command('ping')
            logger.info("MongoDB connection test successful.")
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            raise
        
        db = client[DB_NAME]
        coll = db.get_collection(COLL_NAME)
        logger.info(f"Connected to MongoDB collection: {DB_NAME}.{COLL_NAME}")
        
        # Count total documents
        total_docs = coll.count_documents({})
        logger.info(f"Found {total_docs} documents to check")
        
        if total_docs == 0:
            logger.info("No documents found. Exiting.")
            return
        
        # Process documents in batches with parallel processing
        logger.info("Starting normalization with parallel batch processing...")
        if args and args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
        
        def normalize_document(doc):
            """Normalize a single document and return update info"""
            try:
                update_fields = {}
                needs_update = False
                doc_id = doc.get("_id")
                doc_title = doc.get("title", "unknown")
                
                # Normalize chapter if present
                if "chapter" in doc and doc["chapter"]:
                    original_chapter = doc["chapter"]
                    normalized_chapter = normalize_chapter(original_chapter)
                    if normalized_chapter != original_chapter:
                        update_fields["chapter"] = normalized_chapter
                        needs_update = True
                
                # Normalize section if present
                if "section" in doc and doc["section"]:
                    original_section = doc["section"]
                    normalized_section = normalize_section(original_section)
                    if normalized_section != original_section:
                        update_fields["section"] = normalized_section
                        needs_update = True
                
                if needs_update:
                    return {
                        "doc_id": doc_id,
                        "update_fields": update_fields,
                        "doc_title": doc_title,
                        "status": "needs_update"
                    }
                else:
                    return {
                        "doc_id": doc_id,
                        "status": "skipped"
                    }
            except Exception as e:
                return {
                    "doc_id": doc.get("_id"),
                    "status": "error",
                    "error": str(e),
                    "doc_title": doc.get("title", "unknown")
                }
        
        # Process documents in batches to avoid memory issues
        normalized_count = 0
        skipped_count = 0
        error_count = 0
        batch_size = 1000  # Process 1000 documents at a time
        max_workers = 20  # Parallel workers per batch
        
        # Process in batches using cursor
        cursor = coll.find({}).batch_size(batch_size)
        batch = []
        processed = 0
        
        for doc in cursor:
            batch.append(doc)
            
            # Process batch when it reaches batch_size
            if len(batch) >= batch_size:
                logger.info(f"Processing batch of {len(batch)} documents (total processed: {processed})...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_doc = {
                        executor.submit(normalize_document, doc): i 
                        for i, doc in enumerate(batch)
                    }
                    
                    for future in as_completed(future_to_doc):
                        try:
                            result = future.result()
                            
                            if result["status"] == "needs_update":
                                if args and args.dry_run:
                                    normalized_count += 1
                                else:
                                    # Perform the update
                                    update_result = coll.update_one(
                                        {"_id": result["doc_id"]},
                                        {"$set": result["update_fields"]}
                                    )
                                    if update_result.modified_count > 0:
                                        normalized_count += 1
                            elif result["status"] == "skipped":
                                skipped_count += 1
                            elif result["status"] == "error":
                                error_count += 1
                                logger.warning(f"Failed to normalize document '{result.get('doc_title', 'unknown')}': {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            error_count += 1
                            logger.warning(f"Error processing document: {e}")
                
                processed += len(batch)
                if normalized_count % 100 == 0 and normalized_count > 0:
                    logger.info(f"Normalized {normalized_count} documents so far...")
                
                batch = []  # Clear batch
        
        # Process remaining documents in batch
        if batch:
            logger.info(f"Processing final batch of {len(batch)} documents...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {
                    executor.submit(normalize_document, doc): i 
                    for i, doc in enumerate(batch)
                }
                
                for future in as_completed(future_to_doc):
                    try:
                        result = future.result()
                        
                        if result["status"] == "needs_update":
                            if args and args.dry_run:
                                normalized_count += 1
                            else:
                                update_result = coll.update_one(
                                    {"_id": result["doc_id"]},
                                    {"$set": result["update_fields"]}
                                )
                                if update_result.modified_count > 0:
                                    normalized_count += 1
                        elif result["status"] == "skipped":
                            skipped_count += 1
                        elif result["status"] == "error":
                            error_count += 1
                            logger.warning(f"Failed to normalize document '{result.get('doc_title', 'unknown')}': {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Error processing document: {e}")
            
            processed += len(batch)
        
        # Summary
        logger.info("=" * 60)
        logger.info("Normalization Summary:")
        logger.info(f"  Total documents: {total_docs}")
        logger.info(f"  Normalized: {normalized_count}")
        logger.info(f"  Already normalized (skipped): {skipped_count}")
        logger.info(f"  Errors: {error_count}")
        if args and args.dry_run:
            logger.info("  (DRY RUN - No actual changes made)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Normalization error: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    normalize_all_documents()

