# update_uscis_policy_manual.py
"""
Update USCIS Policy Manual in MongoDB by comparing new JSON with current JSON
and updating only changed documents.
"""
import os
import sys
import json
import logging
import argparse
import datetime
from typing import Any, Dict, List, Tuple
from pathlib import Path
from pymongo import MongoClient, WriteConcern
from pymongo.errors import BulkWriteError

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import config module
import importlib.util
import types

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
            
            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
            if rag_init.exists():
                spec = importlib.util.spec_from_file_location('backend.services.rag', rag_init)
                if spec and spec.loader:
                    rag_mod = importlib.util.module_from_spec(spec)
                    sys.modules['backend.services.rag'] = rag_mod
                    spec.loader.exec_module(rag_mod)
                    setattr(services_mod, 'rag', rag_mod)
                    
                    config_file = backend_dir / 'services' / 'rag' / 'config.py'
                    if config_file.exists():
                        spec = importlib.util.spec_from_file_location('backend.services.rag.config', config_file)
                        if spec and spec.loader:
                            config_mod = importlib.util.module_from_spec(spec)
                            sys.modules['backend.services.rag.config'] = config_mod
                            spec.loader.exec_module(config_mod)
                            setattr(rag_mod, 'config', config_mod)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Update USCIS Policy Manual in MongoDB")
    parser.add_argument("--production", action="store_true", help="Use production environment")
    parser.add_argument("--dev", action="store_true", help="Use dev environment")
    parser.add_argument("--local", action="store_true", help="Use local environment")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading HTML (use existing)")
    parser.add_argument("--skip-convert", action="store_true", help="Skip HTML to JSON conversion (use existing JSON)")
    return parser.parse_args()

args = parse_args() if __name__ == "__main__" else None
env_override = None
if args:
    if args.production:
        env_override = "production"
    elif args.dev:
        env_override = "dev"
    elif args.local:
        env_override = "local"

# Import config
import backend.services.rag.config as config_module

if env_override:
    config_module.load_environment(env_override)
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found in {config_module._env_file_used}")
else:
    MONGO_URI = config_module.MONGO_URI

COLLECTION = config_module.COLLECTION
AUTOUPDATE_CONFIG = config_module.AUTOUPDATE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_uscis_policy")

BASE_DIR = Path(__file__).resolve().parents[2]
HTML_PATH = BASE_DIR / "Data" / "Knowledge" / "Policy Manual _ USCIS.html"
CURRENT_JSON_PATH = BASE_DIR / "Data" / "Knowledge" / "uscis_policy.json"
NEW_JSON_PATH = BASE_DIR / "Data" / "Knowledge" / "uscis_policy_new.json"

USCIS_POLICY_CONF = COLLECTION.get("USCIS_POLICY_SET")
if not USCIS_POLICY_CONF:
    raise ValueError("USCIS_POLICY_SET not found in COLLECTION config")
DB_NAME: str = USCIS_POLICY_CONF["db_name"]
COLL_NAME: str = USCIS_POLICY_CONF["main_collection_name"]

def check_autoupdate_enabled() -> bool:
    """Check if autoupdate is enabled."""
    # Check environment variable first
    env_enabled = os.getenv("USCIS_AUTOUPDATE_ENABLED", "").lower() == "true"
    if env_enabled:
        return True
    
    # Check AUTOUPDATE_CONFIG
    if AUTOUPDATE_CONFIG.get("enabled", False):
        return True
    
    # Check collection-specific config
    if USCIS_POLICY_CONF.get("autoupdate_enabled", False):
        return True
    
    return False

def deep_compare_clauses(clauses1: List[Dict], clauses2: List[Dict]) -> bool:
    """Deep compare two clause arrays."""
    if len(clauses1) != len(clauses2):
        return False
    
    for c1, c2 in zip(clauses1, clauses2):
        if c1.get("number") != c2.get("number"):
            return False
        if c1.get("title", "").strip() != c2.get("title", "").strip():
            return False
        if c1.get("text", "").strip() != c2.get("text", "").strip():
            return False
    
    return True

def compare_documents(current_doc: Dict[str, Any], new_doc: Dict[str, Any]) -> bool:
    """Compare two documents to see if they're different."""
    # Compare text
    if current_doc.get("text", "").strip() != new_doc.get("text", "").strip():
        return False
    
    # Compare references (sorted for comparison)
    refs1 = sorted(current_doc.get("references", []))
    refs2 = sorted(new_doc.get("references", []))
    if refs1 != refs2:
        return False
    
    # Compare clauses (deep comparison)
    if not deep_compare_clauses(current_doc.get("clauses", []), new_doc.get("clauses", [])):
        return False
    
    return True

def find_document_changes(current_docs: List[Dict], new_docs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Compare current and new documents to find changes.
    
    Returns:
        Tuple of (new_documents, updated_documents, deleted_documents)
    """
    # Create title -> document maps
    current_by_title = {d.get("title", ""): d for d in current_docs if d.get("title")}
    new_by_title = {d.get("title", ""): d for d in new_docs if d.get("title")}
    
    new_documents = []
    updated_documents = []
    deleted_titles = []
    
    # Find new documents
    for title, new_doc in new_by_title.items():
        if title not in current_by_title:
            new_documents.append(new_doc)
    
    # Find updated documents
    for title, new_doc in new_by_title.items():
        if title in current_by_title:
            current_doc = current_by_title[title]
            if not compare_documents(current_doc, new_doc):
                updated_documents.append(new_doc)
    
    # Find deleted documents
    for title in current_by_title:
        if title not in new_by_title:
            deleted_titles.append(title)
    
    return new_documents, updated_documents, deleted_titles

def update_mongodb(new_docs: List[Dict], updated_docs: List[Dict], deleted_titles: List[str]):
    """Update MongoDB with new, updated, and deleted documents."""
    client = None
    try:
        # Configure TLS for MongoDB Atlas connections
        tls_config = {}
        if MONGO_URI and "mongodb+srv://" in MONGO_URI:
            tls_config = {"tls": True}
        elif MONGO_URI and ("mongodb.net" in MONGO_URI or "mongodb.com" in MONGO_URI):
            tls_config = {"tls": True}
        
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, **tls_config)
        client.admin.command('ping')
        
        db = client[DB_NAME]
        coll = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=1))
        
        now = datetime.datetime.utcnow()
        
        # Insert new documents
        if new_docs:
            for doc in new_docs:
                doc["created_at"] = now
                doc["updated_at"] = now
            try:
                res = coll.insert_many(new_docs, ordered=False)
                logger.info(f"Inserted {len(res.inserted_ids)} new documents")
            except BulkWriteError as bwe:
                n = bwe.details.get("nInserted", 0)
                logger.warning(f"BulkWriteError; inserted {n} new docs")
        
        # Update modified documents
        if updated_docs:
            updated_count = 0
            for doc in updated_docs:
                try:
                    result = coll.update_one(
                        {"title": doc["title"]},
                        {"$set": {
                            "text": doc.get("text", ""),
                            "date": doc.get("date", ""),
                            "references": doc.get("references", []),
                            "clauses": doc.get("clauses", []),
                            "updated_at": now
                        }}
                    )
                    if result.modified_count > 0:
                        updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update document '{doc.get('title')}': {e}")
            logger.info(f"Updated {updated_count} existing documents")
        
        # Handle deleted documents (log only, don't delete)
        if deleted_titles:
            logger.warning(f"Found {len(deleted_titles)} deleted documents (not removing from MongoDB): {deleted_titles[:5]}...")
        
        # Save last update timestamp
        last_check_file_path = AUTOUPDATE_CONFIG.get("last_check_file", "Data/Knowledge/.last_uscis_check")
        last_check_file = BASE_DIR / last_check_file_path
        last_check_file.parent.mkdir(parents=True, exist_ok=True)
        with open(last_check_file, 'w') as f:
            f.write(now.isoformat())
        logger.info(f"Saved last update timestamp to {last_check_file}")
        
    except Exception as e:
        logger.error(f"Error updating MongoDB: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()

def main():
    """Main function to update USCIS Policy Manual."""
    # Check if autoupdate is enabled
    if not check_autoupdate_enabled():
        logger.info("Autoupdate is disabled. Set USCIS_AUTOUPDATE_ENABLED=true or enable in config to run updates.")
        return 0
    
    logger.info("Starting USCIS Policy Manual update...")
    
    # Step 1: Download HTML (if not skipped)
    if not args or not args.skip_download:
        logger.info("Step 1: Downloading HTML...")
        from download_uscis_policy_manual import download_policy_manual, USCIS_POLICY_URL
        url = USCIS_POLICY_CONF.get("autoupdate_url", USCIS_POLICY_URL)
        if not download_policy_manual(url, HTML_PATH):
            logger.error("Failed to download HTML")
            return 1
    else:
        logger.info("Skipping download (using existing HTML)")
    
    # Step 2: Convert HTML to JSON (if not skipped)
    if not args or not args.skip_convert:
        logger.info("Step 2: Converting HTML to JSON...")
        from convert_uscis_html_to_json import parse_policy_manual_html
        if not HTML_PATH.exists():
            logger.error(f"HTML file not found: {HTML_PATH}")
            return 1
        
        new_json_data = parse_policy_manual_html(HTML_PATH)
        with open(NEW_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Converted HTML to JSON: {NEW_JSON_PATH}")
    else:
        logger.info("Skipping conversion (using existing new JSON)")
        if not NEW_JSON_PATH.exists():
            logger.error(f"New JSON file not found: {NEW_JSON_PATH}")
            return 1
    
    # Step 3: Load current and new JSON
    logger.info("Step 3: Loading JSON files...")
    if not CURRENT_JSON_PATH.exists():
        logger.warning(f"Current JSON not found: {CURRENT_JSON_PATH}. Treating all as new documents.")
        current_docs = []
    else:
        with open(CURRENT_JSON_PATH, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        current_docs = current_data.get("data", {}).get("uscis_policy", {}).get("documents", [])
    
    with open(NEW_JSON_PATH, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    new_docs = new_data.get("data", {}).get("uscis_policy", {}).get("documents", [])
    
    logger.info(f"Current documents: {len(current_docs)}, New documents: {len(new_docs)}")
    
    # Step 4: Compare and find changes
    logger.info("Step 4: Comparing documents...")
    new_documents, updated_documents, deleted_titles = find_document_changes(current_docs, new_docs)
    
    logger.info(f"Changes detected:")
    logger.info(f"  - New documents: {len(new_documents)}")
    logger.info(f"  - Updated documents: {len(updated_documents)}")
    logger.info(f"  - Deleted documents: {len(deleted_titles)}")
    
    # Step 5: Update MongoDB
    if new_documents or updated_documents or deleted_titles:
        logger.info("Step 5: Updating MongoDB...")
        update_mongodb(new_documents, updated_documents, deleted_titles)
        
        # Replace current JSON with new JSON
        NEW_JSON_PATH.replace(CURRENT_JSON_PATH)
        logger.info(f"Replaced current JSON with new JSON: {CURRENT_JSON_PATH}")
    else:
        logger.info("No changes detected. MongoDB is up to date.")
    
    logger.info("Update complete!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

