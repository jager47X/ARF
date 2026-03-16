# migrate_agency_guidance.py
"""
Migration script to update existing agency guidance documents in MongoDB:
- Remove article, section, document_type fields
- Remove clauses structure
- Combine all clause text into a single text field
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from pymongo import MongoClient
from pymongo.errors import BulkWriteError

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
    parser = argparse.ArgumentParser(description="Migrate Agency Guidance documents in MongoDB")
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
        help="Show what would be updated without making changes"
    )
    return parser.parse_args()

# Load environment based on args
args = parse_args() if __name__ == "__main__" else None
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
logger = logging.getLogger("migrate_agency_guidance")

AGENCY_GUIDANCE_CONF = COLLECTION.get("AGENCY_GUIDANCE_SET")
if not AGENCY_GUIDANCE_CONF:
    raise ValueError("AGENCY_GUIDANCE_SET not found in COLLECTION config")
DB_NAME: str = AGENCY_GUIDANCE_CONF["db_name"]
COLL_NAME: str = AGENCY_GUIDANCE_CONF["main_collection_name"]

def combine_clause_text(doc: Dict[str, Any]) -> str:
    """Extract and combine all text from clauses into a single string."""
    text_parts = []

    # Get text from clauses if they exist
    clauses = doc.get("clauses", [])
    if clauses and isinstance(clauses, list):
        for clause in clauses:
            clause_text = clause.get("text", "")
            if clause_text:
                text_parts.append(clause_text)

    # Also check for existing text field
    existing_text = doc.get("text", "")
    if existing_text:
        text_parts.append(existing_text)

    # Join all text parts
    return " ".join(text_parts).strip()

def migrate_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a document to the new structure."""
    # Combine all clause text
    combined_text = combine_clause_text(doc)

    # Build new document with only required fields
    new_doc = {
        "title": doc.get("title", ""),
        "date": doc.get("date", ""),
        "agency": doc.get("agency", ""),
        "text": combined_text
    }

    # Preserve embedding if it exists (document-level)
    if "embedding" in doc:
        new_doc["embedding"] = doc["embedding"]

    # Preserve _id
    if "_id" in doc:
        new_doc["_id"] = doc["_id"]

    return new_doc

def migrate():
    """Migrate all documents in the collection."""
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
        total_count = coll.count_documents({})
        logger.info(f"Found {total_count} documents to migrate")

        if total_count == 0:
            logger.warning("No documents found in collection. Nothing to migrate.")
            return

        # Process documents in batches
        batch_size = 100
        updated_count = 0
        skipped_count = 0

        cursor = coll.find({})

        # First, check a sample document to see its structure
        sample_doc = coll.find_one({})
        if sample_doc:
            logger.info(f"Sample document fields: {list(sample_doc.keys())}")
            if "clauses" in sample_doc:
                logger.info(f"Sample document has {len(sample_doc.get('clauses', []))} clauses")

        for doc in cursor:
            try:
                # Check if document needs migration (has clauses, article, section, or document_type)
                needs_migration = (
                    "clauses" in doc or
                    "article" in doc or
                    "section" in doc or
                    "document_type" in doc
                )

                if not needs_migration:
                    skipped_count += 1
                    if skipped_count <= 3:  # Log first few skipped docs
                        logger.debug(f"Skipping document {doc.get('_id')} - already in new format (fields: {list(doc.keys())})")
                    continue

                # Migrate document
                new_doc = migrate_document(doc)

                if args and args.dry_run:
                    logger.info(f"[DRY RUN] Would update document: {doc.get('_id')} - Title: {doc.get('title', 'N/A')[:50]}")
                    logger.info(f"  - Old structure has: clauses={bool(doc.get('clauses'))}, article={bool(doc.get('article'))}, section={bool(doc.get('section'))}, document_type={bool(doc.get('document_type'))}")
                    logger.info(f"  - New text length: {len(new_doc.get('text', ''))}")
                    updated_count += 1
                else:
                    # Update document in place
                    update_result = coll.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "title": new_doc["title"],
                                "date": new_doc["date"],
                                "agency": new_doc["agency"],
                                "text": new_doc["text"]
                            },
                            "$unset": {
                                "clauses": "",
                                "article": "",
                                "section": "",
                                "document_type": ""
                            }
                        }
                    )

                    if update_result.modified_count > 0:
                        updated_count += 1
                        if updated_count % 100 == 0:
                            logger.info(f"Updated {updated_count} documents...")
                    else:
                        logger.warning(f"Document {doc.get('_id')} was not updated")

            except Exception as e:
                logger.error(f"Error migrating document {doc.get('_id', 'unknown')}: {e}")
                continue

        logger.info(f"Migration complete: {updated_count} documents updated, {skipped_count} documents skipped")

        # Drop old indexes that reference removed fields
        if not (args and args.dry_run):
            try:
                index_info = coll.index_information()
                indexes_to_drop = []

                for idx_name in index_info.keys():
                    if idx_name != "_id_":
                        idx_def = index_info[idx_name]
                        # Check if index references removed fields
                        if isinstance(idx_def, dict) and "key" in idx_def:
                            keys = idx_def["key"]
                            if any(key[0] in ["article", "section", "clauses"] for key in keys):
                                indexes_to_drop.append(idx_name)

                for idx_name in indexes_to_drop:
                    try:
                        coll.drop_index(idx_name)
                        logger.info(f"Dropped index: {idx_name}")
                    except Exception as e:
                        logger.warning(f"Could not drop index {idx_name}: {e}")

                # Ensure title index exists
                try:
                    coll.create_index("title", unique=True)
                    logger.info("Created/verified index: title (unique)")
                except Exception as e:
                    logger.warning(f"Could not create title index: {e}")

            except Exception as e:
                logger.warning(f"Error managing indexes: {e}")

        logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"Migration error: {e}", exc_info=True)
        raise
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    migrate()

