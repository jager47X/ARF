#!/usr/bin/env python3
"""
Direct ingestion script - inserts CFR JSON data exactly as-is into MongoDB.
No transformations, no normalization, no grouping.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from pymongo import MongoClient, WriteConcern
from pymongo.errors import BulkWriteError

# Setup path for module execution
# The directory is 'kyr-backend' but imports use 'backend'
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create 'backend' -> 'kyr-backend' mapping for imports
# Python can't import modules with hyphens, so we create an alias
import importlib.util
import types

# Set up backend module structure to point to kyr-backend directory
# Also set up 'services' module alias for files that use 'from services.rag.config'
if 'backend' not in sys.modules:
    # Create backend module
    backend_mod = types.ModuleType('backend')
    sys.modules['backend'] = backend_mod

    # Create services submodule
    services_mod = types.ModuleType('backend.services')
    backend_mod.services = services_mod
    sys.modules['backend.services'] = services_mod

    # Create rag submodule
    rag_mod = types.ModuleType('backend.services.rag')
    services_mod.rag = rag_mod
    sys.modules['backend.services.rag'] = rag_mod

# Load the actual config module from kyr-backend/services/rag/config.py
config_path = backend_dir / "services" / "rag" / "config.py"
spec = importlib.util.spec_from_file_location("backend.services.rag.config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["backend.services.rag.config"] = config_module
spec.loader.exec_module(config_module)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_cfr_direct")

BASE_DIR = Path(__file__).resolve().parents[2]
CFR_DOCUMENT_PATH = str(BASE_DIR / "Data/Knowledge/code_of_federal_regulations.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Direct CFR ingestion - no transformations")
    parser.add_argument("--production", action="store_true", help="Use production environment")
    parser.add_argument("--dev", action="store_true", help="Use dev environment")
    parser.add_argument("--local", action="store_true", help="Use local environment")
    return parser.parse_args()

args = parse_args()
env_override = None
if args.production:
    env_override = "production"
elif args.dev:
    env_override = "dev"
elif args.local:
    env_override = "local"

if env_override:
    config_module.load_environment(env_override)
    import os
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found in {config_module._env_file_used}")
else:
    MONGO_URI = config_module.MONGO_URI

COLLECTION = config_module.COLLECTION
CFR_CONF = COLLECTION.get("CFR_SET")
if not CFR_CONF:
    raise ValueError("CFR_SET not found in COLLECTION config")

DB_NAME: str = CFR_CONF["db_name"]
COLL_NAME: str = CFR_CONF["main_collection_name"]

def load_json(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load JSON: %s", e)
        return None

def ingest():
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

        db = client[DB_NAME]
        coll = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=0))
        logger.info("Connected to MongoDB (w=0).")

        # Load CFR JSON
        data = load_json(CFR_DOCUMENT_PATH)
        if not data:
            return

        # Get regulations array directly from JSON structure
        cfr_data = data.get("data", {}).get("code_of_federal_regulations", {})
        regulations = cfr_data.get("regulations", [])

        if not regulations:
            logger.warning("No 'regulations' found in JSON.")
            return

        logger.info("Found %d regulations in JSON. Inserting exactly as-is (one by one)...", len(regulations))

        # Insert documents exactly as they are in JSON, one by one to handle large documents
        inserted_count = 0
        failed_count = 0
        for i, reg in enumerate(regulations):
            try:
                coll.insert_one(reg)
                inserted_count += 1
                if (i + 1) % 100 == 0:
                    logger.info("Inserted %d/%d documents...", inserted_count, len(regulations))
            except Exception as e:
                failed_count += 1
                logger.warning("Failed to insert document %d: %s", i + 1, str(e)[:100])

        logger.info("Inserted %d documents. Failed: %d", inserted_count, failed_count)

        # Create basic indexes
        try:
            coll.create_index("title", unique=False)
            logger.info("Index created: title")
        except Exception as e:
            logger.warning(f"Could not create index: {e}")

    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise
    finally:
        if client:
            client.close()
            logger.info("Mongo connection closed.")

if __name__ == "__main__":
    ingest()

