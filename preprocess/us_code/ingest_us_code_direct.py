#!/usr/bin/env python3
"""
Direct ingestion script - inserts US Code JSON data exactly as-is into MongoDB.
No transformations, no normalization, no grouping.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List
from pymongo import MongoClient, WriteConcern
from pymongo.errors import BulkWriteError

# Setup path for module execution - same as ingest_us_code.py
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create 'backend' -> 'kyr-backend' mapping for imports
import importlib.util
import types

# Set up backend module structure to point to kyr-backend directory
if 'backend' not in sys.modules:
    # Create backend module
    backend_mod = types.ModuleType('backend')
    sys.modules['backend'] = backend_mod
    
    # Load services
    services_init = backend_dir / 'services' / '__init__.py'
    services_mod = None
    if services_init.exists():
        spec = importlib.util.spec_from_file_location('backend.services', services_init)
        if spec and spec.loader:
            services_mod = importlib.util.module_from_spec(spec)
            sys.modules['backend.services'] = services_mod
            spec.loader.exec_module(services_mod)
            setattr(backend_mod, 'services', services_mod)
            
            # Also create 'services' module alias
            if 'services' not in sys.modules:
                sys.modules['services'] = services_mod
            
            # Load rag
            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
            rag_mod = None
            if rag_init.exists():
                spec = importlib.util.spec_from_file_location('backend.services.rag', rag_init)
                if spec and spec.loader:
                    rag_mod = importlib.util.module_from_spec(spec)
                    sys.modules['backend.services.rag'] = rag_mod
                    spec.loader.exec_module(rag_mod)
                    setattr(services_mod, 'rag', rag_mod)
                    sys.modules['services.rag'] = rag_mod
                    
                    # Load config
                    config_file = backend_dir / 'services' / 'rag' / 'config.py'
                    if config_file.exists():
                        spec = importlib.util.spec_from_file_location('backend.services.rag.config', config_file)
                        if spec and spec.loader:
                            config_mod = importlib.util.module_from_spec(spec)
                            sys.modules['backend.services.rag.config'] = config_mod
                            spec.loader.exec_module(config_mod)
                            setattr(rag_mod, 'config', config_mod)
                            sys.modules['services.rag.config'] = config_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_us_code_direct")

BASE_DIR = Path(__file__).resolve().parents[2]
US_CODE_DOCUMENT_PATH_NEW = str(Path(__file__).resolve().parent / "usc_xml_temp" / "us_code.json")
US_CODE_DOCUMENT_PATH_OLD = str(BASE_DIR / "Data/Knowledge/us_code.json")
US_CODE_DOCUMENT_PATH = US_CODE_DOCUMENT_PATH_NEW if os.path.exists(US_CODE_DOCUMENT_PATH_NEW) else US_CODE_DOCUMENT_PATH_OLD

def parse_args():
    parser = argparse.ArgumentParser(description="Direct US Code ingestion - no transformations")
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
    import backend.services.rag.config as config_module
    config_module.load_environment(env_override)
    import os
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found")
else:
    import backend.services.rag.config as config_module
    MONGO_URI = config_module.MONGO_URI

COLLECTION = config_module.COLLECTION
US_CODE_CONF = COLLECTION.get("US_CODE_SET")
if not US_CODE_CONF:
    raise ValueError("US_CODE_SET not found in COLLECTION config")

DB_NAME: str = US_CODE_CONF["db_name"]
COLL_NAME: str = US_CODE_CONF["main_collection_name"]

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

        # Load US Code JSON
        data = load_json(US_CODE_DOCUMENT_PATH)
        if not data:
            return
        
        # Get titles array directly from JSON structure
        us_code_data = data.get("data", {}).get("united_states_code", {})
        titles = us_code_data.get("titles", [])
        
        if not titles:
            logger.warning("No 'titles' found in JSON.")
            return
        
        logger.info("Found %d titles in JSON. Inserting exactly as-is (one by one)...", len(titles))
        
        # Insert documents exactly as they are in JSON, one by one to handle large documents
        inserted_count = 0
        failed_count = 0
        for i, title in enumerate(titles):
            try:
                coll.insert_one(title)
                inserted_count += 1
                if (i + 1) % 100 == 0:
                    logger.info("Inserted %d/%d documents...", inserted_count, len(titles))
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












































