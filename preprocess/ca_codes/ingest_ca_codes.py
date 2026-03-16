# ingest_ca_codes.py
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

            # Also create 'services' module alias (for files using 'from services.rag.config')
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

                    # Also create 'services.rag' alias
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

                            # Also create 'services.rag.config' alias
                            sys.modules['services.rag.config'] = config_mod

                            # Load rag_dependencies
                            rag_deps_init = backend_dir / 'services' / 'rag' / 'rag_dependencies' / '__init__.py'
                            if rag_deps_init.exists():
                                spec = importlib.util.spec_from_file_location('backend.services.rag.rag_dependencies', rag_deps_init)
                                if spec and spec.loader:
                                    rag_deps_mod = importlib.util.module_from_spec(spec)
                                    sys.modules['backend.services.rag.rag_dependencies'] = rag_deps_mod
                                    spec.loader.exec_module(rag_deps_mod)
                                    setattr(rag_mod, 'rag_dependencies', rag_deps_mod)

                                    # Also create 'services.rag.rag_dependencies' alias
                                    sys.modules['services.rag.rag_dependencies'] = rag_deps_mod

# Parse arguments before importing config
def parse_args():
    parser = argparse.ArgumentParser(description="Ingest California Codes")
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
# We need to import the module first, then call load_environment to override the default
# Note: The config module auto-loads on import, but we can override it with override=True
import backend.services.rag.config as config_module

# Now load the correct environment (this will override the default with override=True)
if env_override:
    config_module.load_environment(env_override)
    # Re-read MONGO_URI after loading the correct environment
    import os
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found in {config_module._env_file_used}")
else:
    # Use the values that were loaded by the config module's auto-load
    MONGO_URI = config_module.MONGO_URI

# Import other config values (COLLECTION doesn't depend on env vars)
COLLECTION = config_module.COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_ca_codes")
BASE_DIR = Path(__file__).resolve().parents[2]
CA_CODES_DOCUMENT_PATH = str(BASE_DIR / "Data/Knowledge/ca_code.json")
CA_CODES_CONF = COLLECTION.get("CA_CODES_SET")
if not CA_CODES_CONF:
    raise ValueError("CA_CODES_SET not found in COLLECTION config. Please add CA_CODES_SET configuration to config.py")
DB_NAME: str = CA_CODES_CONF["db_name"]
COLL_NAME: str = CA_CODES_CONF["main_collection_name"]

def load_json(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to read JSON: %s", e)
        return None

def normalize_to_hierarchy(entry_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each entry has `clauses: [...]`.
    If `text` (flat) is present and `clauses` missing, wrap it as a single clause.
    """
    code = entry_obj.get("code", "")
    division = entry_obj.get("division", "")
    part = entry_obj.get("part", "")
    chapter = entry_obj.get("chapter", "")
    section = entry_obj.get("section", "")
    title = entry_obj.get("title", "")
    clauses = entry_obj.get("clauses")

    if clauses and isinstance(clauses, list):
        # Already hierarchical — just ensure each clause has number/title/text
        clean_clauses = []
        for c in clauses:
            clean_clauses.append({
                "number": c.get("number"),
                "title": c.get("title") or "",
                "text": c.get("text") or "",
            })
        return {"code": code, "division": division, "part": part, "chapter": chapter, "section": section, "title": title, "clauses": clean_clauses}

    # Flat -> wrap
    text = entry_obj.get("text", "")
    if not text:
        # Keep empty clause list; can be filled later if needed
        return {"code": code, "division": division, "part": part, "chapter": chapter, "section": section, "title": title, "clauses": []}

    # Use clause number 1, clause title identical to section title for consistency
    return {
        "code": code,
        "division": division,
        "part": part,
        "chapter": chapter,
        "section": section,
        "title": title,
        "clauses": [{
            "number": 1,
            "title": title,
            "text": text,
        }]
    }

def ingest():
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        coll = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=0))
        logger.info("Connected to MongoDB (w=0).")

        # Drop non-_id indexes (fast ingest)
        for idx in list(coll.index_information().keys()):
            if idx != "_id_":
                coll.drop_index(idx)
        logger.info("Dropped non-_id indexes.")

        # Load CA Codes JSON
        data = load_json(CA_CODES_DOCUMENT_PATH)
        if not data:
            return
        items = data.get("data", {}).get("california_codes", {}).get("codes", [])
        if not items:
            logger.warning("No 'codes' found in JSON.")
            return

        docs: List[Dict[str, Any]] = [normalize_to_hierarchy(obj) for obj in items]

        # Deduplicate by top-level title (section title)
        existing_titles = {d["title"] for d in coll.find({}, {"title": 1}) if "title" in d}
        new_docs = [d for d in docs if d.get("title") and d["title"] not in existing_titles]
        logger.info("Prepared %d new docs (skipped %d duplicates by title).", len(new_docs), len(docs) - len(new_docs))

        if new_docs:
            try:
                res = coll.insert_many(new_docs, ordered=False)
                logger.info("Inserted %d documents.", len(res.inserted_ids))
            except BulkWriteError as bwe:
                n = bwe.details.get("nInserted", 0)
                logger.warning("BulkWriteError; inserted %d docs.", n)
        else:
            logger.info("No new documents to insert.")

        # Recreate indexes
        coll.create_index("title", unique=True)
        coll.create_index([("code", 1), ("division", 1), ("part", 1), ("chapter", 1), ("section", 1)])
        # Unique across *clause titles* (multikey). Assumes each clause title is unique across CA codes.
        coll.create_index("clauses.title", unique=True)
        logger.info("Indexes created: title (unique), (code,division,part,chapter,section), clauses.title (unique)")

    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise
    finally:
        if client:
            client.close()
            logger.info("Mongo connection closed.")

if __name__ == "__main__":
    ingest()

