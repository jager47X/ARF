# ingest_public_cases.py
import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List
from pymongo import MongoClient, WriteConcern
from pymongo.errors import BulkWriteError, OperationFailure

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
    parser = argparse.ArgumentParser(description="Ingest public cases")
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
        "--from-scratch",
        action="store_true",
        help="Drop collection and ingest from scratch (WARNING: deletes all existing data!)"
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generate embeddings using Voyage-3-large during ingestion"
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

# Now load the correct environment (this will override the default with override=True)
if env_override:
    config_module.load_environment(env_override)
    # Re-read MONGO_URI after loading the correct environment
    import os
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found in {config_module._env_file_used}")
else:
    MONGO_URI = config_module.MONGO_URI

# Import other config values
ALIAS_COLL_NAME = config_module.ALIAS_COLL_NAME
COLLECTION = config_module.COLLECTION
EMBEDDING_DIMENSIONS = config_module.EMBEDDING_DIMENSIONS
from backend.services.rag.rag_dependencies.ai_service import LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_public_cases")

BASE_DIR = Path(__file__).resolve().parents[2]
USC_CASES_PATH = str(BASE_DIR / "Data/Knowledge/public_cases.json")
USC_CONF = COLLECTION["US_CONSTITUTION_SET"]
COLL_NAME = USC_CONF["cases_collection_name"]  # "supreme_court_cases"
DB_NAME: str = USC_CONF["db_name"]  # "public"
def load_json(path: str) -> Any | None:
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        logger.error("Failed to read JSON: %s", e)
        return None

def _normalize_refs(refs: Any) -> List[str]:
    if refs is None: return []
    if isinstance(refs, list): return [str(r) for r in refs if r is not None]
    return [str(refs)]

def _from_list_form(category: str, c: list) -> Dict[str, Any]:
    case_title = str(c[0]).strip() if len(c) > 0 and c[0] is not None else ""
    summary    = str(c[1]).strip() if len(c) > 1 and c[1] is not None else ""
    refs       = _normalize_refs(c[2]) if len(c) > 2 else []
    return {"category": category, "case": case_title, "summary": summary, "references": refs}

def _from_string_form(category: str, c: str) -> Dict[str, Any]:
    return {"category": category, "case": str(c).strip(), "summary": "", "references": []}

def flatten_cases(data: Any) -> List[Dict[str, Any]]:
    """Robust flattener; tolerates nested lists/strings/dicts under each category."""
    all_cases: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for block in data:
            all_cases.extend(flatten_cases(block))
        return all_cases
    if not isinstance(data, dict):
        logger.error("Unexpected JSON structure at top: %s", type(data))
        return all_cases

    cases_node = data.get("cases", {})
    if isinstance(cases_node, dict):
        for category, case_list in cases_node.items():
            if not isinstance(case_list, list):
                continue
            # flatten one nested level (e.g., a list of dicts inside the list)
            flat_items = []
            for item in case_list:
                flat_items.extend(item if isinstance(item, list) else [item])
            for c in flat_items:
                if isinstance(c, dict):
                    all_cases.append({
                        "category": category,
                        "case": (c.get("case") or "").strip(),
                        "summary": (c.get("summary") or "").strip(),
                        "references": _normalize_refs(c.get("references")),
                    })
                elif isinstance(c, list):
                    all_cases.append(_from_list_form(category, c))
                elif isinstance(c, str):
                    all_cases.append(_from_string_form(category, c))
        return all_cases

    if isinstance(cases_node, list):
        category = "Uncategorized"
        for c in cases_node:
            if isinstance(c, dict):
                all_cases.append({
                    "category": category,
                    "case": (c.get("case") or "").strip(),
                    "summary": (c.get("summary") or "").strip(),
                    "references": _normalize_refs(c.get("references")),
                })
            elif isinstance(c, list):
                all_cases.append(_from_list_form(category, c))
            elif isinstance(c, str):
                all_cases.append(_from_string_form(category, c))
        return all_cases

    logger.error("Expected 'cases' to be dict or list; got %s", type(cases_node))
    return all_cases

def compose_case_text(doc: Dict[str, Any]) -> str:
    """Compose text for case embedding from case title and summary"""
    case_title = doc.get("case", "").strip()
    summary = doc.get("summary", "").strip()
    category = doc.get("category", "").strip()
    
    parts = []
    if case_title:
        parts.append(f"Case: {case_title}")
    if category:
        parts.append(f"Category: {category}")
    if summary:
        parts.append(f"Summary: {summary}")
    
    return "\n\n".join(parts) if parts else ""

def generate_embeddings_for_cases(embedder: LLM, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for all cases using batch API (Voyage-3-large)"""
    # Collect all case texts
    case_texts = []
    valid_indices = []
    
    for idx, case_doc in enumerate(cases):
        text = compose_case_text(case_doc)
        if text.strip():
            case_texts.append(text)
            valid_indices.append(idx)
        else:
            logger.warning(f"No text found for case: {case_doc.get('case', 'Unknown')}")
    
    if not case_texts:
        logger.warning("No valid case texts to embed")
        return cases
    
    logger.info(f"Generating embeddings for {len(case_texts)} cases using batch API (Voyage-3-large, 1024 dimensions)...")
    
    # Generate embeddings in batches
    try:
        embeddings = embedder.get_openai_embeddings_batch(case_texts, batch_size=100)
        
        # Assign embeddings back to cases
        for i, idx in enumerate(valid_indices):
            if i < len(embeddings) and embeddings[i] is not None:
                cases[idx]["embedding"] = embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i])
            else:
                logger.warning(f"Failed to generate embedding for case: {cases[idx].get('case', 'Unknown')}")
        
        logger.info(f"Generated embeddings for {len(valid_indices)}/{len(cases)} cases")
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
    
    logger.info("Completed embedding generation for all cases.")
    return cases

def _dedupe_by_case(coll_ack) -> int:
    """Delete duplicates by 'case', keeping one doc per case. Returns #deleted."""
    deleted = 0
    cursor = coll_ack.aggregate([
        {"$match": {"case": {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": "$case", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}}
    ])
    for g in cursor:
        ids = g["ids"]
        keep = ids[0]
        to_remove = ids[1:]
        if to_remove:
            res = coll_ack.delete_many({"_id": {"$in": to_remove}})
            deleted += res.deleted_count
    if deleted:
        logger.info("De-duplicated %d documents by 'case'.", deleted)
    return deleted

def ingest():
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        # Fast, unacknowledged handle for bulk insert
        coll = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=0))
        logger.info("Connected to MongoDB (w=0).")

        # Drop collection if from-scratch
        if args and args.from_scratch:
            logger.warning("DROPPING COLLECTION '%s' - Starting from scratch!", COLL_NAME)
            coll.drop()
            logger.info("Collection dropped. Starting fresh ingestion.")
        else:
            # Drop non-_id indexes (quietly)
            for idx in list(coll.index_information().keys()):
                if idx != "_id_":
                    try: coll.drop_index(idx)
                    except Exception: pass
            logger.info("Dropped non-_id indexes.")

        data = load_json(USC_CASES_PATH)
        if not data:
            logger.error("No data loaded from JSON.")
            return
        cases = flatten_cases(data)
        if not cases:
            logger.info("No cases found.")
            return
        
        logger.info("Loaded %d cases from JSON.", len(cases))

        # Generate embeddings if requested
        if args and args.with_embeddings:
            embedder = LLM(config=USC_CONF)
            cases = generate_embeddings_for_cases(embedder, cases)

        # Precompute existing case titles (only if not from-scratch)
        if args and not args.from_scratch:
            existing_titles = {d.get("case", "") for d in db.get_collection(COLL_NAME, write_concern=WriteConcern(w=1)).find({}, {"case": 1}) if d.get("case")}
            new_docs = [d for d in cases if d.get("case") and d.get("case") not in existing_titles]
            logger.info("Prepared %d new docs (skipped %d by preview dedupe).", len(new_docs), len(cases) - len(new_docs))
        else:
            new_docs = cases
            logger.info("Prepared %d documents for insertion.", len(new_docs))

        if new_docs:
            now = datetime.datetime.utcnow()
            for doc in new_docs:
                doc["created_at"] = now
                doc["updated_at"] = now
            try:
                coll.insert_many(new_docs, ordered=False)
                logger.info("Inserted up to %d new cases.", len(new_docs))
            except BulkWriteError as bwe:
                # Silently swallow duplicate key writeErrors; log count at INFO
                dup_count = sum(1 for err in bwe.details.get("writeErrors", []) if err.get("code") == 11000)
                other_errs = [e for e in bwe.details.get("writeErrors", []) if e.get("code") != 11000]
                if dup_count:
                    logger.info("Skipped %d duplicate documents during insert.", dup_count)
                if other_errs:
                    # Surface non-duplicate issues
                    logger.error("Bulk write had non-duplicate errors: %s", other_errs)

        # Acknowledged handle for cleanup + indexes
        coll_ack = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=1))

        # Proactively dedupe before building unique index
        _dedupe_by_case(coll_ack)

        # Build unique index on 'case' (quiet retry on dup failures)
        try:
            coll_ack.create_index([("case", 1)], name="case_1", unique=True)
        except OperationFailure as e:
            msg = str(e)
            if getattr(e, "code", None) == 11000 or "E11000" in msg:
                logger.info("Unique index build hit duplicates; cleaning and retrying.")
                _dedupe_by_case(coll_ack)
                # Drop partial/broken index if present, then retry
                try: coll_ack.drop_index("case_1")
                except Exception: pass
                coll_ack.create_index([("case", 1)], name="case_1", unique=True)
            else:
                raise

        # Secondary indexes
        coll_ack.create_index([("category", 1)])
        coll_ack.create_index([("references", 1)])
        logger.info("Indexes ready: case(unique), category, references")

    except Exception as e:
        logger.error("Ingest error: %s", e)
    finally:
        if client:
            client.close()
            logger.info("Mongo connection closed.")

if __name__ == "__main__":
    ingest()
