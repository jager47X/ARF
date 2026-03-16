# ingest_agency_guidance.py
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
    parser = argparse.ArgumentParser(description="Ingest Agency Guidance documents")
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
        "--with-embeddings",
        action="store_true",
        help="Generate embeddings for documents at object level"
    )
    parser.add_argument(
        "--batch-embeddings",
        action="store_true",
        help="Use batch API for embeddings (faster, requires more memory)"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Drop collection and start from scratch"
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
    # Use the values that were loaded by the config module's auto-load
    MONGO_URI = config_module.MONGO_URI

# Import other config values
COLLECTION = config_module.COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_agency_guidance")
BASE_DIR = Path(__file__).resolve().parents[2]
AGENCY_GUIDANCE_DOCUMENT_PATH = str(BASE_DIR / "Data/Knowledge/agency_guidance.json")
AGENCY_GUIDANCE_CONF = COLLECTION.get("AGENCY_GUIDANCE_SET")
if not AGENCY_GUIDANCE_CONF:
    raise ValueError("AGENCY_GUIDANCE_SET not found in COLLECTION config. Please add AGENCY_GUIDANCE_SET configuration to config.py")
DB_NAME: str = AGENCY_GUIDANCE_CONF["db_name"]
COLL_NAME: str = AGENCY_GUIDANCE_CONF["main_collection_name"]

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
    Flatten entry to object level: combine all clause text into a single text field.
    Remove document_type, article, section fields.
    """
    title = entry_obj.get("title", "")
    date = entry_obj.get("date", "")
    agency = entry_obj.get("agency", "")
    clauses = entry_obj.get("clauses", [])
    text = entry_obj.get("text", "")

    # Combine all clause text into a single text field
    text_parts = []

    # If clauses exist, extract text from each clause
    if clauses and isinstance(clauses, list):
        for c in clauses:
            clause_text = c.get("text", "")
            if clause_text:
                text_parts.append(clause_text)

    # If flat text exists, add it
    if text:
        text_parts.append(text)

    # Join all text parts
    combined_text = " ".join(text_parts).strip()

    return {
        "title": title,
        "date": date,
        "agency": agency,
        "text": combined_text
    }

def generate_embeddings_for_docs_batch(embedder, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for multiple documents at object level using batch API"""
    # Collect all document texts to embed
    doc_texts = []
    doc_indices = []  # Track which doc each text belongs to

    for doc_idx, doc in enumerate(docs):
        # Document-level text: combine title and text
        doc_text_parts = []
        if doc.get("title"):
            doc_text_parts.append(doc["title"])
        if doc.get("text"):
            doc_text_parts.append(doc["text"])

        doc_text = " ".join(doc_text_parts)
        if doc_text:
            doc_texts.append(doc_text)
            doc_indices.append(doc_idx)

    # Generate document embeddings in batch
    if doc_texts:
        try:
            doc_embeddings = embedder.get_openai_embeddings_batch(doc_texts, batch_size=100)
            for i, doc_idx in enumerate(doc_indices):
                if i < len(doc_embeddings) and doc_embeddings[i] is not None:
                    docs[doc_idx]["embedding"] = doc_embeddings[i].tolist() if hasattr(doc_embeddings[i], 'tolist') else list(doc_embeddings[i])
        except Exception as e:
            logger.warning(f"Failed to generate document embeddings in batch: {e}")

    return docs

def ingest():
    client = None
    try:
        # Configure TLS for MongoDB Atlas connections
        tls_config = {}
        if MONGO_URI and "mongodb+srv://" in MONGO_URI:
            # MongoDB Atlas SRV connections require TLS
            tls_config = {"tls": True}
        elif MONGO_URI and ("mongodb.net" in MONGO_URI or "mongodb.com" in MONGO_URI):
            # MongoDB Atlas standard connections also require TLS
            tls_config = {"tls": True}

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, **tls_config)

        # Test connection with a ping
        try:
            client.admin.command('ping')
            logger.info("MongoDB connection test successful.")
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            raise

        db = client[DB_NAME]
        coll = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=0))
        logger.info("Connected to MongoDB (w=0).")

        # Drop collection if from-scratch
        if args and args.from_scratch:
            logger.warning("DROPPING COLLECTION '%s' - Starting from scratch!", COLL_NAME)
            coll.drop()
            logger.info("Collection dropped. Starting fresh ingestion.")
        else:
            # Drop non-_id indexes (fast ingest)
            try:
                for idx in list(coll.index_information().keys()):
                    if idx != "_id_":
                        coll.drop_index(idx)
                logger.info("Dropped non-_id indexes.")
            except Exception as e:
                logger.warning(f"Could not drop indexes (collection may be empty or new): {e}")
                # Continue anyway - collection might be new

        # Load Agency Guidance JSON
        data = load_json(AGENCY_GUIDANCE_DOCUMENT_PATH)
        if not data:
            return
        items = data.get("data", {}).get("agency_guidance", {}).get("documents", [])
        if not items:
            logger.warning("No 'documents' found in JSON.")
            return

        docs: List[Dict[str, Any]] = [normalize_to_hierarchy(obj) for obj in items]
        logger.info("Loaded %d documents from JSON.", len(docs))

        # Generate embeddings if requested
        if args and args.with_embeddings:
            from backend.services.rag.rag_dependencies.ai_service import LLM
            use_batch = args.batch_embeddings if args else False
            mode_str = "batch" if use_batch else "individual"
            logger.info(f"Generating embeddings using Voyage-3-large (1024 dimensions) in {mode_str} mode...")
            embedder = LLM(config=AGENCY_GUIDANCE_CONF)

            if use_batch:
                # Process in batches to avoid memory issues
                batch_size = 50  # Process 50 documents at a time
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    logger.info("Processing embedding batch %d-%d/%d", i + 1, min(i + batch_size, len(docs)), len(docs))
                    try:
                        batch = generate_embeddings_for_docs_batch(embedder, batch)
                        docs[i:i + batch_size] = batch
                    except Exception as e:
                        logger.error(f"Error processing embedding batch {i}-{i+len(batch)}: {e}")
            else:
                # Individual embeddings (faster per item)
                from concurrent.futures import ThreadPoolExecutor, as_completed
                def generate_embeddings_for_doc(embedder, doc):
                    """Generate embeddings for a single document at object level"""
                    # Document-level embedding: combine title and text
                    doc_text_parts = []
                    if doc.get("title"):
                        doc_text_parts.append(doc["title"])
                    if doc.get("text"):
                        doc_text_parts.append(doc["text"])

                    doc_text = " ".join(doc_text_parts)
                    if doc_text:
                        try:
                            doc_emb = embedder.get_openai_embedding(doc_text)
                            if doc_emb is not None:
                                doc["embedding"] = doc_emb.tolist() if hasattr(doc_emb, 'tolist') else list(doc_emb)
                        except Exception as e:
                            logger.warning(f"Failed to generate document embedding: {e}")

                    return doc

                # Process documents in parallel
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(generate_embeddings_for_doc, embedder, doc): i for i, doc in enumerate(docs)}
                    for future in as_completed(futures):
                        doc_idx = futures[future]
                        try:
                            docs[doc_idx] = future.result()
                        except Exception as e:
                            logger.error(f"Error generating embeddings for doc {doc_idx}: {e}")

            logger.info("Embeddings generation complete.")

        # Separate new docs and existing docs
        existing_docs = {d["title"]: d for d in coll.find({}, {"title": 1, "embedding": 1}) if "title" in d}
        existing_titles = set(existing_docs.keys())

        new_docs = [d for d in docs if d.get("title") and d["title"] not in existing_titles]
        existing_docs_to_update = []

        # Find existing documents that need embeddings or updates
        for doc in docs:
            if doc.get("title") and doc["title"] in existing_titles:
                existing_doc = existing_docs[doc["title"]]
                # Update if: no embedding exists, or if --from-scratch-like update is needed
                if "embedding" not in existing_doc or existing_doc.get("embedding") is None:
                    existing_docs_to_update.append(doc)

        logger.info("Prepared %d new docs, %d existing docs to update (missing embeddings), %d skipped (already have embeddings).",
                   len(new_docs), len(existing_docs_to_update), len(docs) - len(new_docs) - len(existing_docs_to_update))

        # Insert new documents
        if new_docs:
            try:
                res = coll.insert_many(new_docs, ordered=False)
                logger.info("Inserted %d new documents.", len(res.inserted_ids))
            except BulkWriteError as bwe:
                n = bwe.details.get("nInserted", 0)
                logger.warning("BulkWriteError; inserted %d docs.", n)
        else:
            logger.info("No new documents to insert.")

        # Update existing documents with embeddings
        if existing_docs_to_update:
            updated_count = 0
            for doc in existing_docs_to_update:
                try:
                    result = coll.update_one(
                        {"title": doc["title"]},
                        {"$set": {
                            "text": doc.get("text", ""),
                            "date": doc.get("date", ""),
                            "agency": doc.get("agency", ""),
                            "embedding": doc.get("embedding")
                        }}
                    )
                    if result.modified_count > 0:
                        updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update document '{doc.get('title')}': {e}")
            logger.info("Updated %d existing documents with embeddings.", updated_count)
        else:
            logger.info("No existing documents need updating.")

        # Recreate indexes
        coll.create_index("title", unique=True)
        logger.info("Indexes created: title (unique)")

    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise
    finally:
        if client:
            client.close()
            logger.info("Mongo connection closed.")

if __name__ == "__main__":
    ingest()

