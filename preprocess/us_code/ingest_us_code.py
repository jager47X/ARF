# ingest_us_code.py
import os
import sys
import json
import logging
import argparse
import re
import time
import datetime
from typing import Any, Dict, List
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="Ingest United States Code")
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
        help="Generate embeddings for documents and clauses"
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
logger = logging.getLogger("ingest_us_code")
BASE_DIR = Path(__file__).resolve().parents[2]
# Try new location first, fallback to old location
US_CODE_DOCUMENT_PATH_NEW = str(Path(__file__).resolve().parent / "usc_xml_temp" / "us_code.json")
US_CODE_DOCUMENT_PATH_OLD = str(BASE_DIR / "Data/Knowledge/us_code.json")
US_CODE_DOCUMENT_PATH = US_CODE_DOCUMENT_PATH_NEW if os.path.exists(US_CODE_DOCUMENT_PATH_NEW) else US_CODE_DOCUMENT_PATH_OLD
US_CODE_CONF = COLLECTION.get("US_CODE_SET")
if not US_CODE_CONF:
    raise ValueError("US_CODE_SET not found in COLLECTION config. Please add US_CODE_SET configuration to config.py")
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
        logger.error("Failed to read JSON: %s", e)
        return None

def normalize_number(num_str: str) -> str:
    """Normalize number by removing quotes, parentheses, and extra formatting.
    Converts letters to numbers: a->1, b->2, etc."""
    if not num_str:
        return ""
    
    num_str = str(num_str).strip()
    # Remove leading/trailing quotes
    num_str = num_str.strip('"\'')
    # Remove parentheses
    num_str = num_str.strip('()')
    # Remove any remaining quotes
    num_str = num_str.strip('"\'')
    num_str = num_str.strip()
    
    # Convert single letter to number (a=1, b=2, etc.)
    if len(num_str) == 1 and num_str.isalpha():
        letter_num = ord(num_str.lower()) - ord('a') + 1
        return str(letter_num)
    
    return num_str

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

def normalize_title(title_str: str) -> str:
    """Normalize title by removing '--' and '.' characters."""
    if not title_str:
        return ""
    
    title_str = str(title_str).strip()
    # Remove '--' (double dashes)
    title_str = title_str.replace('--', '')
    # Remove '.' (periods)
    title_str = title_str.replace('.', '')
    # Clean up any extra spaces that might result
    title_str = re.sub(r'\s+', ' ', title_str)
    return title_str.strip()

def normalize_to_hierarchy(entry_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert entry to MongoDB structure with `clauses: [...]`.
    Handles both old format (with `clauses` or flat `text`) and new format (with `section` array).
    Normalizes chapter, section, and clause numbers.
    """
    article = entry_obj.get("article", "")
    chapter = entry_obj.get("chapter", "")
    section = entry_obj.get("section", "")
    title = normalize_title(entry_obj.get("title", ""))
    
    # Normalize chapter
    chapter = normalize_chapter(chapter)
    
    # NEW FORMAT: section is an array of section entries
    if isinstance(section, list):
        # Convert section array to clauses array
        clauses = []
        section_num = ""  # Will be set from first section entry
        
        for idx, sec_entry in enumerate(section):
            if isinstance(sec_entry, dict):
                # Get section number from first entry (all entries in same section have same number)
                if not section_num:
                    section_num = str(sec_entry.get("number", ""))
                
                # Renumber clauses sequentially starting from 1
                clause_num = str(idx + 1)
                
                # Get and normalize clause title
                clause_title = normalize_title(sec_entry.get("title", "") or "")
                
                # Each section entry becomes a clause
                clause = {
                    "number": clause_num,
                    "title": clause_title,
                    "text": sec_entry.get("text", "") or "",
                }
                clauses.append(clause)
        
        # Normalize section number format
        section_formatted = normalize_section(section_num)
        
        # If there's only one clause and it has no title, use section title
        if len(clauses) == 1 and not clauses[0].get("title"):
            clauses[0]["title"] = title
        
        # Use first clause's title as document title if no title provided
        if not title and clauses:
            title = clauses[0].get("title", "") or ""
        
        return {
            "article": article,
            "chapter": chapter,
            "section": section_formatted,  # Format as "Section X"
            "title": title,
            "clauses": clauses
        }
    
    # OLD FORMAT: Check for existing clauses
    clauses = entry_obj.get("clauses")
    if clauses and isinstance(clauses, list):
        # Already hierarchical — renumber clauses sequentially starting from 1
        clean_clauses = []
        for idx, c in enumerate(clauses):
            # Renumber clauses sequentially starting from 1
            clause_num = str(idx + 1)
            clause_title = normalize_title(c.get("title") or "")
            clean_clauses.append({
                "number": clause_num,
                "title": clause_title,
                "text": c.get("text") or "",
            })
        
        # Normalize section format
        section_formatted = normalize_section(section)
        
        # If there's only one clause and it has no title, use section title
        if len(clean_clauses) == 1 and not clean_clauses[0].get("title"):
            clean_clauses[0]["title"] = title
        
        return {"article": article, "chapter": chapter, "section": section_formatted, "title": title, "clauses": clean_clauses}

    # OLD FORMAT: Flat -> wrap
    text = entry_obj.get("text", "")
    if not text:
        # Keep empty clause list; can be filled later if needed
        section_formatted = normalize_section(section)
        return {"article": article, "chapter": chapter, "section": section_formatted, "title": title, "clauses": []}

    # Normalize section format
    section_formatted = normalize_section(section)
    
    # Use clause number 1, clause title identical to section title for consistency
    return {
        "article": article,
        "chapter": chapter,
        "section": section_formatted,
        "title": title,
        "clauses": [{
            "number": "1",
            "title": title,
            "text": text,
        }]
    }

def generate_embeddings_for_docs_batch(embedder, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for multiple documents and their clauses using batch API"""
    # Collect all texts to embed
    doc_texts = []
    clause_texts = []
    doc_indices = []  # Track which doc each text belongs to
    clause_indices = []  # Track which clause each text belongs to (doc_idx, clause_idx)
    
    for doc_idx, doc in enumerate(docs):
        # Document-level text
        doc_text_parts = []
        if doc.get("title"):
            doc_text_parts.append(doc["title"])
        if doc.get("article"):
            doc_text_parts.append(f"Article {doc['article']}")
        if doc.get("chapter"):
            doc_text_parts.append(f"Chapter {doc['chapter']}")
        if doc.get("section"):
            doc_text_parts.append(f"Section {doc['section']}")
        
        doc_text = " ".join(doc_text_parts)
        if doc_text:
            doc_texts.append(doc_text)
            doc_indices.append(doc_idx)
        
        # Clause-level texts
        clauses = doc.get("clauses", [])
        for clause_idx, clause in enumerate(clauses):
            clause_text_parts = []
            if clause.get("title"):
                clause_text_parts.append(clause["title"])
            if clause.get("text"):
                clause_text_parts.append(clause["text"])
            
            clause_text = " ".join(clause_text_parts)
            if clause_text:
                clause_texts.append(clause_text)
                clause_indices.append((doc_idx, clause_idx))
    
    # Generate document embeddings in batch
    if doc_texts:
        try:
            doc_embeddings = embedder.get_openai_embeddings_batch(doc_texts, batch_size=100)
            for i, doc_idx in enumerate(doc_indices):
                if i < len(doc_embeddings) and doc_embeddings[i] is not None:
                    docs[doc_idx]["embedding"] = doc_embeddings[i].tolist() if hasattr(doc_embeddings[i], 'tolist') else list(doc_embeddings[i])
        except Exception as e:
            logger.warning(f"Failed to generate document embeddings in batch: {e}")
    
    # Generate clause embeddings in batch
    if clause_texts:
        try:
            clause_embeddings = embedder.get_openai_embeddings_batch(clause_texts, batch_size=100)
            for i, (doc_idx, clause_idx) in enumerate(clause_indices):
                if i < len(clause_embeddings) and clause_embeddings[i] is not None:
                    if "clauses" not in docs[doc_idx]:
                        docs[doc_idx]["clauses"] = []
                    while len(docs[doc_idx]["clauses"]) <= clause_idx:
                        docs[doc_idx]["clauses"].append({})
                    docs[doc_idx]["clauses"][clause_idx]["embedding"] = clause_embeddings[i].tolist() if hasattr(clause_embeddings[i], 'tolist') else list(clause_embeddings[i])
        except Exception as e:
            logger.warning(f"Failed to generate clause embeddings in batch: {e}")
    
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

        # Load US Code JSON
        data = load_json(US_CODE_DOCUMENT_PATH)
        if not data:
            return
        items = data.get("data", {}).get("united_states_code", {}).get("titles", [])
        if not items:
            logger.warning("No 'titles' found in JSON.")
            return

        docs: List[Dict[str, Any]] = [normalize_to_hierarchy(obj) for obj in items]
        logger.info("Loaded %d documents from JSON.", len(docs))

        # Generate embeddings if requested
        if args and args.with_embeddings:
            from backend.services.rag.rag_dependencies.ai_service import LLM
            use_batch = args.batch_embeddings if args else False
            mode_str = "batch" if use_batch else "individual"
            logger.info(f"Generating embeddings using Voyage-3-large (1024 dimensions) in {mode_str} mode...")
            embedder = LLM(config=US_CODE_CONF)
            
            if use_batch:
                # Process in batches to avoid memory issues
                batch_size = 50  # Process 50 documents at a time
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    logger.info("Processing embedding batch %d-%d/%d", i + 1, min(i + batch_size, len(docs)), len(docs))
                    try:
                        batch = generate_embeddings_for_docs_batch(embedder, batch)
                        docs[i:i + batch_size] = batch
                        # Small delay between batches to stay under rate limit (rate limiter handles per-request, this is extra safety)
                        if i + batch_size < len(docs):
                            time.sleep(0.1)  # 100ms delay between batches
                    except Exception as e:
                        logger.error(f"Error processing embedding batch {i}-{i+len(batch)}: {e}")
                        # Wait a bit longer on error before retrying
                        time.sleep(1.0)
            else:
                # Individual embeddings (sequential to stay under rate limit)
                def generate_embeddings_for_doc(embedder, doc):
                    """Generate embeddings for a single document and its clauses"""
                    # Document-level embedding
                    doc_text_parts = []
                    if doc.get("title"):
                        doc_text_parts.append(doc["title"])
                    if doc.get("article"):
                        doc_text_parts.append(f"Article {doc['article']}")
                    if doc.get("chapter"):
                        doc_text_parts.append(f"Chapter {doc['chapter']}")
                    if doc.get("section"):
                        doc_text_parts.append(f"Section {doc['section']}")
                    
                    doc_text = " ".join(doc_text_parts)
                    if doc_text:
                        try:
                            doc_emb = embedder.get_openai_embedding(doc_text)
                            if doc_emb is not None:
                                doc["embedding"] = doc_emb.tolist() if hasattr(doc_emb, 'tolist') else list(doc_emb)
                        except Exception as e:
                            logger.warning(f"Failed to generate document embedding: {e}")
                    
                    # Clause-level embeddings
                    clauses = doc.get("clauses", [])
                    for clause in clauses:
                        clause_text_parts = []
                        if clause.get("title"):
                            clause_text_parts.append(clause["title"])
                        if clause.get("text"):
                            clause_text_parts.append(clause["text"])
                        
                        clause_text = " ".join(clause_text_parts)
                        if clause_text:
                            try:
                                clause_emb = embedder.get_openai_embedding(clause_text)
                                if clause_emb is not None:
                                    clause["embedding"] = clause_emb.tolist() if hasattr(clause_emb, 'tolist') else list(clause_emb)
                            except Exception as e:
                                logger.warning(f"Failed to generate clause embedding: {e}")
                    
                    return doc
                
                # Process documents sequentially to avoid rate limit issues
                # With rate limiting, sequential processing ensures we stay under the limit
                for i, doc in enumerate(docs):
                    if (i + 1) % 100 == 0:
                        logger.info("Processed %d/%d documents", i + 1, len(docs))
                    try:
                        docs[i] = generate_embeddings_for_doc(embedder, doc)
                    except Exception as e:
                        logger.error(f"Error generating embeddings for doc {i}: {e}")
            
            logger.info("Embeddings generation complete.")

        # Deduplicate by top-level title (section title)
        existing_titles = {d["title"] for d in coll.find({}, {"title": 1}) if "title" in d}
        new_docs = [d for d in docs if d.get("title") and d["title"] not in existing_titles]
        logger.info("Prepared %d new docs (skipped %d duplicates by title).", len(new_docs), len(docs) - len(new_docs))

        if new_docs:
            now = datetime.datetime.utcnow()
            for doc in new_docs:
                doc["created_at"] = now
                doc["updated_at"] = now
            try:
                res = coll.insert_many(new_docs, ordered=False)
                logger.info("Inserted %d documents.", len(res.inserted_ids))
            except BulkWriteError as bwe:
                n = bwe.details.get("nInserted", 0)
                logger.warning("BulkWriteError; inserted %d docs.", n)
        else:
            logger.info("No new documents to insert.")

        # Normalize chapter and section fields for ALL existing documents in the database
        logger.info("Normalizing chapter and section fields for all existing documents...")
        normalized_count = 0
        for existing_doc in coll.find({}):
            try:
                update_fields = {}
                needs_update = False
                
                # Normalize chapter if present
                if "chapter" in existing_doc and existing_doc["chapter"]:
                    normalized_chapter = normalize_chapter(existing_doc["chapter"])
                    if normalized_chapter != existing_doc["chapter"]:
                        update_fields["chapter"] = normalized_chapter
                        needs_update = True
                
                # Normalize section if present
                if "section" in existing_doc and existing_doc["section"]:
                    normalized_section = normalize_section(existing_doc["section"])
                    if normalized_section != existing_doc["section"]:
                        update_fields["section"] = normalized_section
                        needs_update = True
                
                if needs_update:
                    result = coll.update_one(
                        {"_id": existing_doc["_id"]},
                        {"$set": update_fields}
                    )
                    if result.modified_count > 0:
                        normalized_count += 1
            except Exception as e:
                logger.warning(f"Failed to normalize document '{existing_doc.get('title', 'unknown')}': {e}")
        
        if normalized_count > 0:
            logger.info("Normalized chapter/section fields for %d existing documents.", normalized_count)
        else:
            logger.info("All documents already have normalized chapter/section fields.")

        # Recreate indexes
        coll.create_index("title", unique=True)
        coll.create_index([("article", 1), ("chapter", 1), ("section", 1)])
        # Unique across *clause titles* (multikey). Assumes each clause title is unique across the US Code.
        coll.create_index("clauses.title", unique=True)
        logger.info("Indexes created: title (unique), (article,chapter,section), clauses.title (unique)")

    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise
    finally:
        if client:
            client.close()
            logger.info("Mongo connection closed.")

if __name__ == "__main__":
    ingest()

