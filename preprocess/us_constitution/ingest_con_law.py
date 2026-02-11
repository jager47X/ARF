# ingest_con_law.py
import os
import sys
import json
import logging
import argparse
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
    parser = argparse.ArgumentParser(description="Ingest US Constitution main document")
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
    parser.add_argument(
        "--batch-embeddings",
        action="store_true",
        default=False,
        help="Use batch embedding generation (slower but more efficient for large datasets). Default: individual embeddings (faster per item)"
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
EMBEDDING_DIMENSIONS = config_module.EMBEDDING_DIMENSIONS
from backend.services.rag.rag_dependencies.ai_service import LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_con_law")
BASE_DIR = Path(__file__).resolve().parents[2]
USCON_DOCUMENT_PATH = str(BASE_DIR / "Data/Knowledge/us_con_law.json")
USC_CONF = COLLECTION["US_CONSTITUTION_SET"]
COLL_NAME = USC_CONF["main_collection_name"]  # "us_constitution"
DB_NAME: str = USC_CONF["db_name"]  # "public"

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

def normalize_to_hierarchy(article_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each entry has `clauses: [...]`.
    If `text` (flat) is present and `clauses` missing, wrap it as a single clause.
    """
    article = article_obj.get("article", "")
    section = article_obj.get("section", "")
    title = article_obj.get("title", "")
    clauses = article_obj.get("clauses")

    if clauses and isinstance(clauses, list):
        # Already hierarchical — just ensure each clause has number/title/text
        clean_clauses = []
        for c in clauses:
            clean_clauses.append({
                "number": c.get("number"),
                "title": c.get("title") or "",
                "text": c.get("text") or "",
                # keywords will be added later by find_and_ingest_alias.py
            })
        return {"article": article, "section": section, "title": title, "clauses": clean_clauses}

    # Flat -> wrap
    text = article_obj.get("text", "")
    if not text:
        # Keep empty clause list; can be filled later if needed
        return {"article": article, "section": section, "title": title, "clauses": []}

    # Use clause number 1, clause title identical to section/article title for consistency
    return {
        "article": article,
        "section": section,
        "title": title,
        "clauses": [{
            "number": 1,
            "title": title,
            "text": text,
        }]
    }

def generate_embeddings_for_docs_batch(embedder: LLM, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        client = MongoClient(MONGO_URI)
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
            for idx in list(coll.index_information().keys()):
                if idx != "_id_":
                    coll.drop_index(idx)
            logger.info("Dropped non-_id indexes.")

        # Load Constitution JSON
        data = load_json(USCON_DOCUMENT_PATH)
        if not data:
            return
        items = data.get("data", {}).get("constitution", {}).get("articles", [])
        if not items:
            logger.warning("No 'articles' found in JSON.")
            return

        docs: List[Dict[str, Any]] = [normalize_to_hierarchy(obj) for obj in items]
        logger.info("Loaded %d documents from JSON.", len(docs))

        # Generate embeddings if requested
        if args and args.with_embeddings:
            use_batch = args.batch_embeddings if args else False
            mode_str = "batch" if use_batch else "individual"
            logger.info(f"Generating embeddings using Voyage-3-large (1024 dimensions) in {mode_str} mode...")
            embedder = LLM(config=USC_CONF)
            
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
                    """Generate embeddings for a single document and its clauses"""
                    # Document-level embedding
                    doc_text_parts = []
                    if doc.get("title"):
                        doc_text_parts.append(doc["title"])
                    if doc.get("article"):
                        doc_text_parts.append(f"Article {doc['article']}")
                    if doc.get("section"):
                        doc_text_parts.append(f"Section {doc['section']}")
                    
                    doc_text = " ".join(doc_text_parts)
                    if doc_text:
                        try:
                            doc_emb = embedder.get_openai_embedding(doc_text)
                            if doc_emb is not None:
                                doc["embedding"] = doc_emb.tolist() if hasattr(doc_emb, 'tolist') else list(doc_emb)
                        except Exception as e:
                            logger.warning(f"Failed to generate document embedding for '{doc.get('title', 'N/A')}': {e}")
                    
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
                                logger.warning(f"Failed to generate clause embedding for '{clause.get('title', 'N/A')}': {e}")
                    
                    return doc
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(generate_embeddings_for_doc, embedder, doc): i for i, doc in enumerate(docs)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            docs[idx] = future.result()
                            if (idx + 1) % 10 == 0:
                                logger.info("Generated embeddings for %d/%d documents", idx + 1, len(docs))
                        except Exception as e:
                            logger.error(f"Error processing document {idx}: {e}")
            
            logger.info("Completed embedding generation for all documents.")

        # Deduplicate by top-level title (section/article title) - only if not from-scratch
        if args and not args.from_scratch:
            existing_titles = {d.get("title") for d in coll.find({}, {"title": 1}) if d.get("title")}
            new_docs = [d for d in docs if d.get("title") and d["title"] not in existing_titles]
            logger.info("Prepared %d new docs (skipped %d duplicates by title).", len(new_docs), len(docs) - len(new_docs))
        else:
            new_docs = docs
            logger.info("Prepared %d documents for insertion.", len(new_docs))

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
        coll.create_index([("article", 1), ("section", 1)])
        # Unique across *clause titles* (multikey). Assumes each clause title is unique across the Constitution.
        coll.create_index("clauses.title", unique=True)
        logger.info("Indexes created: title (unique), (article,section), clauses.title (unique)")
        
        # Add aliases as keywords with embeddings (only if embeddings were generated)
        if args and args.with_embeddings:
            use_batch = args.batch_embeddings if args else False
            logger.info(f"Adding aliases as keywords with embeddings (mode: {'batch' if use_batch else 'individual'})...")
            # Re-initialize embedder if not already created
            if 'embedder' not in locals():
                embedder = LLM(config=USC_CONF)
            add_aliases_as_keywords(coll, embedder, use_batch=use_batch)
            logger.info("Completed adding aliases as keywords.")

    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise
    finally:
        if client:
            client.close()
            logger.info("Mongo connection closed.")

def add_aliases_as_keywords(coll, embedder: LLM, use_batch: bool = False):
    """
    Add aliases from alias_map as keywords with embeddings to each constitution document.
    Keywords are stored as: [{keyword: str, embedding: List[float]}, ...]
    
    Args:
        coll: MongoDB collection
        embedder: LLM instance for generating embeddings
        use_batch: If True, use batch embedding API (slower but more efficient). If False, use individual embeddings (faster per item).
    """
    # Import alias_map from ingest_alias_us_con_law module
    try:
        from services.rag.preprocess.ingest_alias_us_con_law import alias_map
    except ImportError:
        # Fallback: try to import from the file directly
        import importlib.util
        alias_map_path = Path(__file__).parent / "ingest_alias_us_con_law.py"
        if alias_map_path.exists():
            spec = importlib.util.spec_from_file_location("ingest_alias_us_con_law", alias_map_path)
            if spec and spec.loader:
                alias_map_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(alias_map_module)
                alias_map = alias_map_module.alias_map
            else:
                logger.error("Could not load alias_map. Skipping keyword addition.")
                return
        else:
            logger.error("Could not find ingest_alias_us_con_law.py. Skipping keyword addition.")
            return
    
    logger.info(f"Loaded alias_map with {len(alias_map)} entries")
    logger.info(f"Alias_map keys (first 10): {list(alias_map.keys())[:10]}")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Collect all documents and their titles
    docs = list(coll.find({}))
    total_docs = len(docs)
    logger.info(f"Found {total_docs} documents in collection")
    
    # Log document titles for debugging
    doc_titles = [doc.get("title", "").strip() for doc in docs if doc.get("title")]
    logger.info(f"Document titles (first 10): {doc_titles[:10]}")
    
    # Create a case-insensitive lookup map for alias_map
    alias_map_lower = {k.lower(): (k, v) for k, v in alias_map.items()}
    
    # Check which titles have aliases
    titles_with_aliases = []
    titles_without_aliases = []
    for doc in docs:
        title = doc.get("title", "").strip()
        if not title:
            continue
        # Try exact match first, then case-insensitive
        if title in alias_map:
            titles_with_aliases.append(title)
        elif title.lower() in alias_map_lower:
            titles_with_aliases.append(title)
        else:
            titles_without_aliases.append(title)
    
    logger.info(f"Found {len(titles_with_aliases)} documents with aliases in alias_map")
    logger.info(f"Found {len(titles_without_aliases)} documents without aliases")
    if titles_without_aliases:
        logger.debug(f"Documents without aliases (first 10): {titles_without_aliases[:10]}")
    
    def process_document_keywords(doc):
        """Process a single document and add its aliases as keywords"""
        title = doc.get("title", "").strip()
        if not title:
            return None
        
        # Get aliases for this title from alias_map (try exact match, then case-insensitive)
        aliases = alias_map.get(title, [])
        if not aliases and title.lower() in alias_map_lower:
            # Case-insensitive fallback
            original_key, aliases = alias_map_lower[title.lower()]
            logger.debug(f"Matched '{title}' to alias_map key '{original_key}' (case-insensitive)")
        
        if not aliases:
            return doc  # No aliases for this document
        
        # Get existing keywords
        existing_keywords = doc.get("keywords", [])
        existing_keyword_texts = {kw.get("keyword", "").lower(): kw for kw in existing_keywords if isinstance(kw, dict)}
        
        # Collect aliases that need embeddings
        aliases_to_embed = []
        for alias in aliases:
            alias_lower = alias.lower().strip()
            if not alias_lower or alias_lower == title.lower():
                continue  # Skip empty or duplicate of title
            
            # Check if keyword already exists
            if alias_lower in existing_keyword_texts:
                continue
            
            aliases_to_embed.append(alias)
        
        # Generate embeddings (batch or individual based on flag)
        new_keywords = []
        if aliases_to_embed:
            if use_batch:
                # Batch mode: slower but more efficient
                try:
                    embeddings = embedder.get_openai_embeddings_batch(aliases_to_embed, batch_size=100)
                    for i, alias in enumerate(aliases_to_embed):
                        if i < len(embeddings) and embeddings[i] is not None:
                            new_keywords.append({
                                "keyword": alias,
                                "embedding": embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i])
                            })
                except Exception as e:
                    logger.warning(f"Failed to generate batch embeddings for keywords in '{title}': {e}. Trying individual embeddings...")
                    # Fallback to individual embeddings
                    for alias in aliases_to_embed:
                        try:
                            emb = embedder.get_openai_embedding(alias)
                            if emb is not None:
                                new_keywords.append({
                                    "keyword": alias,
                                    "embedding": emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                                })
                        except Exception as e2:
                            logger.error(f"Failed to generate embedding for keyword '{alias}' in '{title}': {e2}")
            else:
                # Individual mode: faster per item
                for alias in aliases_to_embed:
                    try:
                        emb = embedder.get_openai_embedding(alias)
                        if emb is not None:
                            new_keywords.append({
                                "keyword": alias,
                                "embedding": emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                            })
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for keyword '{alias}' in '{title}': {e}")
        
        # Update document with new keywords
        if new_keywords:
            all_keywords = existing_keywords + new_keywords
            coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"keywords": all_keywords}}
            )
            logger.info(f"Added {len(new_keywords)} keywords to '{title}' (total aliases: {len(aliases)})")
        elif aliases:
            logger.debug(f"No new keywords added to '{title}' (all {len(aliases)} aliases already exist or were skipped)")
        
        return doc
    
    # Process all documents
    logger.info("Processing documents to add aliases as keywords...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_document_keywords, doc): i for i, doc in enumerate(docs)}
        processed = 0
        updated = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                processed += 1
                if result and result != docs[idx]:  # Document was updated
                    updated += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_docs} documents for keywords ({updated} updated)")
            except Exception as e:
                logger.error(f"Error processing document {idx} (title: {docs[idx].get('title', 'N/A')}): {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.info(f"Completed processing {processed} documents for keywords ({updated} documents updated)")

if __name__ == "__main__":
    ingest()
