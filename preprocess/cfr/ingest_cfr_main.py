# ingest_cfr.py
import os
import sys
import json
import logging
import argparse
import datetime
from typing import Any, Dict, List
from pathlib import Path
from pymongo import MongoClient, WriteConcern
from pymongo.errors import BulkWriteError
import bson

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
    parser = argparse.ArgumentParser(description="Ingest Code of Federal Regulations main document")
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
logger = logging.getLogger("ingest_cfr")
BASE_DIR = Path(__file__).resolve().parents[2]
CFR_DOCUMENT_PATH = str(BASE_DIR / "Data/Knowledge/code_of_federal_regulations.json")
CFR_CONF = COLLECTION.get("CFR_SET")
if not CFR_CONF:
    raise ValueError("CFR_SET not found in COLLECTION config. Please add CFR_SET configuration to config.py")
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
        logger.error("Failed to read JSON: %s", e)
        return None

def clean_title(title: str) -> str:
    """Remove section numbers like '§ 1.1' or '(section)' from title."""
    import re
    if not title:
        return title
    # Remove patterns like "§ 1.1", "§1.1", "(section)", "(Section)", etc.
    # Pattern matches: § followed by optional space and section number, or (section) in parentheses
    title = re.sub(r'§\s*\d+\.?\d*\s*', '', title)  # Remove "§ 1.1" or "§1.1"
    title = re.sub(r'\(\s*section\s*\)', '', title, flags=re.IGNORECASE)  # Remove "(section)" or "(Section)"
    title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
    return title.strip()

def extract_article_number(article: str) -> str:
    """Extract number from 'Title X' -> 'X'"""
    import re
    if not article:
        return ""
    # Extract number from "Title 50" -> "50"
    match = re.search(r'Title\s+(\d+)', str(article), re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any number
    match = re.search(r'(\d+)', str(article))
    return match.group(1) if match else ""

def extract_part_number(part: str) -> str:
    """Extract number from 'Part X' -> 'X'"""
    import re
    if not part:
        return ""
    # Extract number from "Part 260" -> "260"
    match = re.search(r'Part\s+(\d+)', str(part), re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any number
    match = re.search(r'(\d+)', str(part))
    return match.group(1) if match else ""

def extract_section_number(section: str) -> str:
    """Extract section number from 'Section X.Y' -> 'X.Y'"""
    import re
    if not section:
        return ""
    # Extract section number from "Section 260.20" -> "260.20"
    match = re.search(r'Section\s+(\d+\.?\d*)', str(section), re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any number pattern
    match = re.search(r'(\d+\.\d+)', str(section))
    if match:
        return match.group(1)
    return ""

def extract_chapter_from_section(section: str, part_num: str) -> str:
    """Extract chapter from section number (e.g., 'Section 260.20' with part '260' -> '2')"""
    import re
    if not section or not part_num:
        return ""
    # Extract section number first
    section_num = extract_section_number(section)
    if not section_num:
        return ""
    
    # If section is "260.20" and part is "260", extract "2" (first digit after part)
    # Pattern: part_num.chapter.section -> extract chapter
    if section_num.startswith(part_num + "."):
        remaining = section_num[len(part_num) + 1:]  # Get "20" from "260.20"
        # Extract first digit(s) as chapter
        match = re.search(r'^(\d+)', remaining)
        if match:
            return match.group(1)
    
    return ""

def extract_chapter_number(chapter: str) -> str:
    """Extract chapter number from chapter text (e.g., 'Chapter 1 - Title' -> '1')."""
    import re
    if not chapter:
        return ""
    # Extract number from patterns like "Chapter 1", "Ch. 1", "1", etc.
    match = re.search(r'(\d+)', str(chapter))
    return match.group(1) if match else ""

def normalize_to_hierarchy(entry_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single entry (not yet grouped).
    Returns normalized entry with extracted numbers and sections.
    Handles new structure with 'subchapter' and 'sections' array.
    """
    import re
    article_text = entry_obj.get("article", "")
    part_text = entry_obj.get("part", "")
    chapter_text = entry_obj.get("chapter", "")
    subchapter_text = entry_obj.get("subchapter", "")
    sections_array = entry_obj.get("sections", [])  # New structure: sections array
    clauses = entry_obj.get("clauses")  # Old structure: clauses (for backward compatibility)
    section_text = entry_obj.get("section", "")
    original_title = entry_obj.get("title", "")
    text = entry_obj.get("text", "")

    # Extract numbers
    article_num = extract_article_number(article_text)
    part_num = extract_part_number(part_text)
    section_num = extract_section_number(section_text)
    
    # Extract chapter: try from section number first, then from chapter text
    chapter_num = extract_chapter_from_section(section_text, part_num)
    if not chapter_num:
        chapter_num = extract_chapter_number(chapter_text)

    # Convert sections array or clauses to normalized sections
    sections = []
    
    # New structure: sections array
    if sections_array and isinstance(sections_array, list):
        for idx, s in enumerate(sections_array):
            section_title = s.get("title", "")
            section_text_content = s.get("text", "")
            section_number = s.get("section", "")
            
            # Clean title if present
            if section_title:
                section_title = clean_title(section_title)
            
            # Only add section if we have at least title or text
            if section_text_content or section_title:
                sections.append({
                    "number": idx + 1,  # Use index + 1 as number
                    "title": section_title,
                    "text": section_text_content,
                })
    # Old structure: clauses (for backward compatibility)
    elif clauses and isinstance(clauses, list):
        for c in clauses:
            clause_title = c.get("title", "")
            clause_text = c.get("text", "")
            
            # Clean title if present
            if clause_title:
                clause_title = clean_title(clause_title)
            
            # Section title: only use clause title (never use text as title)
            section_title = clause_title if clause_title else ""
            # Section text: only use clause text (never use title as text)
            section_text_content = clause_text if clause_text else ""
            
            # Only add section if we have at least title or text
            if section_text_content or section_title:
                sections.append({
                    "number": c.get("number"),
                    "title": section_title,
                    "text": section_text_content,
                })
    elif text:
        # If flat structure with text, use empty title and text as content
        sections.append({
            "number": 1,
            "title": "",
            "text": text,
        })
    elif original_title:
        # If only title, use it as title and empty text
        clean_t = clean_title(original_title)
        sections.append({
            "number": 1,
            "title": clean_t,
            "text": "",
        })

    # Use original title as document title (never use section text)
    if original_title:
        doc_title = clean_title(original_title)
    else:
        # Fallback: construct title from metadata
        title_parts = []
        if article_num:
            title_parts.append(f"Title {article_num}")
        if part_num:
            title_parts.append(f"Part {part_num}")
        if chapter_num:
            title_parts.append(f"Chapter {chapter_num}")
        if subchapter_text:
            title_parts.append(subchapter_text)
        if section_num:
            title_parts.append(f"Section {section_num}")
        doc_title = " - ".join(title_parts) if title_parts else ""

    return {
        "article": article_num,
        "part": part_num,
        "chapter": chapter_num if chapter_num else "",
        "subchapter": subchapter_text if subchapter_text else "",  # Include subchapter
        "section": section_text,  # Keep full section identifier
        "title": doc_title,
        "sections": sections  # Sections array with number, title, and text
    }

def group_and_aggregate_sections(normalized_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group entries by (article, part, chapter, subchapter) and aggregate sections.
    Returns one document per group with all sections sorted by number.
    """
    grouped = {}
    
    for entry in normalized_entries:
        # Create grouping key including subchapter
        key = (entry.get("article", ""), entry.get("part", ""), entry.get("chapter", ""), entry.get("subchapter", ""))
        
        if key not in grouped:
            # Initialize group with first entry's metadata
            # Use original title from first entry, or construct from metadata
            title = entry.get("title", "")
            if not title:
                # Construct title from metadata
                title_parts = []
                if entry.get("article"):
                    title_parts.append(f"Title {entry['article']}")
                if entry.get("part"):
                    title_parts.append(f"Part {entry['part']}")
                if entry.get("chapter"):
                    title_parts.append(f"Chapter {entry['chapter']}")
                if entry.get("subchapter"):
                    title_parts.append(entry['subchapter'])
                title = " - ".join(title_parts) if title_parts else ""
            
            grouped[key] = {
                "article": entry.get("article", ""),
                "part": entry.get("part", ""),
                "chapter": entry.get("chapter", ""),
                "subchapter": entry.get("subchapter", ""),
                "title": title,
                "sections": []
            }
        
        # Add sections from this entry to the group
        entry_sections = entry.get("sections", [])
        grouped[key]["sections"].extend(entry_sections)
    
    # Sort sections by number for each group
    result = []
    for key, doc in grouped.items():
        # Sort sections by number (handle both int and string numbers)
        doc["sections"].sort(key=lambda x: (
            float(x.get("number", 0)) if isinstance(x.get("number"), (int, float)) 
            else float(str(x.get("number", "0")).split(".")[0]) if str(x.get("number", "0")).replace(".", "").isdigit()
            else 0
        ))
        result.append(doc)
    
    return result

def generate_embeddings_for_docs_batch(embedder, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for multiple documents and their sections using batch API"""
    # Collect all texts to embed
    doc_texts = []
    section_texts = []
    doc_indices = []  # Track which doc each text belongs to
    section_indices = []  # Track which section each text belongs to (doc_idx, section_idx)
    
    for doc_idx, doc in enumerate(docs):
        # Document-level text
        doc_text_parts = []
        if doc.get("title"):
            doc_text_parts.append(doc["title"])
        if doc.get("article"):
            doc_text_parts.append(f"Article {doc['article']}")
        if doc.get("part"):
            doc_text_parts.append(f"Part {doc['part']}")
        if doc.get("chapter"):
            doc_text_parts.append(f"Chapter {doc['chapter']}")
        if doc.get("section"):
            doc_text_parts.append(doc["section"])
        
        doc_text = " ".join(doc_text_parts)
        if doc_text:
            doc_texts.append(doc_text)
            doc_indices.append(doc_idx)
        
        # Section-level texts
        sections = doc.get("sections", [])
        for section_idx, section in enumerate(sections):
            # Combine title and text for section embedding
            section_title = section.get("title", "")
            section_text = section.get("text", "")
            section_full_text = f"{section_title} {section_text}".strip()
            if section_full_text:
                section_texts.append(section_full_text)
                section_indices.append((doc_idx, section_idx))
    
    # Generate document embeddings in batch
    if doc_texts:
        try:
            doc_embeddings = embedder.get_openai_embeddings_batch(doc_texts, batch_size=100)
            for i, doc_idx in enumerate(doc_indices):
                if i < len(doc_embeddings) and doc_embeddings[i] is not None:
                    docs[doc_idx]["embedding"] = doc_embeddings[i].tolist() if hasattr(doc_embeddings[i], 'tolist') else list(doc_embeddings[i])
        except Exception as e:
            logger.warning(f"Failed to generate document embeddings in batch: {e}")
    
    # Generate section embeddings in batch
    if section_texts:
        try:
            section_embeddings = embedder.get_openai_embeddings_batch(section_texts, batch_size=100)
            for i, (doc_idx, section_idx) in enumerate(section_indices):
                if i < len(section_embeddings) and section_embeddings[i] is not None:
                    if "sections" not in docs[doc_idx]:
                        docs[doc_idx]["sections"] = []
                    while len(docs[doc_idx]["sections"]) <= section_idx:
                        docs[doc_idx]["sections"].append({})
                    docs[doc_idx]["sections"][section_idx]["embedding"] = section_embeddings[i].tolist() if hasattr(section_embeddings[i], 'tolist') else list(section_embeddings[i])
        except Exception as e:
            logger.warning(f"Failed to generate section embeddings in batch: {e}")
    
    return docs

def ingest():
    client = None
    original_voyage_key = None
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

        # Load CFR JSON
        data = load_json(CFR_DOCUMENT_PATH)
        if not data:
            return
        cfr_data = data.get("data", {}).get("code_of_federal_regulations", {})
        # Try both "titles" and "regulations" keys
        items = cfr_data.get("titles") or cfr_data.get("regulations", [])
        if not items:
            logger.warning("No 'titles' or 'regulations' found in JSON.")
            return

        # Normalize all entries
        normalized_entries: List[Dict[str, Any]] = [normalize_to_hierarchy(obj) for obj in items]
        logger.info("Normalized %d entries from JSON.", len(normalized_entries))
        
        # Group entries by (article, part, chapter) and aggregate sections
        docs: List[Dict[str, Any]] = group_and_aggregate_sections(normalized_entries)
        logger.info("Grouped into %d documents (aggregated sections).", len(docs))

        # Generate embeddings if requested
        if args and args.with_embeddings:
            # Use custom Voyage API key for CFR embeddings
            # Set it BEFORE importing ai_service so it picks up the new value
            CUSTOM_VOYAGE_API_KEY = "pa-U-myY6v2T-rzpVT3zbfUiq5Gn9QpSSX_sjoUgWQSUsr"
            if original_voyage_key is None:
                original_voyage_key = os.getenv("VOYAGE_API_KEY")
            os.environ["VOYAGE_API_KEY"] = CUSTOM_VOYAGE_API_KEY
            # Also update the config module's VOYAGE_API_KEY before importing ai_service
            config_module.VOYAGE_API_KEY = CUSTOM_VOYAGE_API_KEY
            logger.info("Using custom Voyage API key for CFR embeddings")
            
            from backend.services.rag.rag_dependencies import ai_service
            # Reload the module to pick up the updated VOYAGE_API_KEY
            import importlib
            importlib.reload(ai_service)
            from backend.services.rag.rag_dependencies.ai_service import LLM
            
            use_batch = args.batch_embeddings if args else False
            mode_str = "batch" if use_batch else "individual"
            logger.info(f"Generating embeddings using Voyage-3-large (1024 dimensions) in {mode_str} mode...")
            embedder = LLM(config=CFR_CONF)
            
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
                    """Generate embeddings for a single document and its sections"""
                    # Document-level embedding
                    doc_text_parts = []
                    if doc.get("title"):
                        doc_text_parts.append(doc["title"])
                    if doc.get("article"):
                        doc_text_parts.append(f"Article {doc['article']}")
                    if doc.get("part"):
                        doc_text_parts.append(f"Part {doc['part']}")
                    if doc.get("chapter"):
                        doc_text_parts.append(f"Chapter {doc['chapter']}")
                    if doc.get("section"):
                        doc_text_parts.append(doc["section"])
                    
                    doc_text = " ".join(doc_text_parts)
                    if doc_text:
                        try:
                            doc_emb = embedder.get_openai_embedding(doc_text)
                            if doc_emb is not None:
                                doc["embedding"] = doc_emb.tolist() if hasattr(doc_emb, 'tolist') else list(doc_emb)
                        except Exception as e:
                            logger.warning(f"Failed to generate document embedding: {e}")
                    
                    # Section-level embeddings
                    sections = doc.get("sections", [])
                    for section in sections:
                        # Combine title and text for section embedding
                        section_title = section.get("title", "")
                        section_text = section.get("text", "")
                        section_full_text = f"{section_title} {section_text}".strip()
                        if section_full_text:
                            try:
                                section_emb = embedder.get_openai_embedding(section_full_text)
                                if section_emb is not None:
                                    section["embedding"] = section_emb.tolist() if hasattr(section_emb, 'tolist') else list(section_emb)
                            except Exception as e:
                                logger.warning(f"Failed to generate section embedding: {e}")
                    
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

        # Deduplicate by top-level title (section/article title)
        existing_titles = {d["title"] for d in coll.find({}, {"title": 1}) if "title" in d}
        new_docs = [d for d in docs if d.get("title") and d["title"] not in existing_titles]
        logger.info("Prepared %d new docs (skipped %d duplicates by title).", len(new_docs), len(docs) - len(new_docs))

        # Split large documents that exceed MongoDB's 16MB BSON limit
        def split_large_document(doc: Dict[str, Any], max_size_bytes: int = 15000000) -> List[Dict[str, Any]]:
            """Split a document if it's too large, keeping sections together."""
            import bson
            try:
                # Estimate document size
                doc_size = len(bson.encode(doc))
                if doc_size <= max_size_bytes:
                    return [doc]
                
                # Document is too large, split by sections
                sections = doc.get("sections", [])
                if not sections:
                    return [doc]  # Can't split if no sections
                
                # Split sections into chunks
                split_docs = []
                current_sections = []
                current_size = 0
                base_doc = {k: v for k, v in doc.items() if k != "sections"}
                
                for section in sections:
                    section_size = len(bson.encode({"sections": [section]}))
                    if current_size + section_size > max_size_bytes and current_sections:
                        # Create a document with current sections
                        split_doc = {**base_doc, "sections": current_sections.copy()}
                        split_doc["title"] = f"{base_doc.get('title', '')} (Part {len(split_docs) + 1})"
                        split_docs.append(split_doc)
                        current_sections = []
                        current_size = 0
                    
                    current_sections.append(section)
                    current_size += section_size
                
                # Add remaining sections
                if current_sections:
                    split_doc = {**base_doc, "sections": current_sections}
                    if len(split_docs) > 0:
                        split_doc["title"] = f"{base_doc.get('title', '')} (Part {len(split_docs) + 1})"
                    split_docs.append(split_doc)
                
                return split_docs if split_docs else [doc]
            except Exception as e:
                logger.warning(f"Error splitting document: {e}, inserting as-is")
                return [doc]
        
        # Split large documents
        final_docs = []
        for doc in new_docs:
            split_docs = split_large_document(doc)
            final_docs.extend(split_docs)
        
        logger.info("After splitting large documents: %d documents to insert.", len(final_docs))

        if final_docs:
            # Add timestamps to all documents
            now = datetime.datetime.utcnow()
            for doc in final_docs:
                doc["created_at"] = now
                doc["updated_at"] = now
            
            # Insert in smaller batches to avoid memory issues
            batch_size = 100
            total_inserted = 0
            for i in range(0, len(final_docs), batch_size):
                batch = final_docs[i:i + batch_size]
                try:
                    res = coll.insert_many(batch, ordered=False)
                    total_inserted += len(res.inserted_ids)
                    logger.info("Inserted batch %d-%d/%d documents (%d total inserted).", 
                              i + 1, min(i + batch_size, len(final_docs)), len(final_docs), total_inserted)
                except BulkWriteError as bwe:
                    n = bwe.details.get("nInserted", 0)
                    total_inserted += n
                    logger.warning("BulkWriteError in batch %d-%d; inserted %d docs (%d total inserted).", 
                                 i + 1, min(i + batch_size, len(final_docs)), n, total_inserted)
                except Exception as e:
                    logger.error(f"Error inserting batch {i + 1}-{min(i + batch_size, len(final_docs))}: {e}")
            
            logger.info("Total inserted: %d documents.", total_inserted)
        else:
            logger.info("No new documents to insert.")

        # Recreate indexes
        coll.create_index("title", unique=True)
        coll.create_index([("article", 1), ("part", 1), ("chapter", 1)])
        # Unique across *section titles* (multikey). Assumes each section title is unique across the CFR.
        coll.create_index("sections.title", unique=True)
        logger.info("Indexes created: title (unique), (article,part,chapter), sections.title (unique)")

    except Exception as e:
        logger.error("Ingest error: %s", e)
        raise
    finally:
        # Restore original Voyage API key if we modified it
        try:
            if original_voyage_key is not None:
                if original_voyage_key:
                    os.environ["VOYAGE_API_KEY"] = original_voyage_key
                    config_module.VOYAGE_API_KEY = original_voyage_key
                else:
                    os.environ.pop("VOYAGE_API_KEY", None)
                logger.info("Restored original Voyage API key")
        except Exception as e:
            logger.debug(f"Could not restore Voyage API key: {e}")
        
        if client:
            client.close()
            logger.info("Mongo connection closed.")

if __name__ == "__main__":
    ingest()

