# add_cfr_embeddings.py
"""
Add embeddings for Code of Federal Regulations (CFR) documents in production MongoDB.
Processes documents in parallel with rate limiting to maintain exactly 1900 requests per minute.
Tracks requests in a rolling 60-second window and waits if approaching the 2000 RPM limit.
Skips documents/sections that already have embeddings.
"""
import os
import sys
import logging
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pymongo import MongoClient, WriteConcern
from pymongo.errors import PyMongoError
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque

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
    
    # Load services
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
            
            # Load rag
            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
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
                            
                            # Load rag_dependencies
                            rag_deps_init = backend_dir / 'services' / 'rag' / 'rag_dependencies' / '__init__.py'
                            if rag_deps_init.exists():
                                spec = importlib.util.spec_from_file_location('backend.services.rag.rag_dependencies', rag_deps_init)
                                if spec and spec.loader:
                                    rag_deps_mod = importlib.util.module_from_spec(spec)
                                    sys.modules['backend.services.rag.rag_dependencies'] = rag_deps_mod
                                    spec.loader.exec_module(rag_deps_mod)
                                    setattr(rag_mod, 'rag_dependencies', rag_deps_mod)
                                    sys.modules['services.rag.rag_dependencies'] = rag_deps_mod

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Add embeddings for CFR documents in production MongoDB")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (.env.production)"
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=1900,
        help="Maximum requests per minute (default: 1900, exactly under 2000 limit)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of documents to process before saving (default: 1, process one by one)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't update MongoDB, just report what would be done"
    )
    return parser.parse_args()

# Load environment based on args
args = parse_args() if __name__ == "__main__" else None
env_override = None
if args:
    if args.production:
        env_override = "production"
    else:
        # Default to production if not specified
        env_override = "production"

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
logger = logging.getLogger("add_cfr_embeddings")

CFR_CONF = COLLECTION.get("CFR_SET")
if not CFR_CONF:
    raise ValueError("CFR_SET not found in COLLECTION config. Please add CFR_SET configuration to config.py")

DB_NAME: str = CFR_CONF["db_name"]
COLL_NAME: str = CFR_CONF["main_collection_name"]

def compose_document_text(doc: Dict[str, Any]) -> str:
    """Compose text for document-level embedding."""
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("article"):
        parts.append(f"Title {doc['article']}")
    if doc.get("part"):
        parts.append(f"Part {doc['part']}")
    if doc.get("chapter"):
        parts.append(f"Chapter {doc['chapter']}")
    if doc.get("subchapter"):
        parts.append(doc["subchapter"])
    return " ".join(parts) if parts else ""

def compose_section_text(section: Dict[str, Any]) -> str:
    """Compose text for section-level embedding."""
    parts = []
    if section.get("title"):
        parts.append(section["title"])
    if section.get("text"):
        parts.append(section["text"])
    return " ".join(parts).strip()

def has_document_embedding(doc: Dict[str, Any]) -> bool:
    """Check if document has an embedding."""
    return doc.get("embedding") is not None and len(doc.get("embedding", [])) > 0

def has_section_embedding(section: Dict[str, Any]) -> bool:
    """Check if section has an embedding."""
    return section.get("embedding") is not None and len(section.get("embedding", [])) > 0

class WindowRateLimiter:
    """Rate limiter that tracks requests in a rolling 60-second window."""
    def __init__(self, max_requests_per_minute: int = 1900):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = Lock()
        self.window_seconds = 60.0
    
    def get_current_count(self) -> int:
        """Get number of requests in the current 60-second window."""
        with self.lock:
            now = time.time()
            # Remove requests older than 60 seconds
            while self.request_times and self.request_times[0] < now - self.window_seconds:
                self.request_times.popleft()
            return len(self.request_times)
    
    def can_make_request(self, num_requests: int = 1) -> Tuple[bool, float]:
        """
        Check if we can make a request without exceeding the limit.
        Returns (can_make, wait_time_if_needed)
        """
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.request_times and self.request_times[0] < now - self.window_seconds:
                self.request_times.popleft()
            
            current_count = len(self.request_times)
            
            # Check if adding these requests would exceed the limit
            if current_count + num_requests > self.max_requests:
                # Calculate wait time until oldest request expires
                if self.request_times:
                    oldest_time = self.request_times[0]
                    wait_time = (oldest_time + self.window_seconds) - now + 0.1  # Small buffer
                    return False, max(0, wait_time)
                else:
                    return True, 0.0
            
            return True, 0.0
    
    def record_request(self, num_requests: int = 1):
        """Record that requests were made."""
        with self.lock:
            now = time.time()
            for _ in range(num_requests):
                self.request_times.append(now)

def process_document(embedder, doc: Dict[str, Any], dry_run: bool = False,
                     embedding_times: List[float] = None,
                     doc_embedding_times: List[float] = None,
                     section_embedding_times: List[float] = None,
                     rate_limiter: Optional[WindowRateLimiter] = None) -> Tuple[Dict[str, Any], bool]:
    """
    Process a single document and add embeddings if missing.
    Returns updated document.
    """
    updated = False
    if embedding_times is None:
        embedding_times = []
    if doc_embedding_times is None:
        doc_embedding_times = []
    if section_embedding_times is None:
        section_embedding_times = []
    
    # Check and generate document-level embedding
    if not has_document_embedding(doc):
        doc_text = compose_document_text(doc)
        if doc_text:
            try:
                if not dry_run:
                    # Wait if needed to stay under rate limit
                    if rate_limiter:
                        can_make, wait_time = rate_limiter.can_make_request(1)
                        if not can_make:
                            time.sleep(wait_time)
                        rate_limiter.record_request(1)
                    
                    emb_start = time.time()
                    doc_emb = embedder.get_openai_embedding(doc_text)
                    emb_time = time.time() - emb_start
                    if embedding_times is not None:
                        embedding_times.append(emb_time)
                    if doc_embedding_times is not None:
                        doc_embedding_times.append(emb_time)
                    if doc_emb is not None:
                        doc["embedding"] = doc_emb.tolist() if hasattr(doc_emb, 'tolist') else list(doc_emb)
                        updated = True
                        logger.info(f"  Generated document embedding for: {doc.get('title', 'N/A')[:80]} ({emb_time:.3f}s)")
                else:
                    logger.info(f"  [DRY RUN] Would generate document embedding for: {doc.get('title', 'N/A')[:80]}")
            except Exception as e:
                error_str = str(e).lower()
                # Re-raise rate limit errors so they can be handled at a higher level
                if "rate limit" in error_str or "rpm" in error_str:
                    raise
                logger.warning(f"  Failed to generate document embedding: {e}")
        else:
            logger.warning(f"  Document has no text for embedding: {doc.get('title', 'N/A')[:80]}")
    else:
        logger.debug(f"  Document already has embedding: {doc.get('title', 'N/A')[:80]}")
    
    # Check and generate section-level embeddings
    sections = doc.get("sections", [])
    num_sections = len(sections)
    if num_sections > 0:
        # Count how many sections need embeddings
        sections_needing_embedding = sum(1 for s in sections if not has_section_embedding(s))
        if sections_needing_embedding > 0:
            logger.info(f"  Processing {sections_needing_embedding}/{num_sections} sections needing embeddings")
        
        for section_idx, section in enumerate(sections):
            if not has_section_embedding(section):
                section_text = compose_section_text(section)
                if section_text:
                    try:
                        if not dry_run:
                            # Wait if needed to stay under rate limit
                            if rate_limiter:
                                can_make, wait_time = rate_limiter.can_make_request(1)
                                if not can_make:
                                    time.sleep(wait_time)
                                rate_limiter.record_request(1)
                            
                            emb_start = time.time()
                            section_emb = embedder.get_openai_embedding(section_text)
                            emb_time = time.time() - emb_start
                            if embedding_times is not None:
                                embedding_times.append(emb_time)
                            if section_embedding_times is not None:
                                section_embedding_times.append(emb_time)
                            if section_emb is not None:
                                section["embedding"] = section_emb.tolist() if hasattr(section_emb, 'tolist') else list(section_emb)
                                updated = True
                                # Log progress every 10 sections for large documents
                                if num_sections > 10 and (section_idx + 1) % 10 == 0:
                                    if section_embedding_times and len(section_embedding_times) >= 10:
                                        avg_sec_time = sum(section_embedding_times[-10:]) / 10
                                        logger.info(f"    Generated {section_idx + 1}/{num_sections} section embeddings... (avg: {avg_sec_time:.3f}s)")
                                else:
                                    logger.debug(f"    Generated section {section_idx + 1} embedding ({emb_time:.3f}s)")
                        else:
                            logger.debug(f"    [DRY RUN] Would generate section {section_idx + 1} embedding")
                    except Exception as e:
                        error_str = str(e).lower()
                        # Re-raise rate limit errors so they can be handled at a higher level
                        if "rate limit" in error_str or "rpm" in error_str:
                            raise
                        logger.warning(f"    Failed to generate section {section_idx + 1} embedding: {e}")
                else:
                    logger.debug(f"    Section {section_idx + 1} has no text for embedding")
            else:
                logger.debug(f"    Section {section_idx + 1} already has embedding")
    
    return doc, updated

def main():
    """Main function to add embeddings for CFR documents."""
    client = None
    original_max_requests = None
    original_voyage_key = None
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
        coll = db.get_collection(COLL_NAME, write_concern=WriteConcern(w=0))
        logger.info(f"Connected to MongoDB collection: {DB_NAME}.{COLL_NAME}")
        
        if args and args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made to MongoDB")
        
        # Use custom Voyage API key for CFR embeddings
        # Set it BEFORE importing ai_service so it picks up the new value
        CUSTOM_VOYAGE_API_KEY = "pa-U-myY6v2T-rzpVT3zbfUiq5Gn9QpSSX_sjoUgWQSUsr"
        if original_voyage_key is None:
            original_voyage_key = os.getenv("VOYAGE_API_KEY")
            os.environ["VOYAGE_API_KEY"] = CUSTOM_VOYAGE_API_KEY
            # Also update the config module's VOYAGE_API_KEY before importing ai_service
            config_module.VOYAGE_API_KEY = CUSTOM_VOYAGE_API_KEY
            logger.info("Using custom Voyage API key for CFR embeddings")
        
        # Initialize embedder with rate limiting (import after setting API key)
        from backend.services.rag.rag_dependencies import ai_service
        # Reload the module to pick up the updated VOYAGE_API_KEY
        import importlib
        importlib.reload(ai_service)
        from backend.services.rag.rag_dependencies.ai_service import LLM, _voyage_rate_limiter
        
        # Set exactly 1900 RPM (under 2000 limit)
        max_rpm = args.max_requests_per_minute if args else 1900
        if max_rpm >= 2000:
            logger.warning(f"Rate limit {max_rpm} is at or above the API limit of 2000. Setting to 1900.")
            max_rpm = 1900
        logger.info(f"Using rate limit: {max_rpm} requests per minute (API limit: 2000)")
        
        # Create window-based rate limiter
        window_rate_limiter = WindowRateLimiter(max_requests_per_minute=max_rpm)
        
        # Temporarily update the global rate limiter (for the embedder's internal limiter)
        if original_max_requests is None:
            original_max_requests = _voyage_rate_limiter.max_requests
            _voyage_rate_limiter.request_times.clear()
            _voyage_rate_limiter.max_requests = max_rpm
            _voyage_rate_limiter.min_delay = 60.0 / max_rpm
            logger.info(f"Updated global rate limiter from {original_max_requests} to {max_rpm} req/min")
        
        embedder = LLM(config=CFR_CONF)
        
        # Get total count of documents
        total_docs = coll.count_documents({})
        logger.info(f"Total documents in collection: {total_docs}")
        
        # Get all documents that need embeddings
        logger.info("Scanning documents to find those needing embeddings...")
        docs_to_process = []
        cursor = coll.find({})
        for doc in cursor:
            needs_doc_embedding = not has_document_embedding(doc)
            needs_section_embeddings = False
            sections = doc.get("sections", [])
            for section in sections:
                if not has_section_embedding(section):
                    needs_section_embeddings = True
                    break
            if needs_doc_embedding or needs_section_embeddings:
                docs_to_process.append(doc)
        
        logger.info(f"Found {len(docs_to_process)} documents needing embeddings out of {total_docs} total")
        
        if not docs_to_process:
            logger.info("No documents need embeddings. All done!")
            return
        
        # Process documents in parallel
        processed_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        # Speed tracking (thread-safe lists)
        embedding_times = []  # Track individual embedding generation times
        doc_embedding_times = []
        section_embedding_times = []
        times_lock = Lock()
        
        max_workers = args.max_workers if args else 10
        logger.info(f"Using {max_workers} parallel workers with {max_rpm} RPM limit")
        
        start_time = time.time()
        
        # Process documents in parallel
        def process_single_document(doc_data: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], bool, Optional[str]]:
            """Process a single document and return results."""
            doc_idx, doc = doc_data
            doc_id = doc.get("_id")
            doc_title = doc.get("title", "N/A")[:80]
            
            try:
                updated_doc, was_updated = process_document(
                    embedder, doc,
                    dry_run=args.dry_run if args else False,
                    embedding_times=embedding_times,
                    doc_embedding_times=doc_embedding_times,
                    section_embedding_times=section_embedding_times,
                    rate_limiter=window_rate_limiter
                )
                return (doc_idx, updated_doc, was_updated, None)
            except Exception as e:
                return (doc_idx, doc, False, str(e))
        
        # Submit all documents for processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(process_single_document, (i, doc)): (i, doc)
                for i, doc in enumerate(docs_to_process)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_doc):
                doc_idx, original_doc = future_to_doc[future]
                processed_count += 1
                
                try:
                    doc_idx, updated_doc, was_updated, error_msg = future.result()
                    
                    if error_msg:
                        error_count += 1
                        logger.error(f"[{processed_count}] Error processing document: {error_msg}")
                        continue
                    
                    if was_updated:
                        updated_count += 1
                        if not (args and args.dry_run):
                            try:
                                coll.update_one(
                                    {"_id": original_doc.get("_id")},
                                    {"$set": updated_doc}
                                )
                                logger.info(f"[{processed_count}/{len(docs_to_process)}] Updated: {updated_doc.get('title', 'N/A')[:80]}")
                            except PyMongoError as e:
                                logger.error(f"[{processed_count}] Failed to update document: {e}")
                                error_count += 1
                        else:
                            logger.info(f"[{processed_count}/{len(docs_to_process)}] [DRY RUN] Would update: {updated_doc.get('title', 'N/A')[:80]}")
                    else:
                        skipped_count += 1
                    
                    # Progress reporting
                    if processed_count % 100 == 0 or processed_count == 1:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        
                        # Get current rate limiter status
                        current_rpm = window_rate_limiter.get_current_count()
                        
                        # Calculate speed metrics
                        with times_lock:
                            avg_doc_time = sum(doc_embedding_times) / len(doc_embedding_times) if doc_embedding_times else 0
                            avg_section_time = sum(section_embedding_times) / len(section_embedding_times) if section_embedding_times else 0
                            total_embeddings = len(doc_embedding_times) + len(section_embedding_times)
                            avg_embedding_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
                        
                        logger.info(f"Progress: {processed_count}/{len(docs_to_process)} documents processed "
                                  f"({updated_count} updated, {skipped_count} skipped, {error_count} errors)")
                        logger.info(f"  Speed: {rate:.2f} docs/sec | Current RPM: {current_rpm}/{max_rpm} | "
                                  f"Avg embedding: {avg_embedding_time:.3f}s")
                        if total_embeddings > 0:
                            embeddings_per_min = 60.0 / avg_embedding_time if avg_embedding_time > 0 else 0
                            logger.info(f"  Throughput: {embeddings_per_min:.1f} embeddings/minute")
                
                except Exception as e:
                    error_count += 1
                    logger.error(f"[{processed_count}] Error getting result: {e}", exc_info=True)
        
        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total documents processed: {processed_count}")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        if processed_count > 0:
            logger.info(f"Average time per document: {elapsed/processed_count:.2f} seconds")
        
        with times_lock:
            total_embeddings = len(doc_embedding_times) + len(section_embedding_times)
            if total_embeddings > 0:
                avg_embedding_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
                embeddings_per_min = 60.0 / avg_embedding_time if avg_embedding_time > 0 else 0
                logger.info(f"Total embeddings generated: {total_embeddings}")
                logger.info(f"Average embedding time: {avg_embedding_time:.3f} seconds")
                logger.info(f"Throughput: {embeddings_per_min:.1f} embeddings/minute")
        
        # Final rate limiter status
        final_rpm = window_rate_limiter.get_current_count()
        logger.info(f"Final rate limiter status: {final_rpm} requests in current window")
        
        if args and args.dry_run:
            logger.info("\nDRY RUN MODE - No changes were made to MongoDB")
            logger.info("Run without --dry-run to apply changes")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        # Restore original rate limiter if we modified it
        try:
            from backend.services.rag.rag_dependencies.ai_service import _voyage_rate_limiter
            if original_max_requests is not None:
                _voyage_rate_limiter.max_requests = original_max_requests
                _voyage_rate_limiter.min_delay = 60.0 / original_max_requests
                logger.info(f"Restored global rate limiter to {original_max_requests} req/min")
        except Exception as e:
            logger.debug(f"Could not restore rate limiter: {e}")
        
        # Restore original Voyage API key if we modified it
        try:
            if original_voyage_key is not None:
                os.environ["VOYAGE_API_KEY"] = original_voyage_key
                config_module.VOYAGE_API_KEY = original_voyage_key
                logger.info("Restored original Voyage API key")
            else:
                os.environ.pop("VOYAGE_API_KEY", None)
        except Exception as e:
            logger.debug(f"Could not restore Voyage API key: {e}")
        
        if client:
            client.close()
            logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

