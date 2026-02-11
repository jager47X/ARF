# check_cfr_embeddings.py
"""
Check how many CFR documents have embeddings in production MongoDB.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from pymongo import MongoClient

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

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Check CFR document embeddings")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (.env.production)"
    )
    return parser.parse_args()

# Load environment based on args
args = parse_args() if __name__ == "__main__" else None
env_override = None
if args:
    if args.production:
        env_override = "production"
    else:
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
logger = logging.getLogger("check_cfr_embeddings")

CFR_CONF = COLLECTION.get("CFR_SET")
if not CFR_CONF:
    raise ValueError("CFR_SET not found in COLLECTION config. Please add CFR_SET configuration to config.py")

DB_NAME: str = CFR_CONF["db_name"]
COLL_NAME: str = CFR_CONF["main_collection_name"]

def has_document_embedding(doc) -> bool:
    """Check if document has an embedding."""
    embedding = doc.get("embedding")
    return embedding is not None and len(embedding) > 0

def has_section_embedding(section) -> bool:
    """Check if section has an embedding."""
    embedding = section.get("embedding")
    return embedding is not None and len(embedding) > 0

def main():
    """Check embedding status of CFR documents."""
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
        
        # Get total count
        total_docs = coll.count_documents({})
        logger.info(f"Total documents in collection: {total_docs}")
        
        # Count documents with document-level embeddings
        docs_with_doc_embedding = coll.count_documents({"embedding": {"$exists": True, "$ne": None, "$ne": []}})
        
        # Use aggregation to count section-level embeddings more efficiently
        logger.info("Analyzing section-level embeddings (this may take a moment)...")
        
        # Count sections and their embeddings using aggregation
        pipeline = [
            {
                "$project": {
                    "has_doc_embedding": {
                        "$cond": [
                            {"$and": [
                                {"$ne": ["$embedding", None]},
                                {"$gt": [{"$size": {"$ifNull": ["$embedding", []]}}, 0]}
                            ]},
                            1, 0
                        ]
                    },
                    "sections": {"$ifNull": ["$sections", []]},
                    "sections_with_embedding": {
                        "$map": {
                            "input": {"$ifNull": ["$sections", []]},
                            "as": "section",
                            "in": {
                                "$cond": [
                                    {"$and": [
                                        {"$ne": ["$$section.embedding", None]},
                                        {"$gt": [{"$size": {"$ifNull": ["$$section.embedding", []]}}, 0]}
                                    ]},
                                    1, 0
                                ]
                            }
                        }
                    }
                }
            },
            {
                "$project": {
                    "has_doc_embedding": 1,
                    "num_sections": {"$size": "$sections"},
                    "num_sections_with_embedding": {"$sum": "$sections_with_embedding"},
                    "has_all_sections_embedded": {
                        "$cond": [
                            {"$and": [
                                {"$gt": [{"$size": "$sections"}, 0]},
                                {"$eq": [
                                    {"$sum": "$sections_with_embedding"},
                                    {"$size": "$sections"}
                                ]}
                            ]},
                            1, 0
                        ]
                    },
                    "has_some_sections_embedded": {
                        "$cond": [
                            {"$and": [
                                {"$gt": [{"$sum": "$sections_with_embedding"}, 0]},
                                {"$lt": [
                                    {"$sum": "$sections_with_embedding"},
                                    {"$size": "$sections"}
                                ]}
                            ]},
                            1, 0
                        ]
                    },
                    "has_no_sections_embedded": {
                        "$cond": [
                            {"$and": [
                                {"$gt": [{"$size": "$sections"}, 0]},
                                {"$eq": [{"$sum": "$sections_with_embedding"}, 0]}
                            ]},
                            1, 0
                        ]
                    },
                    "has_no_sections": {
                        "$cond": [
                            {"$eq": [{"$size": "$sections"}, 0]},
                            1, 0
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_docs": {"$sum": 1},
                    "total_sections": {"$sum": "$num_sections"},
                    "total_sections_with_embedding": {"$sum": "$num_sections_with_embedding"},
                    "docs_with_all_sections_embedded": {"$sum": "$has_all_sections_embedded"},
                    "docs_with_some_sections_embedded": {"$sum": "$has_some_sections_embedded"},
                    "docs_with_no_sections_embedded": {"$sum": "$has_no_sections_embedded"},
                    "docs_with_no_sections": {"$sum": "$has_no_sections"}
                }
            }
        ]
        
        result = list(coll.aggregate(pipeline))
        if result:
            stats = result[0]
            docs_with_all_sections_embedded = stats.get("docs_with_all_sections_embedded", 0)
            docs_with_some_sections_embedded = stats.get("docs_with_some_sections_embedded", 0)
            docs_with_no_sections_embedded = stats.get("docs_with_no_sections_embedded", 0)
            docs_with_no_sections = stats.get("docs_with_no_sections", 0)
            total_sections = stats.get("total_sections", 0)
            sections_with_embedding = stats.get("total_sections_with_embedding", 0)
        else:
            docs_with_all_sections_embedded = 0
            docs_with_some_sections_embedded = 0
            docs_with_no_sections_embedded = 0
            docs_with_no_sections = 0
            total_sections = 0
            sections_with_embedding = 0
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("CFR Embedding Status Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"")
        logger.info(f"Document-level embeddings:")
        logger.info(f"  Documents with embedding: {docs_with_doc_embedding} ({docs_with_doc_embedding/total_docs*100:.1f}%)")
        logger.info(f"  Documents without embedding: {total_docs - docs_with_doc_embedding} ({(total_docs - docs_with_doc_embedding)/total_docs*100:.1f}%)")
        logger.info(f"")
        logger.info(f"Section-level embeddings:")
        logger.info(f"  Total sections: {total_sections}")
        logger.info(f"  Sections with embedding: {sections_with_embedding} ({sections_with_embedding/total_sections*100:.1f}%)" if total_sections > 0 else "  Sections with embedding: 0")
        logger.info(f"  Sections without embedding: {total_sections - sections_with_embedding} ({(total_sections - sections_with_embedding)/total_sections*100:.1f}%)" if total_sections > 0 else "  Sections without embedding: 0")
        logger.info(f"")
        logger.info(f"Documents by section embedding status:")
        logger.info(f"  All sections embedded: {docs_with_all_sections_embedded} ({docs_with_all_sections_embedded/total_docs*100:.1f}%)")
        logger.info(f"  Some sections embedded: {docs_with_some_sections_embedded} ({docs_with_some_sections_embedded/total_docs*100:.1f}%)")
        logger.info(f"  No sections embedded: {docs_with_no_sections_embedded} ({docs_with_no_sections_embedded/total_docs*100:.1f}%)")
        logger.info(f"  No sections in document: {docs_with_no_sections} ({docs_with_no_sections/total_docs*100:.1f}%)")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
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

