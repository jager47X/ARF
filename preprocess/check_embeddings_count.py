# check_embeddings_count.py
"""
Check how many embeddings are done for CFR and US Code in production MongoDB.
Counts both document-level and sub-level (sections/clauses) embeddings.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from pymongo import MongoClient

# Setup path for module execution
# From: kyr-backend/services/rag/preprocess/check_embeddings_count.py
# To: kyr-backend (4 levels up)
backend_dir = Path(__file__).resolve().parent.parent.parent.parent
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
    parser = argparse.ArgumentParser(description="Check embedding counts for CFR and US Code")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (.env.production)"
    )
    return parser.parse_args()

# Import config module and load environment
# The module setup above should have loaded it into sys.modules
# Try to get it from sys.modules first
config_module = sys.modules.get('backend.services.rag.config')
if config_module is None:
    # Try services.rag.config
    config_module = sys.modules.get('services.rag.config')
if config_module is None:
    # If not in sys.modules, try to load it directly from file
    config_file = backend_dir / 'services' / 'rag' / 'config.py'
    if not config_file.exists():
        raise ImportError(f"Could not find config.py file at {config_file}")
    spec = importlib.util.spec_from_file_location('config_module', config_file)
    if spec and spec.loader:
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    else:
        raise ImportError(f"Could not load config.py from {config_file}")

# Load environment based on args
args = parse_args() if __name__ == "__main__" else None
env_override = None
if args:
    if args.production:
        env_override = "production"
    else:
        env_override = "production"

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
logger = logging.getLogger("check_embeddings_count")

CFR_CONF = COLLECTION.get("CFR_SET")
US_CODE_CONF = COLLECTION.get("US_CODE_SET")

if not CFR_CONF:
    raise ValueError("CFR_SET not found in COLLECTION config")
if not US_CODE_CONF:
    raise ValueError("US_CODE_SET not found in COLLECTION config")

def has_document_embedding(doc) -> bool:
    """Check if document has an embedding."""
    embedding = doc.get("embedding")
    return embedding is not None and len(embedding) > 0

def has_section_embedding(section) -> bool:
    """Check if section has an embedding."""
    embedding = section.get("embedding")
    return embedding is not None and len(embedding) > 0

def has_clause_embedding(clause) -> bool:
    """Check if clause has an embedding."""
    embedding = clause.get("embedding")
    return embedding is not None and len(embedding) > 0

def check_cfr_embeddings(client, db_name, coll_name):
    """Check CFR embedding counts."""
    logger.info(f"\n{'='*60}")
    logger.info("Checking CFR (Code of Federal Regulations) Embeddings")
    logger.info(f"{'='*60}")

    db = client[db_name]
    coll = db.get_collection(coll_name)

    # Get total count
    total_docs = coll.count_documents({})
    logger.info(f"Total documents: {total_docs}")

    # Count documents with document-level embeddings
    docs_with_doc_embedding = coll.count_documents({
        "embedding": {"$exists": True, "$nin": [None, []]}
    })

    # Use aggregation to count section-level embeddings
    logger.info("Analyzing section-level embeddings...")

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
                "num_sections_with_embedding": {"$sum": "$sections_with_embedding"}
            }
        },
        {
            "$group": {
                "_id": None,
                "total_docs": {"$sum": 1},
                "total_sections": {"$sum": "$num_sections"},
                "total_sections_with_embedding": {"$sum": "$num_sections_with_embedding"},
                "docs_with_doc_embedding": {"$sum": "$has_doc_embedding"}
            }
        }
    ]

    result = list(coll.aggregate(pipeline))
    if result:
        stats = result[0]
        total_sections = stats.get("total_sections", 0)
        sections_with_embedding = stats.get("total_sections_with_embedding", 0)
    else:
        total_sections = 0
        sections_with_embedding = 0

    # Calculate total embeddings
    total_embeddings = docs_with_doc_embedding + sections_with_embedding

    logger.info("\nDocument-level embeddings:")
    logger.info(f"  Documents with embedding: {docs_with_doc_embedding} ({docs_with_doc_embedding/total_docs*100:.1f}%)" if total_docs > 0 else "  Documents with embedding: 0")
    logger.info(f"  Documents without embedding: {total_docs - docs_with_doc_embedding} ({(total_docs - docs_with_doc_embedding)/total_docs*100:.1f}%)" if total_docs > 0 else "  Documents without embedding: 0")

    logger.info("\nSection-level embeddings:")
    logger.info(f"  Total sections: {total_sections}")
    logger.info(f"  Sections with embedding: {sections_with_embedding} ({sections_with_embedding/total_sections*100:.1f}%)" if total_sections > 0 else "  Sections with embedding: 0")
    logger.info(f"  Sections without embedding: {total_sections - sections_with_embedding} ({(total_sections - sections_with_embedding)/total_sections*100:.1f}%)" if total_sections > 0 else "  Sections without embedding: 0")

    logger.info(f"\nTOTAL EMBEDDINGS: {total_embeddings}")
    logger.info(f"  - Document embeddings: {docs_with_doc_embedding}")
    logger.info(f"  - Section embeddings: {sections_with_embedding}")

    return {
        "total_docs": total_docs,
        "docs_with_doc_embedding": docs_with_doc_embedding,
        "total_sections": total_sections,
        "sections_with_embedding": sections_with_embedding,
        "total_embeddings": total_embeddings
    }

def check_us_code_embeddings(client, db_name, coll_name):
    """Check US Code embedding counts."""
    logger.info(f"\n{'='*60}")
    logger.info("Checking US Code (United States Code) Embeddings")
    logger.info(f"{'='*60}")

    db = client[db_name]
    coll = db.get_collection(coll_name)

    # Get total count
    total_docs = coll.count_documents({})
    logger.info(f"Total documents: {total_docs}")

    # Count documents with document-level embeddings
    docs_with_doc_embedding = coll.count_documents({
        "embedding": {"$exists": True, "$nin": [None, []]}
    })

    # Use aggregation to count clause-level embeddings
    logger.info("Analyzing clause-level embeddings...")

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
                "clauses": {"$ifNull": ["$clauses", []]},
                "clauses_with_embedding": {
                    "$map": {
                        "input": {"$ifNull": ["$clauses", []]},
                        "as": "clause",
                        "in": {
                            "$cond": [
                                {"$and": [
                                    {"$ne": ["$$clause.embedding", None]},
                                    {"$gt": [{"$size": {"$ifNull": ["$$clause.embedding", []]}}, 0]}
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
                "num_clauses": {"$size": "$clauses"},
                "num_clauses_with_embedding": {"$sum": "$clauses_with_embedding"}
            }
        },
        {
            "$group": {
                "_id": None,
                "total_docs": {"$sum": 1},
                "total_clauses": {"$sum": "$num_clauses"},
                "total_clauses_with_embedding": {"$sum": "$num_clauses_with_embedding"},
                "docs_with_doc_embedding": {"$sum": "$has_doc_embedding"}
            }
        }
    ]

    result = list(coll.aggregate(pipeline))
    if result:
        stats = result[0]
        total_clauses = stats.get("total_clauses", 0)
        clauses_with_embedding = stats.get("total_clauses_with_embedding", 0)
    else:
        total_clauses = 0
        clauses_with_embedding = 0

    # Calculate total embeddings
    total_embeddings = docs_with_doc_embedding + clauses_with_embedding

    logger.info("\nDocument-level embeddings:")
    logger.info(f"  Documents with embedding: {docs_with_doc_embedding} ({docs_with_doc_embedding/total_docs*100:.1f}%)" if total_docs > 0 else "  Documents with embedding: 0")
    logger.info(f"  Documents without embedding: {total_docs - docs_with_doc_embedding} ({(total_docs - docs_with_doc_embedding)/total_docs*100:.1f}%)" if total_docs > 0 else "  Documents without embedding: 0")

    logger.info("\nClause-level embeddings:")
    logger.info(f"  Total clauses: {total_clauses}")
    logger.info(f"  Clauses with embedding: {clauses_with_embedding} ({clauses_with_embedding/total_clauses*100:.1f}%)" if total_clauses > 0 else "  Clauses with embedding: 0")
    logger.info(f"  Clauses without embedding: {total_clauses - clauses_with_embedding} ({(total_clauses - clauses_with_embedding)/total_clauses*100:.1f}%)" if total_clauses > 0 else "  Clauses without embedding: 0")

    logger.info(f"\nTOTAL EMBEDDINGS: {total_embeddings}")
    logger.info(f"  - Document embeddings: {docs_with_doc_embedding}")
    logger.info(f"  - Clause embeddings: {clauses_with_embedding}")

    return {
        "total_docs": total_docs,
        "docs_with_doc_embedding": docs_with_doc_embedding,
        "total_clauses": total_clauses,
        "clauses_with_embedding": clauses_with_embedding,
        "total_embeddings": total_embeddings
    }

def main():
    """Check embedding counts for both CFR and US Code."""
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

        # Check CFR embeddings
        cfr_stats = check_cfr_embeddings(
            client,
            CFR_CONF["db_name"],
            CFR_CONF["main_collection_name"]
        )

        # Check US Code embeddings
        us_code_stats = check_us_code_embeddings(
            client,
            US_CODE_CONF["db_name"],
            US_CODE_CONF["main_collection_name"]
        )

        # Print combined summary
        logger.info(f"\n{'='*60}")
        logger.info("COMBINED SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"CFR Total Embeddings: {cfr_stats['total_embeddings']}")
        logger.info(f"  - Document embeddings: {cfr_stats['docs_with_doc_embedding']}")
        logger.info(f"  - Section embeddings: {cfr_stats['sections_with_embedding']}")
        logger.info("")
        logger.info(f"US Code Total Embeddings: {us_code_stats['total_embeddings']}")
        logger.info(f"  - Document embeddings: {us_code_stats['docs_with_doc_embedding']}")
        logger.info(f"  - Clause embeddings: {us_code_stats['clauses_with_embedding']}")
        logger.info("")
        logger.info(f"GRAND TOTAL: {cfr_stats['total_embeddings'] + us_code_stats['total_embeddings']} embeddings")
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

