# check_document_sections.py
"""
Check a specific document or documents with many sections to see if there are duplicates.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from pymongo import MongoClient
from collections import Counter

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib.util
import types

if 'backend' not in sys.modules:
    backend_mod = types.ModuleType('backend')
    sys.modules['backend'] = backend_mod
    
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
            
            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
            if rag_init.exists():
                spec = importlib.util.spec_from_file_location('backend.services.rag', rag_init)
                if spec and spec.loader:
                    rag_mod = importlib.util.module_from_spec(spec)
                    sys.modules['backend.services.rag'] = rag_mod
                    spec.loader.exec_module(rag_mod)
                    setattr(services_mod, 'rag', rag_mod)
                    sys.modules['services.rag'] = rag_mod
                    
                    config_file = backend_dir / 'services' / 'rag' / 'config.py'
                    if config_file.exists():
                        spec = importlib.util.spec_from_file_location('backend.services.rag.config', config_file)
                        if spec and spec.loader:
                            config_mod = importlib.util.module_from_spec(spec)
                            sys.modules['backend.services.rag.config'] = config_mod
                            spec.loader.exec_module(config_mod)
                            setattr(rag_mod, 'config', config_mod)
                            sys.modules['services.rag.config'] = config_mod

def parse_args():
    parser = argparse.ArgumentParser(description="Check document sections")
    parser.add_argument("--production", action="store_true", help="Use production environment")
    parser.add_argument("--doc-id", type=int, help="Check specific document by index (0-based)")
    parser.add_argument("--top-n", type=int, default=10, help="Show top N documents with most sections")
    return parser.parse_args()

args = parse_args() if __name__ == "__main__" else None
env_override = "production" if args and args.production else "production"

import backend.services.rag.config as config_module
config_module.load_environment(env_override)
import os
MONGO_URI = os.getenv("MONGO_URI")
COLLECTION = config_module.COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_document_sections")

CFR_CONF = COLLECTION.get("CFR_SET")
DB_NAME = CFR_CONF["db_name"]
COLL_NAME = CFR_CONF["main_collection_name"]

def main():
    client = None
    try:
        tls_config = {"tls": True} if "mongodb+srv://" in MONGO_URI or "mongodb.net" in MONGO_URI else {}
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, **tls_config)
        client.admin.command('ping')
        
        db = client[DB_NAME]
        coll = db.get_collection(COLL_NAME)
        
        if args and args.doc_id is not None:
            # Check specific document
            doc = list(coll.find().skip(args.doc_id).limit(1))
            if doc:
                doc = doc[0]
                sections = doc.get("sections", [])
                logger.info(f"Document #{args.doc_id}:")
                logger.info(f"  Title: {doc.get('title', 'N/A')}")
                logger.info(f"  ID: {doc.get('_id')}")
                logger.info(f"  Number of sections: {len(sections)}")
                
                # Check for duplicate section titles/text
                section_titles = [s.get("title", "") for s in sections]
                section_texts = [s.get("text", "")[:100] for s in sections]
                
                title_counts = Counter(section_titles)
                text_counts = Counter(section_texts)
                
                duplicate_titles = {k: v for k, v in title_counts.items() if v > 1 and k}
                duplicate_texts = {k: v for k, v in text_counts.items() if v > 1 and k}
                
                if duplicate_titles:
                    logger.warning(f"  Found {len(duplicate_titles)} duplicate section titles:")
                    for title, count in list(duplicate_titles.items())[:5]:
                        logger.warning(f"    '{title[:50]}...' appears {count} times")
                
                if duplicate_texts:
                    logger.warning(f"  Found {len(duplicate_texts)} duplicate section texts (first 100 chars):")
                    for text, count in list(duplicate_texts.items())[:5]:
                        logger.warning(f"    '{text[:50]}...' appears {count} times")
                
                # Show first few sections
                logger.info(f"  First 5 sections:")
                for i, section in enumerate(sections[:5], 1):
                    logger.info(f"    Section {i}: title='{section.get('title', 'N/A')[:50]}', has_embedding={bool(section.get('embedding'))}")
            else:
                logger.error(f"Document #{args.doc_id} not found")
        else:
            # Find documents with most sections
            logger.info("Finding documents with most sections...")
            pipeline = [
                {"$project": {
                    "title": 1,
                    "num_sections": {"$size": {"$ifNull": ["$sections", []]}}
                }},
                {"$sort": {"num_sections": -1}},
                {"$limit": args.top_n if args else 10}
            ]
            
            results = list(coll.aggregate(pipeline))
            logger.info(f"\nTop {len(results)} documents with most sections:")
            for i, doc in enumerate(results, 1):
                logger.info(f"  {i}. {doc.get('title', 'N/A')[:60]}: {doc.get('num_sections', 0)} sections")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    main()











































