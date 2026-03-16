"""
Rename MongoDB collection from uscis_policy_manual to uscis_policy
"""
import os
import sys
from pathlib import Path

from pymongo import MongoClient

# Setup path for imports
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import config
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

import backend.services.rag.config as config_module

config_module.load_environment("production")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found")

DB_NAME = "public"
OLD_COLL_NAME = "uscis_policy_manual"
NEW_COLL_NAME = "uscis_policy"

print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)

try:
    client.admin.command('ping')
    print("MongoDB connection successful.")

    db = client[DB_NAME]

    # Check if old collection exists
    if OLD_COLL_NAME in db.list_collection_names():
        print(f"Found collection: {OLD_COLL_NAME}")

        # Check if new collection already exists
        if NEW_COLL_NAME in db.list_collection_names():
            print(f"WARNING: Collection {NEW_COLL_NAME} already exists!")
            response = input(f"Do you want to drop {NEW_COLL_NAME} and rename {OLD_COLL_NAME}? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                sys.exit(0)
            db[NEW_COLL_NAME].drop()
            print(f"Dropped existing collection: {NEW_COLL_NAME}")

        # Rename collection
        print(f"Renaming collection {OLD_COLL_NAME} to {NEW_COLL_NAME}...")
        db[OLD_COLL_NAME].rename(NEW_COLL_NAME)
        print(f"Successfully renamed collection from {OLD_COLL_NAME} to {NEW_COLL_NAME}")

        # Verify
        if NEW_COLL_NAME in db.list_collection_names():
            count = db[NEW_COLL_NAME].count_documents({})
            print(f"Verification: {NEW_COLL_NAME} exists with {count} documents")
        else:
            print(f"ERROR: Collection {NEW_COLL_NAME} not found after rename!")
    else:
        print(f"Collection {OLD_COLL_NAME} not found. It may have already been renamed or doesn't exist.")
        if NEW_COLL_NAME in db.list_collection_names():
            count = db[NEW_COLL_NAME].count_documents({})
            print(f"Collection {NEW_COLL_NAME} already exists with {count} documents.")

finally:
    client.close()
    print("MongoDB connection closed.")

