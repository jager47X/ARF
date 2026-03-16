#!/usr/bin/env python3
"""Verify that '--' has been removed from titles."""

import os
import sys
from pathlib import Path

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create 'backend' -> 'kyr-backend' mapping for imports
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

# Import config
import backend.services.rag.config as config_module

config_module.load_environment("local")

from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")
US_CODE_CONF = config_module.COLLECTION.get("US_CODE_SET")
DB_NAME = US_CODE_CONF["db_name"]
COLL_NAME = US_CODE_CONF["main_collection_name"]

# Configure TLS for MongoDB Atlas connections
tls_config = {}
if MONGO_URI and "mongodb+srv://" in MONGO_URI:
    tls_config = {"tls": True}
elif MONGO_URI and ("mongodb.net" in MONGO_URI or "mongodb.com" in MONGO_URI):
    tls_config = {"tls": True}

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, **tls_config)
db = client[DB_NAME]
coll = db[COLL_NAME]

print("Verifying that '--' and '.' have been removed from titles...")
print("=" * 80)

# Check for titles with '--' or '.'
issues = []
samples_checked = 0
max_check = 1000

for doc in coll.find({}).limit(max_check):
    samples_checked += 1

    # Check main title
    title = doc.get("title", "")
    if "--" in title:
        issues.append(f"Document title has '--': '{title[:80]}...'")
    if "." in title:
        issues.append(f"Document title has '.': '{title[:80]}...'")

    # Check clause titles
    for clause in doc.get("clauses", []):
        clause_title = clause.get("title", "")
        if "--" in clause_title:
            issues.append(f"Clause title has '--': '{clause_title[:80]}...' (in section {doc.get('section', 'unknown')})")
        if "." in clause_title:
            issues.append(f"Clause title has '.': '{clause_title[:80]}...' (in section {doc.get('section', 'unknown')})")

print(f"\nChecked {samples_checked} documents")

if issues:
    print(f"\nISSUES FOUND ({len(issues)}):")
    for issue in issues[:20]:  # Show first 20
        print(f"  - {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more issues")
else:
    print("\nOK: No '--' or '.' found in any titles!")

client.close()
print("\n" + "=" * 80)
print("Verification complete!")

