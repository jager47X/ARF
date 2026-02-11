#!/usr/bin/env python3
"""Verify that clause numbers are sequential starting from 1."""

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

print("Verifying clause numbers are sequential starting from 1...")
print("=" * 80)

# Get a sample document with multiple clauses
sample = coll.find_one({
    "article": "Title 1",
    "section": {"$regex": "^Section 7$"}
})

if sample:
    print("\nSample document (Title 1, Section 7):")
    print(f"  article: {sample.get('article')}")
    print(f"  chapter: {sample.get('chapter')}")
    print(f"  section: {sample.get('section')}")
    print(f"  title: {sample.get('title')}")
    clauses = sample.get('clauses', [])
    print(f"  clauses count: {len(clauses)}")
    
    if clauses:
        print("\n  Clause numbers:")
        issues = []
        for i, clause in enumerate(clauses):
            clause_num = clause.get('number', '')
            expected = str(i + 1)
            status = "OK" if clause_num == expected else "ISSUE"
            print(f"    [{i}] number: '{clause_num}' (expected: '{expected}') {status}")
            if clause_num != expected:
                issues.append(f"Clause {i}: got '{clause_num}', expected '{expected}'")
        
        if issues:
            print("\n  ISSUES FOUND:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("\n  OK: All clause numbers are sequential starting from 1!")
else:
    print("Sample document not found!")

# Check a few more documents
print("\n" + "=" * 80)
print("Checking multiple documents...")
samples = list(coll.find({"article": "Title 1"}).limit(10))
all_ok = True
for doc in samples:
    clauses = doc.get('clauses', [])
    for i, clause in enumerate(clauses):
        clause_num = clause.get('number', '')
        expected = str(i + 1)
        if clause_num != expected:
            all_ok = False
            print(f"  ISSUE in {doc.get('section')}: clause {i} has '{clause_num}', expected '{expected}'")
            break

if all_ok:
    print("  OK: All checked documents have sequential clause numbers!")

client.close()
print("\n" + "=" * 80)
print("Verification complete!")












































