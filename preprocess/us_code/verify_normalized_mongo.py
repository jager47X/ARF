#!/usr/bin/env python3
"""Verify that MongoDB documents have normalized chapter, section, and clause numbers."""

import os
import sys
from pathlib import Path

# Setup path for module execution - same as ingest_us_code.py
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create 'backend' -> 'kyr-backend' mapping for imports (same as ingest_us_code.py)
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

print("Verifying normalized data in MongoDB...")
print("=" * 80)

# Get a sample document (Title 1, Section 1)
sample = coll.find_one({
    "article": "Title 1",
    "section": {"$regex": "^Section 1$"}
})

if sample:
    print("\nSample document (Title 1, Section 1):")
    print(f"  article: {sample.get('article')}")
    print(f"  chapter: {sample.get('chapter')}")
    print(f"  section: {sample.get('section')}")
    print(f"  title: {sample.get('title')}")
    print(f"  clauses count: {len(sample.get('clauses', []))}")
    
    if sample.get('clauses'):
        print("\n  Clauses:")
        for i, clause in enumerate(sample.get('clauses', [])[:3]):  # Show first 3
            print(f"    [{i}] number: '{clause.get('number')}', title: '{clause.get('title', '')[:50]}...'")
            print(f"        text preview: {clause.get('text', '')[:80]}...")
    
    # Verify normalization
    print("\n  Verification:")
    chapter = sample.get('chapter', '')
    section = sample.get('section', '')
    issues = []
    
    if chapter and not chapter.startswith('Chapter '):
        issues.append(f"Chapter should start with 'Chapter ': got '{chapter}'")
    if section and not section.startswith('Section '):
        issues.append(f"Section should start with 'Section ': got '{section}'")
    
    for clause in sample.get('clauses', []):
        clause_num = clause.get('number', '')
        if clause_num and not clause_num.isdigit():
            issues.append(f"Clause number should be a digit: got '{clause_num}'")
    
    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  OK: All fields normalized correctly!")
else:
    print("Sample document not found!")

client.close()
print("\n" + "=" * 80)
print("Verification complete!")

