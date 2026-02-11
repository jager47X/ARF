#!/usr/bin/env python3
"""Quick script to check document titles"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from services.rag.config import COLLECTION, MONGO_URI
from pymongo import MongoClient

US_CODE_CONF = COLLECTION.get("US_CODE_SET")
DB_NAME = US_CODE_CONF["db_name"]
COLL_NAME = US_CODE_CONF["main_collection_name"]

client = MongoClient(MONGO_URI, tls=True)
db = client[DB_NAME]
coll = db[COLL_NAME]

docs = list(coll.find(
    {"article": "Title 42", "chapter": "Chapter 6A", "section": "Section 300g"},
    {"title": 1, "_id": 1}
).limit(10))

print(f"Found {len(docs)} documents:")
for d in docs:
    print(f"  {d['_id']}: {d.get('title', 'NO TITLE')}")

client.close()





































