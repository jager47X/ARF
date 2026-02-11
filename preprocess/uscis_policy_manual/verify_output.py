"""Verify the output JSON format"""
import json
from pathlib import Path

knowledge_dir = Path(__file__).parent.parent.parent / "Data" / "Knowledge"
json_path = knowledge_dir / "uscis_policy.json"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data['data']['uscis_policy']['documents']

# Find a document with "Background" in title
sample = None
for d in docs:
    if 'Background' in d.get('title', '') and len(d.get('clauses', [])) > 1:
        sample = d
        break

if sample:
    print(f"Title: {sample['title']}")
    print(f"References: {sample.get('references', [])}")
    print(f"\nClauses:")
    for c in sample['clauses'][:5]:
        print(f"  {c['number']}: {c['title'][:60]}")
        print(f"    Text length: {len(c['text'])} chars")
        print(f"    Text preview: {c['text'][:150]}...")
        print()

# Check for documents with references
ref_docs = [d for d in docs if d.get('references')]
print(f"\nTotal documents: {len(docs)}")
print(f"Documents with references: {len(ref_docs)}")
print(f"Documents without 'article' field: {sum(1 for d in docs if 'article' not in d)}")
print(f"Documents with 'references' field: {sum(1 for d in docs if 'references' in d)}")

