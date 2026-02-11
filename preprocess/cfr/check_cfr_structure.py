# check_cfr_structure.py
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CFR_DOCUMENT_PATH = BASE_DIR / "Data/Knowledge/code_of_federal_regulations.json"

with open(CFR_DOCUMENT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("data", {}).get("code_of_federal_regulations", {}).get("titles") or data.get("data", {}).get("code_of_federal_regulations", {}).get("regulations", [])

print(f"Total items: {len(items)}\n")

# Check first 5 entries
for i in range(min(5, len(items))):
    item = items[i]
    print(f"Entry {i+1}:")
    print(f"  article: {item.get('article', '')}")
    print(f"  part: {item.get('part', '')}")
    print(f"  chapter: {item.get('chapter', '')}")
    print(f"  section: {item.get('section', '')}")
    print(f"  title: {item.get('title', '')[:100]}...")
    print(f"  sections count: {len(item.get('sections', []))}")
    if item.get('sections'):
        print(f"  First section: number={item['sections'][0].get('number')}, title={item['sections'][0].get('title', '')[:50]}, text={item['sections'][0].get('text', '')[:50]}")
    print()












































