#!/usr/bin/env python3
"""Check that clause numbers are converted from letters to numbers."""

import json
from pathlib import Path

json_path = Path("usc_xml_temp/us_code.json")

with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

entries = json_data['data']['united_states_code']['titles']

print("Checking for entries with multiple sections (clauses)...")
print("=" * 80)

found_examples = 0
for i, entry in enumerate(entries[:1000]):
    sections = entry.get('section', [])
    if len(sections) > 1:
        found_examples += 1
        print(f"\nEntry {i}:")
        print(f"  Article: {entry.get('article')}")
        print(f"  Chapter: {entry.get('chapter')}")
        print(f"  Number of sections: {len(sections)}")
        for j, sec in enumerate(sections[:5]):  # Show first 5
            text_preview = sec.get('text', '')[:100]
            print(f"    [{j}] number: '{sec.get('number')}', text: '{text_preview}...'")

        if found_examples >= 3:
            break

print(f"\n{'=' * 80}")
print(f"Found {found_examples} examples of entries with multiple sections")












































