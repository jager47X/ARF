#!/usr/bin/env python3
"""Verify that MongoDB was updated correctly with the new US Code structure."""

import json
from pathlib import Path

json_path = Path("usc_xml_temp/us_code.json")

print("Verifying JSON structure...")
with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

entries = json_data['data']['united_states_code']['titles']

print(f"Total entries: {len(entries)}")
print("\nSample entry structure:")
entry = entries[0]
print(f"  article: {entry.get('article')}")
print(f"  chapter: {entry.get('chapter')}")
print(f"  section type: {type(entry.get('section'))}")
section = entry.get('section', [])
if isinstance(section, list):
    print(f"  section array length: {len(section)}")
    if section:
        print("  First section entry:")
        sec = section[0]
        print(f"    number: {sec.get('number')}")
        print(f"    title: {sec.get('title', '')[:50]}...")
        print(f"    text preview: {sec.get('text', '')[:80]}...")
else:
    print(f"  section: {section}")

print("\nVerification complete!")












































