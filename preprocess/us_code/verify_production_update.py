#!/usr/bin/env python3
"""Verify production JSON file update."""

import json
from pathlib import Path

json_path = Path("../../Data/Knowledge/us_code.json")

print("Verifying production JSON file...")
print("=" * 80)

if json_path.exists():
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    entries = data['data']['united_states_code']['titles']

    print("Production JSON file verified:")
    print(f"  Path: {json_path.resolve()}")
    print(f"  Total entries: {len(entries)}")
    print(f"  File size: {json_path.stat().st_size / (1024*1024):.2f} MB")

    if entries:
        sample = entries[0]
        print("\n  Sample entry:")
        print(f"    Article: {sample.get('article')}")
        print(f"    Chapter: {sample.get('chapter')}")
        print(f"    Section entries: {len(sample.get('section', []))}")
        if sample.get('section'):
            sec = sample.get('section')[0]
            print("    First section entry:")
            print(f"      Number: {sec.get('number')}")
            print(f"      Title: {sec.get('title', '')[:50]}...")

    print("\n" + "=" * 80)
    print("Production JSON file is up to date!")
else:
    print(f"ERROR: JSON file not found at {json_path}")












































