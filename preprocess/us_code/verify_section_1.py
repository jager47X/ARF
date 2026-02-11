# -*- coding: utf-8 -*-
import json
from pathlib import Path

# Load the JSON file
json_path = Path("usc_xml_temp/us_code.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find Title 1, Section 1
titles = data['data']['united_states_code']['titles']
title1_sections = [s for s in titles if s.get('article') == 'Title 1']

print(f"Title 1 has {len(title1_sections)} sections\n")

# Find section with section number containing "1"
found = False
for section in title1_sections:
    section_num = section.get('section', '')
    if '1' in str(section_num) or section_num == '§ 1' or section_num == '§ 1.':
        print("=" * 80)
        print("CURRENT STRUCTURE (from old JSON file):")
        print("=" * 80)
        print(f"Article: {section.get('article')}")
        print(f"Chapter: {section.get('chapter')} (should be just '1')")
        print(f"Section: {section.get('section')} (should be array)")
        print(f"Title: {section.get('title')}")
        
        # Check structure
        if 'clauses' in section:
            clauses = section.get('clauses', [])
            print(f"\nOLD FORMAT: Has 'clauses' array with {len(clauses)} entries")
            print("   Should have 'section' array instead")
            if clauses:
                print(f"\nFirst clause content:")
                print(json.dumps(clauses[0], indent=2))
                text = clauses[0].get('text', '')
                print(f"\nText preview (first 500 chars):")
                print(text[:500])
                if text.startswith('"') and text.endswith('"'):
                    print("\nISSUE: Text has extra quotes at start/end")
                print(f"\nFull text length: {len(text)} characters")
        elif 'section' in section and isinstance(section.get('section'), list):
            sec_array = section.get('section', [])
            print(f"\nNEW FORMAT: Has 'section' array with {len(sec_array)} entries")
            if sec_array:
                print(f"\nFirst section entry:")
                print(json.dumps(sec_array[0], indent=2))
        else:
            print(f"\nUNEXPECTED: 'section' field is not an array")
        
        found = True
        break

if not found:
    print("Could not find Section 1 in Title 1")

print("\n" + "=" * 80)
print("EXPECTED STRUCTURE (after regeneration):")
print("=" * 80)
print("""
{
  "article": "Title 1",
  "chapter": "1",
  "section": [
    {
      "number": "1",
      "title": "Words denoting number, gender, and so forth",
      "text": "(1) In divisions A through D, the term 'this Act' refers to divisions A through D."
    }
  ]
}
""")
