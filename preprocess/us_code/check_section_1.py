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

# Find section with section number "§ 1" or "1"
for section in title1_sections:
    section_num = section.get('section', '')
    if '1' in section_num or section_num == '§ 1' or section_num == '§ 1.':
        print("=" * 80)
        print(f"Article: {section.get('article')}")
        print(f"Chapter: {section.get('chapter')}")
        print(f"Section: {section.get('section')}")
        print(f"Title: {section.get('title')}")
        
        # Check if it has clauses (old format) or section array (new format)
        if 'clauses' in section:
            clauses = section.get('clauses', [])
            print(f"\nHas {len(clauses)} clauses (old format)")
            if clauses:
                print(f"\nFirst clause:")
                print(json.dumps(clauses[0], indent=2))
                print(f"\nText (first 1000 chars):")
                print(clauses[0].get('text', '')[:1000])
        elif 'section' in section and isinstance(section.get('section'), list):
            sec_array = section.get('section', [])
            print(f"\nHas {len(sec_array)} section entries (new format)")
            if sec_array:
                print(f"\nFirst section entry:")
                print(json.dumps(sec_array[0], indent=2))
        print("\n" + "=" * 80)
        break
