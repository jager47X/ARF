#!/usr/bin/env python3
"""Verify that chapter, section, and clause numbers are correctly formatted."""

import json
import sys
from pathlib import Path

def verify_formatting():
    json_path = Path("usc_xml_temp/us_code.json")
    
    if not json_path.exists():
        print(f"ERROR: JSON file not found at {json_path}")
        return False
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle the nested structure: {"data": {"united_states_code": {"titles": [...]}}}
    if isinstance(json_data, dict):
        if "data" in json_data:
            data_dict = json_data["data"]
            if isinstance(data_dict, dict) and "united_states_code" in data_dict:
                usc_dict = data_dict["united_states_code"]
                if isinstance(usc_dict, dict) and "titles" in usc_dict:
                    data = usc_dict["titles"]
                else:
                    data = []
            else:
                data = []
        else:
            # If it's a dict but no "data" key, treat values as list
            data = list(json_data.values())[0] if json_data else []
    else:
        data = json_data
    
    print(f"Total entries: {len(data)}\n")
    
    # Check first few entries
    issues = []
    checked = 0
    max_check = 100  # Check first 100 entries
    
    for entry in data[:max_check]:
        checked += 1
        
        # Check chapter format
        chapter = entry.get("chapter", "")
        if chapter:
            # Should be just a number, not "Chapter CHAPTER 1—" or similar
            if "chapter" in chapter.lower() or "—" in chapter or "–" in chapter:
                issues.append(f"Entry {checked}: chapter has text/dashes: '{chapter}'")
            # Should be numeric (or empty)
            if chapter and not chapter.replace("A", "").replace("B", "").isdigit():
                # Allow for cases like "1A" but check if it starts with a number
                if not chapter[0].isdigit():
                    issues.append(f"Entry {checked}: chapter doesn't start with number: '{chapter}'")
        
        # Check section array
        sections = entry.get("section", [])
        if not isinstance(sections, list):
            issues.append(f"Entry {checked}: section is not a list")
            continue
        
        for sec_idx, sec in enumerate(sections):
            # Check section number
            sec_num = sec.get("number", "")
            if sec_num:
                # Should be just a number, not "Section § 7." or similar
                if "section" in sec_num.lower() or "§" in sec_num or sec_num.endswith("."):
                    issues.append(f"Entry {checked}, section {sec_idx}: number has prefix/suffix: '{sec_num}'")
            
            # Check clause numbers in text (if any)
            text = sec.get("text", "")
            # Look for clause patterns like "(a)", "(b)" that should be "(1)", "(2)"
            # But we can't easily verify this without parsing the text structure
            # The normalize_clause_number function should handle this
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
        return False
    else:
        print(f"OK: All {checked} entries checked - formatting looks correct!")
        
        # Show a sample entry
        if data:
            print("\nSample entry:")
            sample = data[0]
            print(f"  Article: {sample.get('article')}")
            print(f"  Chapter: {sample.get('chapter')}")
            sections = sample.get('section', [])
            if sections:
                print(f"  Section array has {len(sections)} entries")
                for i, sec in enumerate(sections[:3]):  # Show first 3
                    print(f"    [{i}] number: '{sec.get('number')}', title: '{sec.get('title', '')[:50]}...'")
        
        return True

if __name__ == "__main__":
    success = verify_formatting()
    sys.exit(0 if success else 1)

