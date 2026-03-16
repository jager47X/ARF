#!/usr/bin/env python3
"""
Fix the format of ca_regulations.json to match the standard format.
Reorders fields and filters out invalid entries.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def is_valid_regulation(reg: Dict[str, Any]) -> bool:
    """Check if a regulation entry has valid content."""
    if not reg.get("clauses"):
        return False

    # Check if clauses contain actual regulation text (not just navigation)
    for clause in reg["clauses"]:
        text = clause.get("text", "")
        # Skip if it's just navigation breadcrumbs
        if text.startswith("Home\n") or "Division" in text[:100]:
            # Check if there's actual content beyond navigation
            lines = text.split("\n")
            content_lines = [line for line in lines if line.strip() and not any(kw in line for kw in ["Home", "Title", "Division", "Chapter"])]
            if len("\n".join(content_lines)) < 50:
                continue
        # If we have substantial text, it's valid
        if len(text) > 100:
            return True

    return False

def extract_section_info(text: str) -> tuple[str, str]:
    """Extract section number and title from text."""
    section_num = ""
    title = ""

    # Try to find section number
    section_match = re.search(r'(?:Section|§)\s*(\d+(?:\.\d+)?)', text[:500], re.IGNORECASE)
    if section_match:
        section_num = section_match.group(1)

    # Try to extract title from text
    lines = text.split("\n")
    for line in lines[:20]:
        line = line.strip()
        if not line:
            continue
        # Skip navigation lines
        if any(kw in line.lower() for kw in ["home", "title", "division", "chapter", "section"]):
            continue
        # Look for a meaningful title (10-200 chars, not all caps)
        if 10 <= len(line) <= 200 and not line.isupper():
            title = line
            break

    return section_num, title

def fix_regulation_format(reg: Dict[str, Any]) -> Dict[str, Any]:
    """Fix the format of a single regulation entry."""
    # Extract current values
    article = reg.get("article", "")
    part = reg.get("part", "")
    section = reg.get("section", "")
    title = reg.get("title", "")
    clauses = reg.get("clauses", [])

    # Try to extract better section info from clauses
    if clauses:
        first_clause_text = clauses[0].get("text", "")
        section_num, extracted_title = extract_section_info(first_clause_text)

        if section_num and not section.startswith("Section "):
            section = f"Section {section_num}"
        elif not section or section == "Section":
            section = section_num if section_num else ""

        if extracted_title and (not title or title == "Regulation Section"):
            title = extracted_title

    # Clean up clauses - remove navigation-only text
    cleaned_clauses = []
    for clause in clauses:
        text = clause.get("text", "")
        # Skip if it's just navigation
        if text.startswith("Home\n") and "Division" in text[:200]:
            # Try to extract actual content
            lines = text.split("\n")
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(kw in line.lower() for kw in ["home", "title", "division", "chapter"]):
                    content_lines.append(line)
            if content_lines:
                text = "\n".join(content_lines)
            else:
                continue  # Skip this clause entirely

        if len(text) > 30:  # Only keep substantial content
            cleaned_clauses.append({
                "number": clause.get("number", len(cleaned_clauses) + 1),
                "title": clause.get("title", ""),
                "text": text[:10000]  # Limit length
            })

    # If no valid clauses, return None (will be filtered out)
    if not cleaned_clauses:
        return None

    # Return in correct format order: article, part, section, title, clauses
    return {
        "article": article,
        "part": part,
        "section": section if section else "",
        "title": title if title else part or article,
        "clauses": cleaned_clauses
    }

def fix_json_file(input_path: Path, output_path: Path = None):
    """Fix the format of ca_regulations.json."""
    if output_path is None:
        output_path = input_path

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regulations = data.get("data", {}).get("california_code_of_regulations", {}).get("regulations", [])
    print(f"Found {len(regulations)} regulations")

    # Fix each regulation
    fixed_regulations = []
    skipped = 0

    for reg in regulations:
        fixed = fix_regulation_format(reg)
        if fixed and is_valid_regulation(fixed):
            fixed_regulations.append(fixed)
        else:
            skipped += 1

    print(f"Fixed {len(fixed_regulations)} regulations, skipped {skipped} invalid entries")

    # Create output structure
    output_data = {
        "data": {
            "california_code_of_regulations": {
                "regulations": fixed_regulations
            }
        }
    }

    # Write output
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Fixed JSON file created: {output_path} ({file_size:.2f} MB)")
    print(f"Total valid regulations: {len(fixed_regulations)}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    input_path = base_dir / "Data" / "Knowledge" / "ca_regulations.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        exit(1)

    fix_json_file(input_path)


