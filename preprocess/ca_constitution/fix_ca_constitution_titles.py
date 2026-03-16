#!/usr/bin/env python3
"""
Fix the titles in ca_constitution.json to be meaningful and correct.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Article titles mapping
ARTICLE_TITLES = {
    "Article I": "Declaration of Rights",
    "Article II": "Voting, Initiative, Referendum, and Recall",
    "Article III": "State of California",
    "Article IV": "Legislative",
    "Article V": "Executive",
    "Article VI": "Judicial",
    "Article VII": "Public Officers and Employees",
    "Article VIII": "Local Government",
    "Article IX": "Education",
    "Article X": "Water",
    "Article XA": "Water Resources Development",
    "Article XB": "Marine Resources Protection Act of 1990",
    "Article XI": "Local Government",
    "Article XII": "Public Utilities",
    "Article XIII": "Taxation",
    "Article XIII A": "Tax Limitation",
    "Article XIII B": "Government Spending Limitation",
    "Article XIII C": "Voter Approval for Local Tax Levies",
    "Article XIII D": "Assessment and Property-Related Fee Reform",
    "Article XIV": "Labor Relations",
    "Article XV": "Usury",
    "Article XVI": "Public Finance",
    "Article XVII": "Land Ownership and Limitations",
    "Article XVIII": "Amending and Revising the Constitution",
    "Article XIX": "Motor Vehicle Revenues",
    "Article XIX A": "Loans from the Public Transportation Account or Local Transportation Funds",
    "Article XIX B": "Parking Revenue Bonds",
    "Article XIX C": "Enforcement of Certain Provisions",
    "Article XIX D": "Vehicle License Fee Revenues",
    "Article XX": "Miscellaneous Subjects",
    "Article XXI": "Redistricting",
    "Article XXII": "Architectural and Engineering Services",
    "Article XXXIV": "Public Housing Project Law",
    "Article XXXV": "Medical Research",
}

def extract_section_number(text: str) -> str:
    """Extract section number from text."""
    # Look for patterns like "Section 1", "§ 1", "1.", etc.
    patterns = [
        r'Section\s+(\d+(?:\.\d+)?)',
        r'§\s*(\d+(?:\.\d+)?)',
        r'^(\d+(?:\.\d+)?)\s*[\.:]',
        r'\((\d+(?:\.\d+)?)\)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text[:200], re.IGNORECASE)
        if match:
            return match.group(1)

    return ""

def extract_section_title(text: str) -> str:
    """Extract a meaningful title from section text."""
    # Remove common prefixes
    text = re.sub(r'^(Section|§)\s*\d+(?:\.\d+)?\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)

    # Get first sentence or meaningful phrase
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if not line:
            continue
        # Skip navigation/UI text
        if any(kw in line.lower() for kw in ['home', 'ballotpedia', 'sign up', 'newsletter', 'see also', 'external links']):
            continue
        # Skip if it's just punctuation
        if line in [':', '.', ',', ';', '-']:
            continue
        # Get first 100 chars as title
        if len(line) > 10:
            title = line[:100].strip()
            # Remove trailing punctuation
            title = re.sub(r'[\.;:]+$', '', title)
            if title:
                return title

    # Fallback: use first 50 chars of text
    clean_text = re.sub(r'^\s*[:\-\.]+\s*', '', text[:200]).strip()
    if clean_text:
        return clean_text[:50]

    return ""

def fix_article_title(article: Dict[str, Any]) -> str:
    """Fix the title for an article entry."""
    article_num = article.get("article", "")

    # If article number is missing, try to extract from text
    if article_num == "Article" or not article_num:
        clauses = article.get("clauses", [])
        if clauses:
            first_text = clauses[0].get("text", "")
            # Look for article number in text
            article_match = re.search(r'Article\s+([IVXLCDM]+|\d+|[IVXLCDM]+\s*[A-Z]?)', first_text[:500], re.IGNORECASE)
            if article_match:
                article_num = f"Article {article_match.group(1).strip()}"
                article["article"] = article_num  # Update the article field

    # Use mapping if available
    if article_num in ARTICLE_TITLES:
        return f"{article_num} - {ARTICLE_TITLES[article_num]}"

    # Try to extract from clauses
    clauses = article.get("clauses", [])
    if clauses:
        first_text = clauses[0].get("text", "")
        # Look for article title in text
        title_match = re.search(r'Article\s+[IVXLCDM\d]+\s+of\s+the\s+California\s+Constitution\s+is\s+labeled\s+([^\n\.]+)', first_text[:500], re.IGNORECASE)
        if title_match:
            extracted_title = title_match.group(1).strip()
            return f"{article_num} - {extracted_title}"
        # Alternative pattern
        title_match = re.search(r'Article\s+[IVXLCDM\d]+\s*[:\-]?\s*([^\n\.]+)', first_text[:500], re.IGNORECASE)
        if title_match:
            extracted_title = title_match.group(1).strip()
            if len(extracted_title) > 5 and extracted_title.lower() not in ['of', 'the', 'california', 'constitution']:
                return f"{article_num} - {extracted_title}"

    # Default
    return article_num if article_num else "Article"

def fix_clause_title(clause: Dict[str, Any], article_num: str) -> str:
    """Fix the title for a clause entry."""
    current_title = clause.get("title", "")
    text = clause.get("text", "")

    # Skip if title is already meaningful (not just punctuation)
    if current_title and len(current_title) > 2 and current_title not in [':', '.', ',', ';', '-', ':', '.']:
        # Check if it's a section number
        if re.match(r'^\d+(?:\.\d+)?$', current_title.strip()):
            return f"Section {current_title.strip()}"
        # Check if it starts with "Section"
        if current_title.strip().startswith("Section"):
            return current_title.strip()
        # Otherwise keep it if it's meaningful
        if len(current_title) > 5:
            return current_title

    # Try to extract section number
    section_num = extract_section_number(text)
    if section_num:
        return f"Section {section_num}"

    # Try to extract meaningful title
    section_title = extract_section_title(text)
    if section_title:
        return section_title

    # Fallback: use clause number
    clause_num = clause.get("number", "")
    if clause_num:
        return f"Section {clause_num}"

    return "Section"

def fix_constitution_json(input_path: Path, output_path: Path = None):
    """Fix titles in ca_constitution.json."""
    if output_path is None:
        output_path = input_path

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("data", {}).get("california_constitution", {}).get("articles", [])
    print(f"Found {len(articles)} articles")

    fixed_count = 0

    for article in articles:
        article_num = article.get("article", "")

        # Fix article title
        old_title = article.get("title", "")
        new_title = fix_article_title(article)
        if old_title != new_title:
            article["title"] = new_title
            fixed_count += 1
            print(f"Fixed article title: {article_num} - '{old_title[:50]}...' -> '{new_title}'")

        # Fix clause titles
        clauses = article.get("clauses", [])
        for clause in clauses:
            old_clause_title = clause.get("title", "")
            new_clause_title = fix_clause_title(clause, article_num)
            if old_clause_title != new_clause_title:
                clause["title"] = new_clause_title
                fixed_count += 1

    print(f"Fixed {fixed_count} titles")

    # Write output
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Fixed JSON file created: {output_path} ({file_size:.2f} MB)")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    input_path = base_dir / "Data" / "Knowledge" / "ca_constitution.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        exit(1)

    fix_constitution_json(input_path)

