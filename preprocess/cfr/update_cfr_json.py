# update_cfr_json.py
"""
Update CFR JSON file to match the new MongoDB structure:
- Replace 'clauses' with 'sections'
- Extract chapter number only
- Move text to title
- Ensure part field is present
"""
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_cfr_json")

BASE_DIR = Path(__file__).resolve().parents[2]
CFR_DOCUMENT_PATH = str(BASE_DIR / "Data/Knowledge/code_of_federal_regulations.json")

def clean_title(title: str) -> str:
    """Remove section numbers like '§ 1.1' or '(section)' from title."""
    if not title:
        return title
    title = re.sub(r'§\s*\d+\.?\d*\s*', '', title)
    title = re.sub(r'\(\s*section\s*\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def extract_article_number(article: str) -> str:
    """Extract number from 'Title X' -> 'X'"""
    if not article:
        return ""
    # Extract number from "Title 50" -> "50"
    match = re.search(r'Title\s+(\d+)', str(article), re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any number
    match = re.search(r'(\d+)', str(article))
    return match.group(1) if match else ""

def extract_part_number(part: str) -> str:
    """Extract number from 'Part X' -> 'X'"""
    if not part:
        return ""
    # Extract number from "Part 260" -> "260"
    match = re.search(r'Part\s+(\d+)', str(part), re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any number
    match = re.search(r'(\d+)', str(part))
    return match.group(1) if match else ""

def extract_section_number(section: str) -> str:
    """Extract section number from 'Section X.Y' -> 'X.Y'"""
    if not section:
        return ""
    # Extract section number from "Section 260.20" -> "260.20"
    match = re.search(r'Section\s+(\d+\.?\d*)', str(section), re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: extract any number pattern
    match = re.search(r'(\d+\.\d+)', str(section))
    if match:
        return match.group(1)
    return ""

def extract_chapter_from_section(section: str, part_num: str) -> str:
    """Extract chapter from section number (e.g., 'Section 260.20' with part '260' -> '2')"""
    if not section or not part_num:
        return ""
    # Extract section number first
    section_num = extract_section_number(section)
    if not section_num:
        return ""

    # If section is "260.20" and part is "260", extract "2" (first digit after part)
    # Pattern: part_num.chapter.section -> extract chapter
    if section_num.startswith(part_num + "."):
        remaining = section_num[len(part_num) + 1:]  # Get "20" from "260.20"
        # Extract first digit(s) as chapter
        match = re.search(r'^(\d+)', remaining)
        if match:
            return match.group(1)

    return ""

def extract_chapter_number(chapter: str) -> str:
    """Extract chapter number from chapter text (e.g., 'Chapter 1 - Title' -> '1')."""
    if not chapter:
        return ""
    # Extract number from patterns like "Chapter 1", "Ch. 1", "1", etc.
    match = re.search(r'(\d+)', str(chapter))
    return match.group(1) if match else ""

def normalize_to_hierarchy(entry_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single entry (not yet grouped).
    Returns normalized entry with extracted numbers and sections.
    """
    article_text = entry_obj.get("article", "")
    part_text = entry_obj.get("part", "")
    chapter_text = entry_obj.get("chapter", "")
    section_text = entry_obj.get("section", "")
    original_title = entry_obj.get("title", "")
    clauses = entry_obj.get("clauses")
    text = entry_obj.get("text", "")

    # Extract numbers
    article_num = extract_article_number(article_text)
    part_num = extract_part_number(part_text)
    section_num = extract_section_number(section_text)

    # Extract chapter: try from section number first, then from chapter text
    chapter_num = extract_chapter_from_section(section_text, part_num)
    if not chapter_num:
        chapter_num = extract_chapter_number(chapter_text)

    # Convert clauses to sections (with distinct 'title' and 'text' fields)
    sections = []
    if clauses and isinstance(clauses, list):
        for c in clauses:
            clause_title = c.get("title", "")
            clause_text = c.get("text", "")

            # Clean title if present
            if clause_title:
                clause_title = clean_title(clause_title)

            # Section title: only use clause title (never use text as title)
            section_title = clause_title if clause_title else ""
            # Section text: only use clause text (never use title as text)
            section_text_content = clause_text if clause_text else ""

            # Only add section if we have at least title or text
            if section_text_content or section_title:
                sections.append({
                    "number": c.get("number"),
                    "title": section_title,
                    "text": section_text_content,
                })
    elif text:
        # If flat structure with text, use empty title and text as content
        sections.append({
            "number": 1,
            "title": "",
            "text": text,
        })
    elif original_title:
        # If only title, use it as title and empty text
        clean_t = clean_title(original_title)
        sections.append({
            "number": 1,
            "title": clean_t,
            "text": "",
        })

    # Use original title as document title (never use section text)
    if original_title:
        doc_title = clean_title(original_title)
    else:
        # Fallback: construct title from metadata
        title_parts = []
        if article_num:
            title_parts.append(f"Title {article_num}")
        if part_num:
            title_parts.append(f"Part {part_num}")
        if chapter_num:
            title_parts.append(f"Chapter {chapter_num}")
        if section_num:
            title_parts.append(f"Section {section_num}")
        doc_title = " - ".join(title_parts) if title_parts else ""

    return {
        "article": article_num,
        "part": part_num,
        "chapter": chapter_num if chapter_num else "",
        "section": section_text,  # Keep full section identifier
        "title": doc_title,
        "sections": sections  # Sections array with number, title, and text
    }

def group_and_aggregate_sections(normalized_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group entries by (article, part, chapter) and aggregate sections.
    Returns one document per group with all sections sorted by number.
    """
    grouped = {}

    for entry in normalized_entries:
        # Create grouping key
        key = (entry.get("article", ""), entry.get("part", ""), entry.get("chapter", ""))

        if key not in grouped:
            # Initialize group with first entry's metadata
            # Use original title from first entry, or construct from metadata
            title = entry.get("title", "")
            if not title:
                # Construct title from metadata
                title_parts = []
                if entry.get("article"):
                    title_parts.append(f"Title {entry['article']}")
                if entry.get("part"):
                    title_parts.append(f"Part {entry['part']}")
                if entry.get("chapter"):
                    title_parts.append(f"Chapter {entry['chapter']}")
                title = " - ".join(title_parts) if title_parts else ""

            grouped[key] = {
                "article": entry.get("article", ""),
                "part": entry.get("part", ""),
                "chapter": entry.get("chapter", ""),
                "title": title,
                "sections": []
            }

        # Add sections from this entry to the group
        entry_sections = entry.get("sections", [])
        grouped[key]["sections"].extend(entry_sections)

    # Sort sections by number for each group
    result = []
    for key, doc in grouped.items():
        # Sort sections by number (handle both int and string numbers)
        doc["sections"].sort(key=lambda x: (
            float(x.get("number", 0)) if isinstance(x.get("number"), (int, float))
            else float(str(x.get("number", "0")).split(".")[0]) if str(x.get("number", "0")).replace(".", "").isdigit()
            else 0
        ))
        result.append(doc)

    return result

def update_json_file():
    """Update the CFR JSON file with the new structure."""
    logger.info(f"Reading CFR JSON from {CFR_DOCUMENT_PATH}...")

    if not os.path.exists(CFR_DOCUMENT_PATH):
        logger.error(f"File not found: {CFR_DOCUMENT_PATH}")
        return

    # Read current JSON
    with open(CFR_DOCUMENT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get the regulations/titles array
    cfr_data = data.get("data", {}).get("code_of_federal_regulations", {})
    items = cfr_data.get("titles") or cfr_data.get("regulations", [])

    if not items:
        logger.warning("No 'titles' or 'regulations' found in JSON.")
        return

    logger.info(f"Found {len(items)} entries to transform...")

    # Normalize all entries
    normalized_entries = [normalize_to_hierarchy(item) for item in items]
    logger.info(f"Normalized {len(normalized_entries)} entries.")

    # Group entries by (article, part, chapter) and aggregate sections
    transformed_items = group_and_aggregate_sections(normalized_entries)
    logger.info(f"Grouped into {len(transformed_items)} documents (aggregated sections).")

    # Update the data structure
    if "titles" in cfr_data:
        cfr_data["titles"] = transformed_items
    elif "regulations" in cfr_data:
        cfr_data["regulations"] = transformed_items
    else:
        cfr_data["titles"] = transformed_items

    # Write back to file
    logger.info(f"Writing updated JSON to {CFR_DOCUMENT_PATH}...")
    temp_path = CFR_DOCUMENT_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Replace original file
    os.replace(temp_path, CFR_DOCUMENT_PATH)

    file_size = os.path.getsize(CFR_DOCUMENT_PATH) / (1024 * 1024)
    logger.info(f"JSON file updated successfully! File size: {file_size:.2f} MB")
    logger.info(f"Transformed {len(transformed_items)} entries")

if __name__ == "__main__":
    update_json_file()

