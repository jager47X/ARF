# fetch_cfr_from_scratch.py
"""
Fetch Code of Federal Regulations (CFR) from scratch using GovInfo ECFR XML bulk data.

Downloads all 50 titles of the CFR from:
https://www.govinfo.gov/bulkdata/ECFR/title-X/ECFR-titleX.xml

Parses the XML and converts to JSON format compatible with the RAG system.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("fetch_cfr")

# GovInfo ECFR XML base URL
GOVINFO_ECFR_BASE = "https://www.govinfo.gov/bulkdata/ECFR"

# All 50 CFR titles (some may not exist or be empty)
CFR_TITLES = list(range(1, 51))  # 1-50

# Thread-safe locks
progress_lock = Lock()
save_lock = Lock()
rate_limit_lock = Lock()
last_request_time = [0.0]

def get_headers() -> Dict[str, str]:
    """Get HTTP headers for requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

def fetch_with_retry(url: str, retries: int = 3, min_delay: float = 1.0) -> Optional[requests.Response]:
    """Fetch URL with retry logic and rate limiting."""
    attempt = 0
    while attempt < retries:
        # Rate limiting: ensure minimum delay between requests
        with rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - last_request_time[0]
            if time_since_last < min_delay:
                time.sleep(min_delay - time_since_last)
            last_request_time[0] = time.time()

        try:
            response = requests.get(url, headers=get_headers(), timeout=(60, 180))
            response.raise_for_status()
            if attempt > 0:
                logger.info(f"Successfully fetched {url} on attempt {attempt + 1}")
            return response
        except requests.exceptions.Timeout as e:
            wait_time = min((2 ** attempt) * 5, 300)
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
        except requests.exceptions.ConnectionError as e:
            wait_time = min((2 ** attempt) * 3, 300)
            logger.warning(f"Connection error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            if status_code >= 500 or status_code == 429:
                wait_time = min((2 ** attempt) * 5, 300)
                logger.warning(f"HTTP {status_code} error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
            elif status_code == 404:
                logger.warning(f"Title not found at {url} (404)")
                return None
            else:
                logger.error(f"HTTP {status_code if e.response else 'unknown'} error for {url}: {e}")
                return None
        except Exception as e:
            wait_time = min((2 ** attempt) * 2, 120)
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1

    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

def download_ecfr_xml(title_num: int, output_dir: Path, retries: int = 3, min_delay: float = 1.0) -> Optional[Path]:
    """Download ECFR XML file for a specific title from GovInfo."""
    # ECFR XML URL format: https://www.govinfo.gov/bulkdata/ECFR/title-50/ECFR-title50.xml
    url = f"{GOVINFO_ECFR_BASE}/title-{title_num}/ECFR-title{title_num}.xml"

    output_file = output_dir / f"ecfr_title{title_num:02d}.xml"

    # Skip if already downloaded
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Title {title_num} ECFR XML already exists ({file_size:.2f} MB), skipping download")
        return output_file

    logger.info(f"Downloading ECFR Title {title_num} from GovInfo: {url}")
    response = fetch_with_retry(url, retries=retries, min_delay=min_delay)

    if not response:
        logger.warning(f"Could not download Title {title_num}")
        return None

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(response.content)
        file_size = len(response.content) / (1024 * 1024)  # MB
        logger.info(f"Successfully downloaded Title {title_num} ({file_size:.2f} MB)")
        return output_file
    except Exception as e:
        logger.error(f"Error saving Title {title_num}: {e}")
        return None

def normalize_text(text: str) -> str:
    """Normalize text: remove smart quotes, HTML entities, clean whitespace."""
    if not text:
        return ""

    # Decode HTML entities
    text = unescape(text)

    # Replace smart quotes and typographic characters
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
        '\u00A0': ' ',  # Non-breaking space
        '\u2009': ' ',  # Thin space
        '\u2002': ' ',  # En space
        '\u2003': ' ',  # Em space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()

    return text

def clean_title(title: str, section_num: str = "") -> str:
    """Remove section numbers like '§ 6.1 ' or '(section)' from title."""
    if not title:
        return title

    # Remove section number patterns like "§ 6.1 ", "§6.1 ", "(§ 6.1)", etc.
    title = re.sub(r'§\s*\d+\.?\d*\s*', '', title)
    title = re.sub(r'\(\s*§\s*\d+\.?\d*\s*\)', '', title)
    title = re.sub(r'\(\s*section\s*\)', '', title, flags=re.IGNORECASE)

    # Remove leading/trailing whitespace
    title = re.sub(r'\s+', ' ', title)
    title = title.strip()

    return title

def extract_text_from_element(elem: ET.Element) -> str:
    """Extract text content from XML element, handling nested elements."""
    if elem is None:
        return ""

    text_parts = []

    # Get direct text
    if elem.text:
        text = elem.text.strip()
        if text:
            text_parts.append(text)

    # Get text from all child elements recursively
    for child in elem:
        child_text = extract_text_from_element(child)
        if child_text:
            text_parts.append(child_text)

        # Get tail text (text after the element)
        if child.tail:
            tail_text = child.tail.strip()
            if tail_text:
                text_parts.append(tail_text)

    return " ".join(text_parts).strip()

def parse_ecfr_section(div8_elem: ET.Element, title_num: int, part_num: str = "", chapter: str = "", subchapter_name: str = "") -> Optional[Dict[str, Any]]:
    """Parse an ECFR section from DIV8 XML element."""
    try:
        # ECFR XML structure: DIV8 with N="§ 1.1" contains HEAD and P elements
        section_num_attr = div8_elem.get("N", "")
        if not section_num_attr:
            return None

        # Extract section number (e.g., "§ 1.1" -> "1.1" or just "1.1")
        section_num = section_num_attr.replace("§", "").strip()
        if not section_num:
            return None

        # Extract heading
        head_elem = div8_elem.find(".//HEAD")
        section_title = ""
        if head_elem is not None:
            section_title = normalize_text(extract_text_from_element(head_elem))

        # Clean title to remove section numbers like "§ 6.1 "
        section_title = clean_title(section_title, section_num)

        # Extract text from all P elements
        section_text_parts = []
        for p_elem in div8_elem.findall(".//P"):
            p_text = normalize_text(extract_text_from_element(p_elem))
            if p_text:
                section_text_parts.append(p_text)

        section_text = "\n\n".join(section_text_parts).strip()

        # If no text, try to get any text from the section element
        if not section_text:
            section_text = normalize_text(extract_text_from_element(div8_elem))
            # Remove the heading from the text if it's duplicated
            if section_title and section_text.startswith(section_title):
                section_text = section_text[len(section_title):].strip()

        # Skip if no meaningful content
        if not section_text and not section_title:
            return None

        # Ensure we have at least some content
        if len(section_text) < 10 and not section_title:
            return None

        return {
            "article": f"Title {title_num}",
            "part": f"Part {part_num}" if part_num else "",
            "chapter": chapter if chapter else "",
            "subchapter": subchapter_name if subchapter_name else "",  # Keep as string
            "_subchapter_name": subchapter_name,  # Temporary field for grouping
            "sections": [  # Array named "sections"
                {
                    "section": f"Section {section_num}",
                    "title": section_title or f"Section {section_num}",
                    "text": section_text
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error parsing ECFR section: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def parse_ecfr_xml_file(xml_path: Path, title_num: int) -> List[Dict[str, Any]]:
    """Parse an ECFR XML file and return all sections."""
    if not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        return []

    try:
        logger.info(f"Parsing {xml_path.name}...")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        sections = []

        # ECFR XML structure:
        # DIV1 (Title) -> DIV3 (Chapter) -> DIV4 (Subchapter) -> DIV5 (Part) -> DIV8 (Section)

        # Track current part, chapter, and subchapter as we traverse the tree
        current_part = ""
        current_chapter = ""
        current_subchapter = ""

        def process_element(elem: ET.Element, part_num: str, chapter: str, subchapter: str):
            """Recursively process elements, tracking part, chapter, and subchapter context."""
            nonlocal current_part, current_chapter, current_subchapter

            tag = elem.tag if isinstance(elem.tag, str) else elem.tag.split('}')[-1] if '}' in str(elem.tag) else str(elem.tag)

            # Update context based on element type
            if tag == "DIV5":
                # This is a Part
                part_head = elem.find(".//HEAD")
                if part_head is not None:
                    part_text = normalize_text(extract_text_from_element(part_head))
                    if "PART" in part_text.upper():
                        # Extract part number (e.g., "PART 1" -> "1")
                        part_match = re.search(r'PART\s+(\d+[A-Z]?)', part_text, re.IGNORECASE)
                        if part_match:
                            part_num = part_match.group(1)
                            current_part = part_num
            elif tag == "DIV3":
                # This is a Chapter
                chapter_head = elem.find(".//HEAD")
                if chapter_head is not None:
                    chapter_text = normalize_text(extract_text_from_element(chapter_head))
                    if chapter_text:
                        chapter = chapter_text.strip()
                        current_chapter = chapter
            elif tag == "DIV4":
                # This is a Subchapter
                subchapter_head = elem.find(".//HEAD")
                if subchapter_head is not None:
                    subchapter_text = normalize_text(extract_text_from_element(subchapter_head))
                    if subchapter_text:
                        subchapter = subchapter_text.strip()
                        current_subchapter = subchapter
            elif tag == "DIV8":
                # This is a section - parse it with current context
                parsed_section = parse_ecfr_section(elem, title_num, part_num, chapter, subchapter)
                if parsed_section:
                    sections.append(parsed_section)
                return  # Don't recurse into DIV8 children

            # Recurse into children
            for child in elem:
                process_element(child, part_num, chapter, subchapter)

        # Start processing from root
        process_element(root, "", "", "")

        logger.info(f"Successfully parsed {len(sections)} sections from Title {title_num} ECFR XML")
        return sections

    except ET.ParseError as e:
        logger.error(f"XML parse error in {xml_path.name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing {xml_path.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []

def process_title(title_num: int, xml_dir: Path, min_delay: float = 1.0) -> List[Dict[str, Any]]:
    """Process a single CFR title: download XML if needed, then parse it."""
    logger.info(f"Processing Title {title_num}...")

    # Download XML if needed
    xml_file = download_ecfr_xml(title_num, xml_dir, retries=3, min_delay=min_delay)
    if not xml_file:
        logger.warning(f"Could not download Title {title_num} ECFR XML")
        return []

    # Parse XML
    sections = parse_ecfr_xml_file(xml_file, title_num)
    return sections

def extract_title_number(article: str) -> int:
    """Extract title number from article string (e.g., 'Title 50' -> 50)."""
    try:
        if article.startswith("Title "):
            return int(article.split()[1])
    except (ValueError, IndexError):
        pass
    return 999  # Put unknown titles at the end

def group_sections_by_subchapter(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group sections by subchapter, combining sections in the same subchapter into one entry."""
    from collections import defaultdict

    # Group sections by (article, part, chapter, subchapter_name)
    grouped = defaultdict(list)

    for section in sections:
        article = section.get("article", "")
        part = section.get("part", "")
        chapter = section.get("chapter", "")
        subchapter_name = section.get("_subchapter_name", "")  # Temporary field for grouping
        sections_array = section.get("sections", [])  # Array is now named "sections"

        # Create grouping key including subchapter name
        group_key = (article, part, chapter, subchapter_name)

        # Extract section data from sections array and add to group
        for section_item in sections_array:
            grouped[group_key].append(section_item)

    # Convert grouped data into final structure
    result = []
    for (article, part, chapter, subchapter_name), sections_items in grouped.items():
        result.append({
            "article": article,
            "part": part,
            "chapter": chapter,
            "subchapter": subchapter_name,  # Keep as string
            "sections": sections_items  # Array named "sections"
        })

    return result

def save_progress(sections: List[Dict[str, Any]], output_path: Path):
    """Save current progress to JSON file (thread-safe), organized by title 1-50 and grouped by subchapter."""
    with save_lock:
        # Group sections by subchapter
        grouped_sections = group_sections_by_subchapter(sections)

        # Organize by title number
        from collections import defaultdict
        title_groups = defaultdict(list)

        for section in grouped_sections:
            article = section.get("article", "")
            title_num = extract_title_number(article)
            title_groups[title_num].append(section)

        # Sort by title number and flatten
        organized_sections = []
        for title_num in sorted(title_groups.keys()):
            organized_sections.extend(title_groups[title_num])

        output_data = {
            "data": {
                "code_of_federal_regulations": {
                    "regulations": organized_sections
                }
            }
        }

        # Write to temporary file first, then rename (atomic write)
        temp_path = output_path.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        temp_path.replace(output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Progress saved: {len(organized_sections)} grouped entries ({file_size:.2f} MB)")

def main():
    """Main function to fetch all CFR titles from GovInfo ECFR XML."""
    parser = argparse.ArgumentParser(description="Fetch Code of Federal Regulations from GovInfo ECFR XML")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel title fetching (default: 4)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Minimum delay between requests in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--title",
        type=int,
        default=None,
        help="Fetch only a specific title number (for testing)"
    )
    parser.add_argument(
        "--from-title",
        type=int,
        default=None,
        help="Start processing from this title number (e.g., 50 to process titles 50 down to 1)"
    )
    parser.add_argument(
        "--to-title",
        type=int,
        default=None,
        help="End processing at this title number (default: 1 when using --from-title)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading XML files and only parse existing files"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = base_dir / "Data" / "Knowledge" / "code_of_federal_regulations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create XML download directory
    xml_dir = script_dir / "cfr_xml_temp"
    xml_dir.mkdir(parents=True, exist_ok=True)

    # Determine which titles to process
    if args.title:
        titles_to_process = [args.title]
        logger.info(f"Processing only Title {args.title}")
    elif args.from_title:
        # Process from title X down to Y
        from_title = args.from_title
        to_title = args.to_title if args.to_title else 1
        if from_title >= to_title:
            titles_to_process = list(range(from_title, to_title - 1, -1))
        else:
            titles_to_process = list(range(from_title, to_title + 1))
        logger.info(f"Processing titles from {from_title} to {to_title}: {titles_to_process}")
    else:
        titles_to_process = CFR_TITLES
        logger.info(f"Processing all {len(titles_to_process)} CFR titles...")

    all_sections = []
    num_workers = max(1, args.workers)
    min_delay = max(0.1, args.delay)

    logger.info(f"Using {num_workers} workers with {min_delay}s minimum delay")
    if args.skip_download:
        logger.info("Skipping downloads - will only parse existing XML files")
    logger.info("This may take a significant amount of time.")
    logger.info("Progress will be saved periodically.")
    logger.info(f"Fetching from: {GOVINFO_ECFR_BASE}")

    start_time = time.time()
    processed_count = [0]

    def update_progress():
        with progress_lock:
            processed_count[0] += 1
            current = processed_count[0]
            if current % 5 == 0 or current == len(titles_to_process):
                logger.info(f"Processed {current}/{len(titles_to_process)} titles")

    # If skip_download, only parse existing files
    if args.skip_download:
        logger.info("Parsing existing XML files only...")
        for title_num in titles_to_process:
            xml_file = xml_dir / f"ecfr_title{title_num:02d}.xml"
            if xml_file.exists():
                sections = parse_ecfr_xml_file(xml_file, title_num)
                if sections:
                    all_sections.extend(sections)
                    logger.info(f"Title {title_num}: Parsed {len(sections)} sections (Total: {len(all_sections)})")
                else:
                    logger.warning(f"Title {title_num}: No sections found")
            else:
                logger.warning(f"Title {title_num}: XML file not found, skipping")

            # Save progress periodically
            if len(all_sections) > 0 and len(all_sections) % 1000 == 0:
                save_progress(all_sections, output_path)
    else:
        # Download and parse in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_title = {
                executor.submit(process_title, title_num, xml_dir, min_delay): title_num
                for title_num in titles_to_process
            }

            for future in as_completed(future_to_title):
                title_num = future_to_title[future]
                try:
                    sections = future.result()
                    if sections:
                        all_sections.extend(sections)
                        logger.info(f"Title {title_num}: Added {len(sections)} sections (Total: {len(all_sections)})")
                    else:
                        logger.warning(f"Title {title_num}: No sections found")
                    update_progress()

                    # Save progress periodically
                    if len(all_sections) > 0 and len(all_sections) % 1000 == 0:
                        save_progress(all_sections, output_path)
                except Exception as e:
                    logger.error(f"Error processing Title {title_num}: {e}", exc_info=True)
                    update_progress()

    # Final save
    save_progress(all_sections, output_path)

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logger.info(f"\n{'='*60}")
    logger.info("Processing complete!")
    logger.info(f"Total sections: {len(all_sections)}")
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
    logger.info(f"Output file: {output_path}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

