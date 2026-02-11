# fetch_cfr.py
"""
Fetch Code of Federal Regulations (CFR) from eCFR API (ecfr.gov/api/versioner/v1).

IMPORTANT NOTE: The eCFR API endpoint (ecfr.gov/api/versioner/v1) appears to be DEPRECATED
as of 2024/2025. This script will attempt to use it, but may fail.

Alternative approaches:
1. Use GovInfo bulk XML data (similar to parse_usc_xml.py) - recommended
2. Parse eCFR HTML pages directly (more complex, requires web scraping)
3. Use official CFR XML from GPO/GovInfo

Downloads and processes all 50 titles of the CFR from the official eCFR API (if available).
"""
import os
import sys
import json
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("fetch_cfr")

# eCFR API base URL (Note: API may be deprecated, will fall back to GovInfo XML)
ECFR_API_BASE = "https://ecfr.gov/api/versioner/v1"
GOVINFO_ECFR_BASE = "https://www.govinfo.gov/bulkdata/CFR"
GOVINFO_ECFR_XML_BASE = "https://www.govinfo.gov/bulkdata/ECFR"

# Thread-safe locks
progress_lock = Lock()
save_lock = Lock()
rate_limit_lock = Lock()
last_request_time = [0.0]

# All 50 CFR titles (some may not exist or be empty)
CFR_TITLES = list(range(1, 51))  # 1-50

def get_headers() -> Dict[str, str]:
    """Get HTTP headers for requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

def fetch_with_retry(url: str, params: Optional[Dict] = None, retries: int = 3, min_delay: float = 0.5) -> Optional[requests.Response]:
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
            response = requests.get(url, params=params, headers=get_headers(), timeout=(30, 60))
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

def get_latest_version() -> Optional[str]:
    """Get the latest CFR version identifier from the API.
    Note: The eCFR API may be deprecated. This function will return None if the API is unavailable.
    """
    url = f"{ECFR_API_BASE}/versions"
    response = fetch_with_retry(url, retries=1)  # Only try once since API is likely deprecated
    
    if not response:
        logger.warning("eCFR API is not available (likely deprecated). Will use GovInfo XML fallback.")
        return None
    
    # Check if API returned deprecated error
    if response.status_code == 501:
        logger.warning("eCFR API returned 501 (Deprecated). Will use GovInfo XML fallback.")
        return None
    
    try:
        versions = response.json()
        if isinstance(versions, list) and len(versions) > 0:
            # Versions are typically sorted with latest first
            latest = versions[0]
            version_id = latest.get("identifier") or latest.get("date")
            logger.info(f"Latest CFR version: {version_id}")
            return version_id
        else:
            logger.warning("No versions found in API response. Will use GovInfo XML fallback.")
            return None
    except Exception as e:
        logger.warning(f"Error parsing versions: {e}. Will use GovInfo XML fallback.")
        return None

def fetch_title_structure(version_id: str, title_num: int) -> Optional[Dict[str, Any]]:
    """Fetch the structure (parts and sections) of a CFR title."""
    url = f"{ECFR_API_BASE}/versions/{version_id}/titles/{title_num}"
    response = fetch_with_retry(url)
    
    if not response:
        return None
    
    try:
        return response.json()
    except Exception as e:
        logger.error(f"Error parsing title {title_num} structure: {e}")
        return None

def fetch_title_content(version_id: str, title_num: int) -> Optional[Dict[str, Any]]:
    """Fetch the full content of a CFR title."""
    url = f"{ECFR_API_BASE}/versions/{version_id}/titles/{title_num}/full"
    response = fetch_with_retry(url)
    
    if not response:
        return None
    
    try:
        return response.json()
    except Exception as e:
        logger.error(f"Error parsing title {title_num} content: {e}")
        return None

def extract_text_from_node(node: Dict[str, Any]) -> str:
    """Extract text content from a CFR API node structure."""
    if isinstance(node, str):
        return node.strip()
    
    if isinstance(node, dict):
        # Try common text fields
        text = node.get("text") or node.get("content") or node.get("body") or node.get("label")
        if text:
            return str(text).strip()
        
        # If it's a structured node, try to extract from children
        children = node.get("children") or node.get("subparts") or node.get("sections") or []
        if children:
            text_parts = []
            for child in children:
                child_text = extract_text_from_node(child)
                if child_text:
                    text_parts.append(child_text)
            return "\n".join(text_parts)
    
    if isinstance(node, list):
        text_parts = []
        for item in node:
            item_text = extract_text_from_node(item)
            if item_text:
                text_parts.append(item_text)
        return "\n".join(text_parts)
    
    return ""

def parse_cfr_section(section_data: Dict[str, Any], title_num: int, part_num: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Parse a CFR section from API response into our format."""
    try:
        # Extract section number
        section_num = section_data.get("label") or section_data.get("number") or section_data.get("section")
        if not section_num:
            # Try to extract from structure
            structure = section_data.get("structure") or {}
            section_num = structure.get("label") or structure.get("section")
        
        if not section_num:
            logger.warning(f"Could not find section number in section data: {section_data.keys()}")
            return None
        
        # Extract title/heading
        title = section_data.get("title") or section_data.get("heading") or section_data.get("label")
        if not title:
            title = f"Section {section_num}"
        
        # Extract text content
        text = extract_text_from_node(section_data)
        
        # If no text found, try content field
        if not text or len(text.strip()) < 10:
            text = section_data.get("content") or section_data.get("text") or ""
            if isinstance(text, dict):
                text = extract_text_from_node(text)
        
        # Clean up text
        if text:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            text = text.strip()
        
        # Validate we have meaningful content
        if not text or len(text) < 10:
            logger.debug(f"Section {section_num} has insufficient content, skipping")
            return None
        
        # Determine part number
        if not part_num:
            part_num = section_data.get("part") or ""
            if not part_num:
                # Try to extract from structure
                structure = section_data.get("structure") or {}
                part_num = structure.get("part") or ""
        
        return {
            "article": f"Title {title_num}",
            "part": f"Part {part_num}" if part_num else "",
            "section": f"Section {section_num}",
            "title": str(title).strip(),
            "clauses": [
                {
                    "number": 1,
                    "title": str(title).strip(),
                    "text": text
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error parsing section: {e}")
        return None

def process_title(version_id: str, title_num: int, min_delay: float = 0.5) -> List[Dict[str, Any]]:
    """Process a single CFR title and return all sections."""
    logger.info(f"Processing Title {title_num}...")
    sections = []
    
    # Fetch full title content
    title_data = fetch_title_content(version_id, title_num)
    
    if not title_data:
        logger.warning(f"Could not fetch Title {title_num}")
        return sections
    
    # The API structure may vary, so we need to handle different response formats
    # Common structures:
    # 1. title_data has "parts" array
    # 2. title_data has "structure" with nested parts/sections
    # 3. title_data has "children" array
    
    def extract_sections_from_node(node: Dict[str, Any], part_num: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recursively extract sections from a node."""
        sections_list = []
        
        # Check if this node is a section
        if node.get("type") == "section" or "section" in str(node.get("label", "")).lower():
            section = parse_cfr_section(node, title_num, part_num)
            if section:
                sections_list.append(section)
        
        # Check for nested parts
        parts = node.get("parts") or node.get("children") or []
        if isinstance(parts, list):
            for part in parts:
                # Extract part number
                current_part = part.get("label") or part.get("number") or part_num
                if current_part and current_part != part_num:
                    part_num = current_part
                
                # Recursively extract sections
                sections_list.extend(extract_sections_from_node(part, part_num))
        
        # Check for sections array
        sections_array = node.get("sections") or []
        if isinstance(sections_array, list):
            for section_data in sections_array:
                section = parse_cfr_section(section_data, title_num, part_num)
                if section:
                    sections_list.append(section)
        
        return sections_list
    
    # Extract sections from title data
    sections = extract_sections_from_node(title_data)
    
    # If no sections found, try alternative structure
    if not sections:
        # Try to find sections in structure field
        structure = title_data.get("structure") or {}
        sections = extract_sections_from_node(structure)
    
    # If still no sections, try direct children
    if not sections:
        children = title_data.get("children") or []
        for child in children:
            sections.extend(extract_sections_from_node(child))
    
    logger.info(f"Title {title_num}: Found {len(sections)} sections")
    return sections

def download_ecfr_xml(title_num: int, output_dir: Path, retries: int = 3) -> Optional[Path]:
    """Download ECFR XML file for a specific title from GovInfo ECFR bulk data."""
    # ECFR XML URL format: https://www.govinfo.gov/bulkdata/ECFR/title-50/ECFR-title50.xml
    url = f"{GOVINFO_ECFR_XML_BASE}/title-{title_num}/ECFR-title{title_num}.xml"
    
    output_file = output_dir / f"ecfr_title{title_num:02d}.xml"
    
    # Skip if already downloaded
    if output_file.exists():
        logger.info(f"Title {title_num} ECFR XML already exists, skipping download")
        return output_file
    
    for attempt in range(retries):
        try:
            if attempt > 0:
                wait_time = (2 ** attempt) * 3
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                time.sleep(1.0)  # Initial delay
            
            logger.info(f"Downloading ECFR Title {title_num} from GovInfo: {url} (attempt {attempt + 1})")
            response = requests.get(url, timeout=(60, 120), headers=get_headers())
            
            if response.status_code == 200:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(response.content)
                logger.info(f"Successfully downloaded Title {title_num} ({len(response.content)} bytes)")
                return output_file
            elif response.status_code == 404:
                logger.warning(f"Title {title_num} not found at {url}")
                return None
            else:
                logger.warning(f"HTTP {response.status_code} for Title {title_num} at {url}")
                if attempt < retries - 1:
                    time.sleep((2 ** attempt) * 5)
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout downloading Title {title_num} (attempt {attempt + 1})")
            if attempt < retries - 1:
                time.sleep((2 ** attempt) * 5)
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error downloading Title {title_num} (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep((2 ** attempt) * 5)
        except Exception as e:
            logger.warning(f"Error downloading Title {title_num}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    
    logger.warning(f"Failed to download Title {title_num} after {retries} attempts")
    return None

def download_cfr_xml(title_num: int, year: int, output_dir: Path, retries: int = 3) -> Optional[Path]:
    """Download CFR XML file for a specific title and year from GovInfo."""
    # GovInfo CFR XML URL patterns (try different formats)
    url_patterns = [
        f"{GOVINFO_ECFR_BASE}/{year}/title-{title_num}/CFR-{year}-title{title_num}.xml",
        f"{GOVINFO_ECFR_BASE}/{year}/title-{title_num}/CFR-{year}-title-{title_num}.xml",
        f"{GOVINFO_ECFR_BASE}/{year}/title{title_num}/CFR-{year}-title{title_num}.xml",
    ]
    
    output_file = output_dir / f"cfr{title_num:02d}_{year}.xml"
    
    # Skip if already downloaded
    if output_file.exists():
        logger.info(f"Title {title_num} XML already exists, skipping download")
        return output_file
    
    # Try each URL pattern
    for url in url_patterns:
        for attempt in range(retries):
            try:
                if attempt > 0:
                    wait_time = (2 ** attempt) * 3
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    time.sleep(1.0)  # Initial delay
                
                logger.info(f"Downloading CFR Title {title_num} ({year}) from GovInfo: {url} (attempt {attempt + 1})")
                response = requests.get(url, timeout=(60, 120), headers=get_headers())
                
                if response.status_code == 200:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_file.write_bytes(response.content)
                    logger.info(f"Successfully downloaded Title {title_num} ({len(response.content)} bytes)")
                    return output_file
                elif response.status_code == 404:
                    logger.debug(f"Title {title_num} not found at {url}")
                    break  # Try next URL pattern
                else:
                    logger.warning(f"HTTP {response.status_code} for Title {title_num} at {url}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout downloading Title {title_num} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep((2 ** attempt) * 5)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error downloading Title {title_num} (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep((2 ** attempt) * 5)
            except Exception as e:
                logger.warning(f"Error downloading Title {title_num}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
    
    logger.warning(f"Failed to download Title {title_num} for year {year} after trying all URL patterns")
    return None

def extract_text_from_xml_element(elem: ET.Element) -> str:
    """Extract text content from XML element, handling nested elements."""
    if elem is None:
        return ""
    
    text_parts = []
    if elem.text:
        text_parts.append(elem.text.strip())
    
    for child in elem:
        child_text = extract_text_from_xml_element(child)
        if child_text:
            text_parts.append(child_text)
        if child.tail:
            text_parts.append(child.tail.strip())
    
    return " ".join(text_parts).strip()

def parse_cfr_xml_section(section_elem: ET.Element, title_num: int, part_num: str = "") -> Optional[Dict[str, Any]]:
    """Parse a CFR section from XML element."""
    try:
        # CFR XML structure: SECTION -> SECTNO (section number), SUBJECT (title), content
        sectno_elem = section_elem.find(".//SECTNO")
        section_num = sectno_elem.text.strip() if sectno_elem is not None and sectno_elem.text else ""
        
        subject_elem = section_elem.find(".//SUBJECT")
        section_title = subject_elem.text.strip() if subject_elem is not None and subject_elem.text else ""
        
        # Extract text from the section (excluding SECTNO and SUBJECT)
        section_text = ""
        for elem in section_elem:
            if elem.tag not in ["SECTNO", "SUBJECT"]:
                text = extract_text_from_xml_element(elem)
                if text:
                    section_text += text + " "
        section_text = section_text.strip()
        
        # Try to get part number from parent structure
        if not part_num:
            part_elem = section_elem.find(".//PARTNO")
            if part_elem is not None and part_elem.text:
                part_num = part_elem.text.strip()
        
        # Skip if no meaningful content
        if not section_text and not section_title:
            return None
        
        return {
            "article": f"Title {title_num}",
            "part": f"Part {part_num}" if part_num else "",
            "section": f"Section {section_num}",
            "title": section_title or f"Section {section_num}",
            "clauses": [
                {
                    "number": 1,
                    "title": section_title or f"Section {section_num}",
                    "text": section_text
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error parsing CFR XML section: {e}")
        return None

def parse_ecfr_xml_section(div8_elem: ET.Element, title_num: int, part_num: str = "", chapter: str = "") -> Optional[Dict[str, Any]]:
    """Parse an ECFR section from DIV8 XML element."""
    try:
        # ECFR XML structure: DIV8 with N="§ 1.1" contains HEAD and P elements
        section_num_attr = div8_elem.get("N", "")
        if not section_num_attr or not section_num_attr.startswith("§"):
            return None
        
        # Extract section number (e.g., "§ 1.1" -> "1.1")
        section_num = section_num_attr.replace("§", "").strip()
        
        # Extract heading
        head_elem = div8_elem.find(".//HEAD")
        section_title = ""
        if head_elem is not None:
            section_title = extract_text_from_xml_element(head_elem).strip()
        
        # Extract text from all P elements
        section_text_parts = []
        for p_elem in div8_elem.findall(".//P"):
            p_text = extract_text_from_xml_element(p_elem)
            if p_text:
                section_text_parts.append(p_text)
        
        section_text = "\n\n".join(section_text_parts).strip()
        
        # Clean up text
        if section_text:
            section_text = re.sub(r'\s+', ' ', section_text)
            section_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', section_text)
            section_text = section_text.strip()
        
        # Skip if no meaningful content
        if not section_text and not section_title:
            return None
        
        return {
            "article": f"Title {title_num}",
            "part": f"Part {part_num}" if part_num else "",
            "chapter": chapter if chapter else "",
            "section": f"Section {section_num}",
            "title": section_title or f"Section {section_num}",
            "clauses": [
                {
                    "number": 1,
                    "title": section_title or f"Section {section_num}",
                    "text": section_text
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error parsing ECFR XML section: {e}")
        return None

def parse_ecfr_xml_file(xml_path: Path, title_num: int) -> List[Dict[str, Any]]:
    """Parse an ECFR XML file and return all sections."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        sections = []
        
        # ECFR XML structure:
        # DIV1 (Title) -> DIV3 (Chapter) -> DIV4 (Subchapter) -> DIV5 (Part) -> DIV8 (Section)
        
        # Track current part and chapter as we traverse the tree
        current_part = ""
        current_chapter = ""
        
        def process_element(elem: ET.Element, part_num: str, chapter: str):
            """Recursively process elements, tracking part and chapter context."""
            nonlocal current_part, current_chapter
            
            # Update context based on element type
            if elem.tag == "DIV5":
                part_head = elem.find(".//HEAD")
                if part_head is not None:
                    part_text = extract_text_from_xml_element(part_head)
                    if "PART" in part_text.upper():
                        part_match = re.search(r'PART\s+(\d+)', part_text, re.IGNORECASE)
                        if part_match:
                            part_num = part_match.group(1)
                            current_part = part_num
            elif elem.tag == "DIV3":
                chapter_head = elem.find(".//HEAD")
                if chapter_head is not None:
                    chapter_text = extract_text_from_xml_element(chapter_head)
                    if chapter_text:
                        chapter = chapter_text.strip()
                        current_chapter = chapter
            elif elem.tag == "DIV8":
                # This is a section - parse it with current context
                parsed_section = parse_ecfr_xml_section(elem, title_num, part_num, chapter)
                if parsed_section:
                    sections.append(parsed_section)
                return  # Don't recurse into DIV8 children
            
            # Recurse into children
            for child in elem:
                process_element(child, part_num, chapter)
        
        # Start processing from root
        process_element(root, "", "")
        
        logger.info(f"Successfully parsed {len(sections)} sections from Title {title_num} ECFR XML")
        return sections
        
    except ET.ParseError as e:
        logger.error(f"XML parse error in {xml_path.name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing {xml_path.name}: {e}")
        return []

def parse_cfr_xml_file(xml_path: Path, title_num: int) -> List[Dict[str, Any]]:
    """Parse a CFR XML file and return all sections."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        sections = []
        
        # Find all SECTION elements
        section_elems = root.findall(".//SECTION")
        logger.info(f"Found {len(section_elems)} sections in Title {title_num} XML")
        
        # Try to get part number from structure
        part_num = ""
        part_elem = root.find(".//PARTNO")
        if part_elem is not None and part_elem.text:
            part_num = part_elem.text.strip()
        
        for section_elem in section_elems:
            parsed_section = parse_cfr_xml_section(section_elem, title_num, part_num)
            if parsed_section:
                sections.append(parsed_section)
        
        logger.info(f"Successfully parsed {len(sections)} sections from Title {title_num}")
        return sections
        
    except ET.ParseError as e:
        logger.error(f"XML parse error in {xml_path.name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing {xml_path.name}: {e}")
        return []

def process_title_from_ecfr_xml(title_num: int, xml_dir: Path, min_delay: float = 0.5) -> List[Dict[str, Any]]:
    """Process a CFR title from GovInfo ECFR XML."""
    logger.info(f"Processing Title {title_num} from GovInfo ECFR XML...")
    
    # Download XML if needed
    xml_file = download_ecfr_xml(title_num, xml_dir, retries=3)
    if not xml_file:
        logger.warning(f"Could not download Title {title_num} ECFR XML")
        return []
    
    # Parse XML
    sections = parse_ecfr_xml_file(xml_file, title_num)
    return sections

def process_title_from_govinfo(title_num: int, year: int, xml_dir: Path, min_delay: float = 0.5) -> List[Dict[str, Any]]:
    """Process a CFR title from GovInfo XML."""
    logger.info(f"Processing Title {title_num} from GovInfo XML ({year})...")
    
    # Download XML if needed
    xml_file = download_cfr_xml(title_num, year, xml_dir, retries=3)
    if not xml_file:
        logger.warning(f"Could not download Title {title_num} XML")
        return []
    
    # Parse XML
    sections = parse_cfr_xml_file(xml_file, title_num)
    return sections

def extract_title_number(article: str) -> int:
    """Extract title number from article string (e.g., 'Title 50' -> 50)."""
    try:
        if article.startswith("Title "):
            return int(article.split()[1])
    except (ValueError, IndexError):
        pass
    return 999  # Put unknown titles at the end

def save_progress(sections: List[Dict[str, Any]], output_path: Path):
    """Save current progress to JSON file (thread-safe), organized by title 1-50."""
    with save_lock:
        # Organize sections by title number
        from collections import defaultdict
        title_groups = defaultdict(list)
        
        for section in sections:
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
        logger.info(f"Progress saved: {len(organized_sections)} sections ({file_size:.2f} MB)")

def main():
    """Main function to fetch all CFR titles.
    
    Note: The eCFR API (ecfr.gov/api/versioner/v1) appears to be deprecated.
    This script will attempt to use the API first, but will inform the user
    if it's unavailable. For production use, consider using GovInfo bulk XML data
    instead (similar to parse_usc_xml.py).
    """
    parser = argparse.ArgumentParser(description="Fetch Code of Federal Regulations from eCFR API")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel title fetching (default: 4)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Minimum delay between requests in seconds (default: 0.8)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific CFR version identifier (default: latest)"
    )
    parser.add_argument(
        "--title",
        type=int,
        default=None,
        help="Fetch only a specific title number (for testing)"
    )
    parser.add_argument(
        "--use-govinfo",
        action="store_true",
        help="Use GovInfo bulk XML data instead of eCFR API (recommended)"
    )
    parser.add_argument(
        "--use-ecfr-xml",
        action="store_true",
        help="Use GovInfo ECFR XML format (https://www.govinfo.gov/bulkdata/ECFR/title-X/ECFR-titleX.xml)"
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
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = base_dir / "Data" / "Knowledge" / "cfr.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    # Check if user wants to use ECFR XML format
    use_ecfr_xml = args.use_ecfr_xml
    if use_ecfr_xml:
        logger.info("Using GovInfo ECFR XML format (bulkdata/ECFR)")
    
    # Check if user wants to use GovInfo (recommended since API is deprecated)
    use_govinfo = args.use_govinfo or use_ecfr_xml
    if not use_govinfo:
        # Try API first, fall back to GovInfo if it fails
        if args.version:
            version_id = args.version
            logger.info(f"Using specified version: {version_id}")
        else:
            logger.info("Attempting to fetch latest CFR version from eCFR API...")
            version_id = get_latest_version()
            if not version_id:
                logger.warning("eCFR API is not available. Falling back to GovInfo XML...")
                use_govinfo = True
    
    if use_ecfr_xml:
        logger.info("Using GovInfo ECFR XML format (bulkdata/ECFR)")
        # Create XML download directory
        script_dir = Path(__file__).resolve().parent
        xml_dir = script_dir / "cfr_xml_temp"
        xml_dir.mkdir(parents=True, exist_ok=True)
        
        all_sections = []
        num_workers = max(1, args.workers)
        min_delay = max(0.1, args.delay)
        
        logger.info(f"Using {num_workers} workers with {min_delay}s minimum delay")
        logger.info("This may take a significant amount of time.")
        logger.info("Progress will be saved periodically.")
        
        start_time = time.time()
        processed_count = [0]
        
        def update_progress():
            with progress_lock:
                processed_count[0] += 1
                current = processed_count[0]
                if current % 5 == 0 or current == len(titles_to_process):
                    logger.info(f"Processed {current}/{len(titles_to_process)} titles")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_title = {
                executor.submit(process_title_from_ecfr_xml, title_num, xml_dir, min_delay): title_num
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
        return
    
    if use_govinfo:
        logger.info("Using GovInfo bulk XML data (recommended approach)")
        # Try current year first, then previous year if not available
        current_year = datetime.now().year
        # CFR is typically published annually, so try current and previous year
        years_to_try = [current_year, current_year - 1]
        logger.info(f"Will try years: {years_to_try}")
        cfr_year = None
        
        # Create XML download directory
        script_dir = Path(__file__).resolve().parent
        xml_dir = script_dir / "cfr_xml_temp"
        xml_dir.mkdir(parents=True, exist_ok=True)
        
        all_sections = []
        num_workers = max(1, args.workers)
        min_delay = max(0.1, args.delay)
        
        logger.info(f"Using {num_workers} workers with {min_delay}s minimum delay")
        logger.info("This may take a significant amount of time.")
        logger.info("Progress will be saved periodically.")
        
        start_time = time.time()
        processed_count = [0]
        
        def update_progress():
            with progress_lock:
                processed_count[0] += 1
                current = processed_count[0]
                if current % 5 == 0 or current == len(titles_to_process):
                    logger.info(f"Processed {current}/{len(titles_to_process)} titles")
        
        # First, try to find a working year by testing title 1
        test_title = titles_to_process[0] if titles_to_process else 1
        for year in years_to_try:
            test_file = download_cfr_xml(test_title, year, xml_dir, retries=1)
            if test_file:
                cfr_year = year
                logger.info(f"Found working CFR data for year {cfr_year}")
                break
        
        if not cfr_year:
            logger.error(f"Could not find CFR data for years {years_to_try}")
            logger.error("CFR data may not be available yet, or URL structure may have changed.")
            logger.error("Please check https://www.govinfo.gov/bulkdata/CFR for available years.")
            return
        
        logger.info(f"Using CFR data for year {cfr_year}")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_title = {
                executor.submit(process_title_from_govinfo, title_num, cfr_year, xml_dir, min_delay): title_num
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
        return
    
    # Get version identifier (only if not using GovInfo)
    if args.version:
        version_id = args.version
        logger.info(f"Using specified version: {version_id}")
    else:
        logger.info("Attempting to fetch latest CFR version from eCFR API...")
        logger.warning("Note: The eCFR API may be deprecated. If this fails, consider using --use-govinfo")
        version_id = get_latest_version()
        if not version_id:
            logger.error("=" * 60)
            logger.error("eCFR API is not available (likely deprecated).")
            logger.error("Options:")
            logger.error("1. Use --use-govinfo flag to use GovInfo bulk XML data (recommended)")
            logger.error("2. Check if there's an updated eCFR API endpoint")
            logger.error("=" * 60)
            return
    
    all_sections = []
    num_workers = max(1, args.workers)
    min_delay = max(0.1, args.delay)
    
    logger.info(f"Using {num_workers} workers with {min_delay}s minimum delay")
    logger.info("This may take a significant amount of time.")
    logger.info("Progress will be saved periodically.")
    
    start_time = time.time()
    
    # Process titles in parallel
    processed_count = [0]
    
    def update_progress():
        with progress_lock:
            processed_count[0] += 1
            current = processed_count[0]
            if current % 5 == 0 or current == len(titles_to_process):
                logger.info(f"Processed {current}/{len(titles_to_process)} titles")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_title = {
            executor.submit(process_title, version_id, title_num, min_delay): title_num
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

