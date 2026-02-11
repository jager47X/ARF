# fetch_us_code_uslm.py
"""
Fetch all United States Code data in USLM (United States Legislative Markup) format.
Downloads and processes all 54 titles of the USC from uscode.house.gov in USLM XML format.
Properly handles USLM namespaces and hierarchical structures.
"""
import os
import sys
import json
import logging
import argparse
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_us_code_uslm")

# Base URLs for USC data in USLM format
USC_BASE_URL = "https://uscode.house.gov"
USC_XML_BASE = "https://uscode.house.gov/xml/"
USC_DOWNLOAD_BASE = "https://uscode.house.gov/download/"
GOVINFO_USC_BASE = "https://www.govinfo.gov/bulkdata/USCODE"

# USLM namespace
USLM_NS = "http://schemas.gpo.gov/xml/uslm"
XHTML_NS = "http://www.w3.org/1999/xhtml"
DC_NS = "http://purl.org/dc/elements/1.1/"

# Namespace dictionary for parsing
NAMESPACES = {
    'uslm': USLM_NS,
    'xhtml': XHTML_NS,
    'dc': DC_NS
}

# All 54 titles of the USC (note: Title 53 is reserved, so there are effectively 53 active titles)
USC_TITLES = list(range(1, 55))  # 1-54, but some may not exist

# Thread-safe locks
progress_lock = Lock()
save_lock = Lock()

def download_xml_file(title_num: int, output_dir: Path, retries: int = 3) -> Optional[Path]:
    """
    Download XML file for a specific USC title in USLM format.
    Returns path to downloaded file or None if failed.
    """
    # Try different URL patterns - US Code XML files are in the download directory
    url_patterns = [
        f"{USC_DOWNLOAD_BASE}usc{title_num:02d}.xml",
        f"{USC_DOWNLOAD_BASE}xml/usc{title_num:02d}.xml",
        f"{USC_XML_BASE}usc{title_num:02d}.xml",
        f"{USC_BASE_URL}/xml/usc{title_num:02d}.xml",
        f"{USC_BASE_URL}/download/xml/usc{title_num:02d}.xml",
        f"{GOVINFO_USC_BASE}/usc{title_num:02d}.xml",
    ]
    
    output_file = output_dir / f"usc{title_num:02d}.xml"
    
    # Skip if already downloaded
    if output_file.exists():
        logger.info(f"Title {title_num} already downloaded, skipping...")
        return output_file
    
    for pattern in url_patterns:
        for attempt in range(retries):
            try:
                # Add delay before each request
                if attempt > 0:
                    wait_time = (2 ** attempt) * 3
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2.0)  # Initial delay
                
                logger.info(f"Downloading Title {title_num} from {pattern} (attempt {attempt + 1})")
                response = requests.get(pattern, timeout=(60, 120), headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/xml, text/xml, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Connection': 'keep-alive'
                })
                
                if response.status_code == 200:
                    # Check if we got actual XML, not HTML error page
                    content = response.content
                    content_str = content[:500].decode('utf-8', errors='ignore').lower()
                    
                    # Check for HTML error pages
                    if 'document not found' in content_str or ('<html' in content_str[:200] and 'xhtml' in content_str[:200]):
                        logger.warning(f"Got HTML error page for Title {title_num} at {pattern}")
                        if pattern == url_patterns[-1]:  # Last pattern
                            break
                        continue
                    
                    # Check if it's valid XML (USLM format)
                    if b'<?xml' in content[:100] and (b'<uslm' in content[:500] or b'<title' in content[:500] or b'uslm' in content[:500].lower()):
                        output_file.write_bytes(content)
                        file_size = len(content) / (1024 * 1024)  # MB
                        logger.info(f"Successfully downloaded Title {title_num} ({file_size:.2f} MB)")
                        return output_file
                    else:
                        logger.warning(f"Downloaded content doesn't appear to be valid USLM XML for Title {title_num}")
                        if pattern == url_patterns[-1]:  # Last pattern
                            break
                        continue
                elif response.status_code == 404:
                    logger.warning(f"Title {title_num} not found at {pattern}")
                    break  # Try next pattern
                else:
                    logger.warning(f"HTTP {response.status_code} for Title {title_num} at {pattern}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout downloading Title {title_num} from {pattern} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 5
                    time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error downloading Title {title_num} from {pattern} (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 5
                    time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"Error downloading Title {title_num} from {pattern}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
    logger.error(f"Failed to download Title {title_num} after trying all patterns")
    return None

def extract_text_from_element(elem: ET.Element, preserve_structure: bool = False) -> str:
    """Extract text content from XML element, handling nested elements and USLM structure."""
    if elem is None:
        return ""
    
    # Get direct text
    text_parts = []
    if elem.text:
        text_parts.append(elem.text.strip())
    
    # Get text from all child elements recursively
    for child in elem:
        # Skip certain elements that don't contribute to main text
        tag = child.tag
        if isinstance(tag, str):
            # Remove namespace prefix for comparison
            local_tag = tag.split('}')[-1] if '}' in tag else tag
            
            # Skip metadata and structural elements that don't contain substantive text
            if local_tag in ['num', 'heading', 'marker', 'ref']:
                # These might have text we want, but handle separately
                pass
            
            child_text = extract_text_from_element(child, preserve_structure)
            if child_text:
                if preserve_structure:
                    text_parts.append(child_text)
                else:
                    text_parts.append(child_text)
        
        if child.tail:
            text_parts.append(child.tail.strip())
    
    result = " ".join(text_parts).strip()
    # Clean up excessive whitespace
    result = " ".join(result.split())
    return result

def find_element_with_namespace(elem: ET.Element, tag: str, namespaces: Dict[str, str] = None) -> Optional[ET.Element]:
    """Find element handling both namespaced and non-namespaced tags."""
    if namespaces is None:
        namespaces = NAMESPACES
    
    # Try with namespace
    for prefix, ns_uri in namespaces.items():
        try:
            result = elem.find(f".//{{{ns_uri}}}{tag}")
            if result is not None:
                return result
        except:
            pass
    
    # Try without namespace (in case namespace is already in tag)
    try:
        result = elem.find(f".//{tag}")
        if result is not None:
            return result
    except:
        pass
    
    # Try with any namespace (wildcard)
    try:
        result = elem.find(f".//*[local-name()='{tag}']")
        if result is not None:
            return result
    except:
        pass
    
    return None

def findall_elements_with_namespace(elem: ET.Element, tag: str, namespaces: Dict[str, str] = None) -> List[ET.Element]:
    """Find all elements handling both namespaced and non-namespaced tags."""
    if namespaces is None:
        namespaces = NAMESPACES
    
    results = []
    
    # Try with namespace
    for prefix, ns_uri in namespaces.items():
        try:
            found = elem.findall(f".//{{{ns_uri}}}{tag}")
            if found:
                results.extend(found)
        except:
            pass
    
    # Try without namespace
    try:
        found = elem.findall(f".//{tag}")
        if found:
            results.extend(found)
    except:
        pass
    
    # Try with local-name() to match regardless of namespace
    try:
        found = elem.findall(f".//*[local-name()='{tag}']")
        if found:
            results.extend(found)
    except:
        pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for elem in results:
        elem_id = id(elem)
        if elem_id not in seen:
            seen.add(elem_id)
            unique_results.append(elem)
    
    return unique_results

def get_element_text(elem: ET.Element, tag: str, default: str = "") -> str:
    """Get text from a child element, handling namespaces."""
    if elem is None:
        return default
    
    child = find_element_with_namespace(elem, tag)
    if child is not None:
        text = extract_text_from_element(child)
        return text.strip() if text else default
    
    return default

def parse_subsection(subsec_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a subsection element from USLM XML."""
    try:
        subsec_num = get_element_text(subsec_elem, "num", "")
        subsec_heading = get_element_text(subsec_elem, "heading", "")
        
        # Get subsection text content
        text_elem = find_element_with_namespace(subsec_elem, "text")
        if text_elem is None:
            # Try to get content from paragraph elements
            para_elems = findall_elements_with_namespace(subsec_elem, "p")
            if para_elems:
                subsec_text = " ".join([extract_text_from_element(p) for p in para_elems])
            else:
                # Get all text from subsection element
                subsec_text = extract_text_from_element(subsec_elem)
        else:
            subsec_text = extract_text_from_element(text_elem)
        
        if not subsec_text and not subsec_heading:
            return None
        
        return {
            "number": subsec_num or "",
            "title": subsec_heading or f"Subsection {subsec_num}",
            "text": subsec_text
        }
    except Exception as e:
        logger.debug(f"Error parsing subsection: {e}")
        return None

def parse_paragraph(para_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a paragraph element from USLM XML."""
    try:
        para_num = get_element_text(para_elem, "num", "")
        para_heading = get_element_text(para_elem, "heading", "")
        para_text = extract_text_from_element(para_elem)
        
        if not para_text and not para_heading:
            return None
        
        return {
            "number": para_num or "",
            "title": para_heading or "",
            "text": para_text
        }
    except Exception as e:
        logger.debug(f"Error parsing paragraph: {e}")
        return None

def parse_section(section_elem: ET.Element, title_num: int, chapter_num: str = "", 
                 subchapter_num: str = "", part_num: str = "") -> Optional[Dict[str, Any]]:
    """Parse a single section element from USLM XML."""
    try:
        # Get section number
        section_num = get_element_text(section_elem, "num", "")
        
        # Get section heading/title
        section_title = get_element_text(section_elem, "heading", "")
        
        # Get section text content - try multiple approaches
        section_text = ""
        
        # First, try to get text from <text> element
        text_elem = find_element_with_namespace(section_elem, "text")
        if text_elem is not None:
            section_text = extract_text_from_element(text_elem)
        
        # If no text element, try to get content from paragraph elements
        if not section_text:
            para_elems = findall_elements_with_namespace(section_elem, "p")
            if para_elems:
                section_text = " ".join([extract_text_from_element(p) for p in para_elems if extract_text_from_element(p)])
        
        # If still no text, try to get all text from section element (excluding structural elements)
        if not section_text:
            # Get text but exclude num, heading, and other structural elements
            text_parts = []
            if section_elem.text:
                text_parts.append(section_elem.text.strip())
            
            for child in section_elem:
                tag = child.tag
                if isinstance(tag, str):
                    local_tag = tag.split('}')[-1] if '}' in tag else tag
                    # Skip structural elements
                    if local_tag not in ['num', 'heading', 'subheading', 'crossheading']:
                        child_text = extract_text_from_element(child)
                        if child_text:
                            text_parts.append(child_text)
                if child.tail:
                    text_parts.append(child.tail.strip())
            
            section_text = " ".join([p for p in text_parts if p]).strip()
        
        # Skip if no meaningful content
        if not section_text and not section_title:
            return None
        
        # Parse clauses (subsections, paragraphs, subparagraphs)
        clauses = []
        clause_counter = 1
        
        # Try to find subsections first
        subsection_elems = findall_elements_with_namespace(section_elem, "subsection")
        if subsection_elems:
            for subsec_elem in subsection_elems:
                subsec_data = parse_subsection(subsec_elem)
                if subsec_data:
                    # Ensure clause has a number
                    if not subsec_data.get("number"):
                        subsec_data["number"] = clause_counter
                        clause_counter += 1
                    clauses.append(subsec_data)
        
        # If no subsections, try paragraphs
        if not clauses:
            para_elems = findall_elements_with_namespace(section_elem, "paragraph")
            if para_elems:
                for para_elem in para_elems:
                    para_data = parse_paragraph(para_elem)
                    if para_data:
                        if not para_data.get("number"):
                            para_data["number"] = clause_counter
                            clause_counter += 1
                        clauses.append(para_data)
        
        # If still no clauses, use main text as single clause
        if not clauses and section_text:
            clauses.append({
                "number": 1,
                "title": section_title or "",
                "text": section_text
            })
        
        # Ensure we have at least one clause
        if not clauses:
            if section_text:
                clauses.append({
                    "number": 1,
                    "title": section_title or "",
                    "text": section_text
                })
            else:
                # Even without text, create a clause with just the title
                clauses.append({
                    "number": 1,
                    "title": section_title or f"Section {section_num}",
                    "text": ""
                })
        
        # Build chapter string
        chapter_str = ""
        if chapter_num:
            chapter_str = f"Chapter {chapter_num}"
            if subchapter_num:
                chapter_str += f", Subchapter {subchapter_num}"
            if part_num:
                chapter_str += f", Part {part_num}"
        
        return {
            "article": f"Title {title_num}",
            "chapter": chapter_str,
            "section": f"Section {section_num}",
            "title": section_title or f"Section {section_num}",
            "clauses": clauses
        }
        
    except Exception as e:
        logger.error(f"Error parsing section: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def get_hierarchy_from_ancestors(elem: ET.Element) -> Dict[str, str]:
    """Extract hierarchy (chapter, subchapter, part) from element's ancestors."""
    hierarchy = {
        "chapter": "",
        "subchapter": "",
        "part": "",
        "subpart": ""
    }
    
    # Walk up the tree to find parent elements
    parent = elem.getparent() if hasattr(elem, 'getparent') else None
    if parent is None:
        # ElementTree doesn't have getparent, need to track during traversal
        return hierarchy
    
    # Try to find chapter, subchapter, part in ancestors
    current = parent
    while current is not None:
        tag = current.tag
        if isinstance(tag, str):
            local_tag = tag.split('}')[-1] if '}' in tag else tag
            
            if local_tag == "chapter" and not hierarchy["chapter"]:
                hierarchy["chapter"] = get_element_text(current, "num", "")
            elif local_tag == "subchapter" and not hierarchy["subchapter"]:
                hierarchy["subchapter"] = get_element_text(current, "num", "")
            elif local_tag == "part" and not hierarchy["part"]:
                hierarchy["part"] = get_element_text(current, "num", "")
            elif local_tag == "subpart" and not hierarchy["subpart"]:
                hierarchy["subpart"] = get_element_text(current, "num", "")
        
        # Try to get parent (ElementTree limitation - we'll handle this differently)
        if hasattr(current, 'getparent'):
            current = current.getparent()
        else:
            break
    
    return hierarchy

def parse_title_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse a single title XML file in USLM format and return list of sections."""
    if not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        return []
    
    try:
        logger.info(f"Parsing {xml_path.name}...")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Register namespaces for easier searching
        for prefix, uri in NAMESPACES.items():
            ET.register_namespace(prefix, uri)
        
        # Extract title number - try multiple approaches
        title_num = 0
        title_elem = find_element_with_namespace(root, "title")
        
        if title_elem is not None:
            title_num_str = get_element_text(title_elem, "num", "")
            try:
                title_num = int(title_num_str)
            except:
                # Try to extract from filename
                filename = xml_path.stem
                if filename.startswith("usc"):
                    try:
                        title_num = int(filename[3:5])
                    except:
                        pass
        
        if title_num == 0:
            logger.warning(f"Could not determine title number for {xml_path.name}")
            # Try to extract from filename as fallback
            filename = xml_path.stem
            if filename.startswith("usc"):
                try:
                    title_num = int(filename[3:5])
                except:
                    pass
        
        if title_num == 0:
            logger.error(f"Could not determine title number for {xml_path.name}, skipping")
            return []
        
        sections = []
        
        # Build a map of element to its hierarchy by traversing the tree
        # Since ElementTree doesn't track parents well, we'll do a recursive traversal
        hierarchy_map = {}
        
        def traverse_with_hierarchy(elem: ET.Element, current_hierarchy: Dict[str, str]):
            """Recursively traverse tree, tracking hierarchy context."""
            tag = elem.tag
            if isinstance(tag, str):
                local_tag = tag.split('}')[-1] if '}' in tag else tag
                
                # Update hierarchy based on current element
                new_hierarchy = current_hierarchy.copy()
                
                if local_tag == "chapter":
                    num = get_element_text(elem, "num", "")
                    new_hierarchy["chapter"] = num
                    new_hierarchy["subchapter"] = ""  # Reset subchapter when entering new chapter
                    new_hierarchy["part"] = ""  # Reset part when entering new chapter
                elif local_tag == "subchapter":
                    num = get_element_text(elem, "num", "")
                    new_hierarchy["subchapter"] = num
                    new_hierarchy["part"] = ""  # Reset part when entering new subchapter
                elif local_tag == "part":
                    num = get_element_text(elem, "num", "")
                    new_hierarchy["part"] = num
                elif local_tag == "subpart":
                    num = get_element_text(elem, "num", "")
                    new_hierarchy["subpart"] = num
                elif local_tag == "section":
                    # Store hierarchy for this section
                    hierarchy_map[id(elem)] = new_hierarchy.copy()
            
            # Recursively process children
            for child in elem:
                traverse_with_hierarchy(child, new_hierarchy)
        
        # Start traversal from root or title element
        start_elem = title_elem if title_elem is not None else root
        initial_hierarchy = {"chapter": "", "subchapter": "", "part": "", "subpart": ""}
        traverse_with_hierarchy(start_elem, initial_hierarchy)
        
        # Find all sections - try multiple namespace approaches
        section_elems = findall_elements_with_namespace(root, "section")
        
        if not section_elems:
            # Try finding within title element
            if title_elem is not None:
                section_elems = findall_elements_with_namespace(title_elem, "section")
        
        logger.info(f"Found {len(section_elems)} sections in Title {title_num}")
        
        # Process sections
        for section_elem in section_elems:
            # Get hierarchy for this section
            section_id = id(section_elem)
            hierarchy = hierarchy_map.get(section_id, {
                "chapter": "",
                "subchapter": "",
                "part": "",
                "subpart": ""
            })
            
            parsed_section = parse_section(section_elem, title_num, 
                                         hierarchy.get("chapter", ""),
                                         hierarchy.get("subchapter", ""),
                                         hierarchy.get("part", ""))
            if parsed_section:
                sections.append(parsed_section)
        
        logger.info(f"Successfully parsed {len(sections)} sections from Title {title_num}")
        return sections
        
    except ET.ParseError as e:
        logger.error(f"XML parse error in {xml_path.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []
    except Exception as e:
        logger.error(f"Error parsing {xml_path.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []

def download_all_titles(output_dir: Path, num_workers: int = 2) -> List[Path]:
    """Download all available USC title XML files with multi-worker support."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    
    logger.info(f"Starting download of USC USLM XML files to {output_dir} with {num_workers} workers")
    
    # Create a session per worker for better connection handling
    def download_with_tracking(title_num: int) -> Optional[Path]:
        """Download a single title with tracking."""
        session = requests.Session()
        try:
            return download_xml_file(title_num, output_dir, session=session)
        finally:
            session.close()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_with_tracking, title_num): title_num 
                  for title_num in USC_TITLES}
        
        for future in as_completed(futures):
            title_num = futures[future]
            try:
                xml_file = future.result()
                if xml_file:
                    downloaded_files.append(xml_file)
            except Exception as e:
                logger.error(f"Error downloading Title {title_num}: {e}")
            time.sleep(0.5)  # Small delay between completions
    
    logger.info(f"Downloaded {len(downloaded_files)} title files")
    return downloaded_files

def parse_all_titles(xml_dir: Path, num_workers: int = 4) -> List[Dict[str, Any]]:
    """Parse all XML files in directory and return combined sections with multi-worker support."""
    all_sections = []
    
    xml_files = sorted(xml_dir.glob("usc*.xml"))
    logger.info(f"Found {len(xml_files)} XML files to parse with {num_workers} workers")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_title_xml, xml_file): xml_file 
                  for xml_file in xml_files}
        
        for future in as_completed(futures):
            xml_file = futures[future]
            try:
                sections = future.result()
                all_sections.extend(sections)
                logger.info(f"Parsed {xml_file.name}: {len(sections)} sections (total: {len(all_sections)})")
            except Exception as e:
                logger.error(f"Error parsing {xml_file.name}: {e}")
    
    logger.info(f"Total sections parsed: {len(all_sections)}")
    return all_sections

def create_json_output(sections: List[Dict[str, Any]], output_path: Path):
    """Create JSON file in the required format."""
    output_data = {
        "data": {
            "united_states_code": {
                "titles": sections
            }
        }
    }
    
    logger.info(f"Writing {len(sections)} sections to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first, then rename (atomic write)
    temp_path = output_path.with_suffix('.json.tmp')
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    temp_path.replace(output_path)
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"JSON file created: {output_path} ({file_size:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Download and parse USC XML files in USLM format")
    parser.add_argument("--download-dir", type=str, default="usc_uslm_xml_temp",
                       help="Directory to store downloaded XML files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, use existing XML files")
    parser.add_argument("--titles", type=str, default=None,
                       help="Comma-separated list of title numbers to process (e.g., '1,5,18')")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of worker threads for parallel processing (default: 2 for downloads, 4 for parsing)")
    parser.add_argument("--parse-workers", type=int, default=4,
                       help="Number of worker threads for parsing (default: 4)")
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    download_dir = script_dir / args.download_dir
    output_path = Path(args.output) if args.output else base_dir / "Data" / "Knowledge" / "us_code.json"
    
    # Filter titles if specified
    titles_to_process = list(USC_TITLES)
    if args.titles:
        try:
            title_nums = [int(t.strip()) for t in args.titles.split(",")]
            titles_to_process = [t for t in USC_TITLES if t in title_nums]
            logger.info(f"Processing only titles: {titles_to_process}")
        except ValueError:
            logger.error(f"Invalid title list: {args.titles}")
            return 1
    
    # Create a modified download function that uses filtered titles
    def download_filtered_titles(output_dir: Path, num_workers: int = 2) -> List[Path]:
        """Download filtered USC title XML files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        
        logger.info(f"Starting download of USC USLM XML files to {output_dir} with {num_workers} workers")
        
        def download_with_tracking(title_num: int) -> Optional[Path]:
            """Download a single title with tracking."""
            return download_xml_file(title_num, output_dir)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(download_with_tracking, title_num): title_num 
                      for title_num in titles_to_process}
            
            for future in as_completed(futures):
                title_num = futures[future]
                try:
                    xml_file = future.result()
                    if xml_file:
                        downloaded_files.append(xml_file)
                except Exception as e:
                    logger.error(f"Error downloading Title {title_num}: {e}")
                time.sleep(0.5)
        
        logger.info(f"Downloaded {len(downloaded_files)} title files")
        return downloaded_files
    
    # Download XML files if needed
    if not args.skip_download:
        download_filtered_titles(download_dir, num_workers=args.workers)
    
    # Parse XML files
    sections = parse_all_titles(download_dir, num_workers=args.parse_workers)
    
    if not sections:
        logger.error("No sections parsed. Check XML files and parsing logic.")
        return 1
    
    # Create JSON output
    create_json_output(sections, output_path)
    
    logger.info("Processing complete!")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

