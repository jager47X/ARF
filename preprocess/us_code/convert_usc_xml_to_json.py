# convert_usc_xml_to_json.py
"""
Convert US Code XML files to JSON format with strict legal text normalization.
Reads XML files from a directory and converts them to clean JSON containing
ONLY binding statutory text, excluding notes, amendments, editorial content, etc.
"""
import argparse
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_usc_xml")

# USLM namespace - support multiple namespace versions
USLM_NS_V1 = "http://xml.house.gov/schemas/uslm/1.0"
USLM_NS_GPO = "http://schemas.gpo.gov/xml/uslm"
XHTML_NS = "http://www.w3.org/1999/xhtml"
DC_NS = "http://purl.org/dc/elements/1.1/"

# Namespace dictionary for parsing - include both namespace versions
NAMESPACES = {
    'uslm': USLM_NS_V1,
    'uslm_gpo': USLM_NS_GPO,
    'xhtml': XHTML_NS,
    'dc': DC_NS
}

# Elements to exclude (non-statutory content)
NON_STATUTORY_ELEMENTS = {
    'note', 'notes', 'amendment', 'amendments', 'editorial', 'editorialnote',
    'effectivedate', 'effective', 'publiclaw', 'statutesatlarge', 'waiver',
    'temporary', 'temp', 'repeal', 'repealed', 'sourcecredit', 'credit',
    'history', 'codification', 'prior', 'references', 'reference'
}

def is_non_statutory_element(elem: ET.Element) -> bool:
    """Check if element represents non-statutory content that should be excluded."""
    if elem is None:
        return False

    tag = elem.tag
    if isinstance(tag, str):
        local_tag = tag.split('}')[-1] if '}' in tag else tag
        local_tag_lower = local_tag.lower()

        # Check tag name
        if local_tag_lower in NON_STATUTORY_ELEMENTS:
            return True

        # Check for common patterns in attributes or text
        if hasattr(elem, 'attrib'):
            for attr_key, attr_val in elem.attrib.items():
                attr_lower = attr_val.lower() if isinstance(attr_val, str) else ""
                if any(non_stat in attr_lower for non_stat in NON_STATUTORY_ELEMENTS):
                    return True

    return False

def normalize_text(text: str) -> str:
    """Normalize text: remove smart quotes, typographic artifacts, HTML entities."""
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

    # Remove leading/trailing quotes if the entire text is wrapped in quotes
    # This handles cases where XML text extraction adds quotes
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        # Check if it's actually quoted content or just starts/ends with quotes
        # Only remove if it looks like the entire text is wrapped
        inner = text[1:-1]
        if '"' not in inner or (inner.count('"') % 2 == 0):
            text = inner

    # Convert clause letters to numbers: (a) -> (1), (b) -> (2), etc.
    # Match patterns like "(a)", "(b)", etc. that are likely clause identifiers
    def convert_clause_letter(match):
        letter = match.group(1).lower()
        if letter.isalpha() and len(letter) == 1:
            # Single letter: a=1, b=2, etc.
            num = ord(letter) - ord('a') + 1
            return f"({num})"
        return match.group(0)  # Return unchanged if not a single letter

    # Match (letter) patterns, but be careful not to match things like "(1)" or "(see)"
    # Look for patterns like "(a)", "(b)", etc. that are likely clause identifiers
    text = re.sub(r'\(([a-z])\)', convert_clause_letter, text, flags=re.IGNORECASE)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def normalize_clause_number(num: str) -> str:
    """Normalize clause number to string format: 'main', '(1)', '(2)', etc.
    Converts letters to numbers: a->1, b->2, etc."""
    if not num:
        return "main"

    num = str(num).strip()

    # Remove any existing § symbols or section markers
    num = re.sub(r'§+\s*', '', num)
    num = num.strip()

    if not num:
        return "main"

    # Remove extra parentheses if duplicated
    num = re.sub(r'^\(+', '(', num)
    num = re.sub(r'\)+$', ')', num)

    # Extract content from parentheses or get the whole thing
    match = re.match(r'^\(?([a-z0-9]+)\)?$', num, re.IGNORECASE)
    if match:
        content = match.group(1).lower()

        # Convert letters to numbers: a->1, b->2, ..., z->26
        if content.isalpha():
            # Convert single letter to number (a=1, b=2, etc.)
            letter_num = ord(content) - ord('a') + 1
            return f"({letter_num})"
        else:
            # Already a number, just wrap in parentheses
            return f"({content})"

    # If it's just a single letter without parentheses, convert to number
    if re.match(r'^[a-z]$', num, re.IGNORECASE):
        letter_num = ord(num.lower()) - ord('a') + 1
        return f"({letter_num})"

    # If it's a number without parentheses, wrap it
    if num.isdigit():
        return f"({num})"

    # Default to main if we can't parse it
    return "main"

def format_section_number(section_num: str) -> str:
    """Format section number with § symbol."""
    if not section_num:
        return ""

    section_num = section_num.strip()

    # Remove existing § symbols
    section_num = re.sub(r'§+\s*', '', section_num)

    # Add § symbol
    return f"§ {section_num}"

def extract_text_from_element(elem: ET.Element, preserve_structure: bool = False, exclude_non_statutory: bool = True) -> str:
    """Extract text content from XML element, handling nested elements and USLM structure.
    Excludes non-statutory content like notes, amendments, etc."""
    if elem is None:
        return ""

    # Skip non-statutory elements
    if exclude_non_statutory and is_non_statutory_element(elem):
        return ""

    # Get direct text
    text_parts = []
    if elem.text:
        text = elem.text.strip()
        if text:
            text_parts.append(text)

    # Get text from all child elements recursively
    for child in elem:
        # Skip non-statutory child elements
        if exclude_non_statutory and is_non_statutory_element(child):
            continue

        tag = child.tag
        if isinstance(tag, str):
            # Remove namespace prefix for comparison
            local_tag = tag.split('}')[-1] if '}' in tag else tag
            local_tag_lower = local_tag.lower()

            # Skip known non-statutory element types
            if local_tag_lower in NON_STATUTORY_ELEMENTS:
                continue

            # Extract text from child element
            child_text = extract_text_from_element(child, preserve_structure, exclude_non_statutory)
            if child_text:
                text_parts.append(child_text)

        # Include tail text (text after the element)
        if child.tail:
            tail_text = child.tail.strip()
            if tail_text:
                text_parts.append(tail_text)

    result = " ".join(text_parts).strip()

    # Normalize text (remove smart quotes, etc.)
    result = normalize_text(result)

    # Clean up excessive whitespace while preserving structure if needed
    if not preserve_structure:
        result = " ".join(result.split())

    return result

def find_element_with_namespace(elem: ET.Element, tag: str, namespaces: Dict[str, str] = None) -> Optional[ET.Element]:
    """Find element handling both namespaced and non-namespaced tags."""
    if namespaces is None:
        namespaces = NAMESPACES

    if elem is None:
        return None

    # Strategy 1: Try with each namespace explicitly (including both USLM versions)
    for prefix, ns_uri in namespaces.items():
        try:
            result = elem.find(f".//{{{ns_uri}}}{tag}")
            if result is not None:
                return result
        except Exception as e:
            logger.debug(f"Error finding element with namespace {ns_uri}: {e}")
            pass

    # Also try common USLM namespace variations directly
    for ns_uri in [USLM_NS_V1, USLM_NS_GPO]:
        if ns_uri not in namespaces.values():
            try:
                result = elem.find(f".//{{{ns_uri}}}{tag}")
                if result is not None:
                    return result
            except Exception:
                pass

    # Strategy 2: Try without namespace (in case namespace is already in tag or no namespace)
    try:
        result = elem.find(f".//{tag}")
        if result is not None:
            return result
    except Exception as e:
        logger.debug(f"Error finding element without namespace: {e}")
        pass

    # Strategy 3: Try with local-name() XPath to match regardless of namespace
    try:
        result = elem.find(f".//*[local-name()='{tag}']")
        if result is not None:
            return result
    except Exception as e:
        logger.debug(f"Error finding element with local-name(): {e}")
        pass

    # Strategy 4: Try iterating manually to catch any edge cases
    try:
        for child in elem.iter():
            tag_name = child.tag
            if isinstance(tag_name, str):
                # Remove namespace prefix for comparison
                local_tag = tag_name.split('}')[-1] if '}' in tag_name else tag_name
                if local_tag == tag:
                    return child
    except Exception as e:
        logger.debug(f"Error iterating elements: {e}")
        pass

    return None

def findall_elements_with_namespace(elem: ET.Element, tag: str, namespaces: Dict[str, str] = None) -> List[ET.Element]:
    """Find all elements handling both namespaced and non-namespaced tags."""
    if namespaces is None:
        namespaces = NAMESPACES

    if elem is None:
        return []

    results = []

    # Strategy 1: Try with each namespace explicitly (including both USLM versions)
    for prefix, ns_uri in namespaces.items():
        try:
            found = elem.findall(f".//{{{ns_uri}}}{tag}")
            if found:
                results.extend(found)
        except Exception as e:
            logger.debug(f"Error finding elements with namespace {ns_uri}: {e}")
            pass

    # Also try common USLM namespace variations directly
    for ns_uri in [USLM_NS_V1, USLM_NS_GPO]:
        if ns_uri not in namespaces.values():
            try:
                found = elem.findall(f".//{{{ns_uri}}}{tag}")
                if found:
                    results.extend(found)
            except Exception:
                pass

    # Strategy 2: Try without namespace (in case namespace is already in tag or no namespace)
    try:
        found = elem.findall(f".//{tag}")
        if found:
            results.extend(found)
    except Exception as e:
        logger.debug(f"Error finding elements without namespace: {e}")
        pass

    # Strategy 3: Try with local-name() XPath to match regardless of namespace
    try:
        found = elem.findall(f".//*[local-name()='{tag}']")
        if found:
            results.extend(found)
    except Exception as e:
        logger.debug(f"Error finding elements with local-name(): {e}")
        pass

    # Strategy 4: Try iterating manually to catch any edge cases
    try:
        for child in elem.iter():
            tag_name = child.tag
            if isinstance(tag_name, str):
                # Remove namespace prefix for comparison
                local_tag = tag_name.split('}')[-1] if '}' in tag_name else tag_name
                if local_tag == tag:
                    results.append(child)
    except Exception as e:
        logger.debug(f"Error iterating elements: {e}")
        pass

    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result_elem in results:
        elem_id = id(result_elem)
        if elem_id not in seen:
            seen.add(elem_id)
            unique_results.append(result_elem)

    return unique_results

def get_element_text(elem: ET.Element, tag: str, default: str = "") -> str:
    """Get text from a child element, handling namespaces."""
    if elem is None:
        return default

    child = find_element_with_namespace(elem, tag)
    if child is not None:
        # For structural elements like num/heading, we want the text even if it's in a note context
        # But we still normalize it
        text = extract_text_from_element(child, exclude_non_statutory=False)
        text = normalize_text(text) if text else ""
        return text.strip() if text else default

    return default

def parse_subsection(subsec_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a subsection element from USLM XML, excluding non-statutory content."""
    try:
        # Skip if this is a non-statutory element
        if is_non_statutory_element(subsec_elem):
            return None

        subsec_num = get_element_text(subsec_elem, "num", "")
        subsec_heading = get_element_text(subsec_elem, "heading", "")

        # Normalize heading
        subsec_heading = normalize_text(subsec_heading) if subsec_heading else ""

        # Get subsection text content (excluding non-statutory content)
        subsec_text = ""
        try:
            text_elem = find_element_with_namespace(subsec_elem, "text")
            if text_elem is None:
                # Try to get content from paragraph elements
                para_elems = findall_elements_with_namespace(subsec_elem, "p")
                if para_elems:
                    para_texts = []
                    for p in para_elems:
                        # Skip non-statutory paragraphs
                        if is_non_statutory_element(p):
                            continue
                        try:
                            p_text = extract_text_from_element(p, exclude_non_statutory=True)
                            if p_text:
                                para_texts.append(p_text)
                        except Exception as e:
                            logger.debug(f"Error extracting paragraph text in subsection: {e}")
                            continue
                    if para_texts:
                        subsec_text = " ".join(para_texts)
                else:
                    # Get all text from subsection element (excluding non-statutory)
                    subsec_text = extract_text_from_element(subsec_elem, exclude_non_statutory=True)
            else:
                # Skip if text element is non-statutory
                if not is_non_statutory_element(text_elem):
                    subsec_text = extract_text_from_element(text_elem, exclude_non_statutory=True)
        except Exception as e:
            logger.debug(f"Error extracting subsection text: {e}")

        # Normalize text
        subsec_text = normalize_text(subsec_text) if subsec_text else ""

        # Skip if no meaningful statutory content
        if not subsec_text and not subsec_heading:
            return None

        # Normalize clause number
        clause_num = normalize_clause_number(subsec_num)

        return {
            "number": clause_num,
            "title": subsec_heading,
            "text": subsec_text
        }
    except Exception as e:
        logger.debug(f"Error parsing subsection: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def parse_paragraph(para_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a paragraph element from USLM XML, excluding non-statutory content."""
    try:
        # Skip if this is a non-statutory element
        if is_non_statutory_element(para_elem):
            return None

        para_num = get_element_text(para_elem, "num", "")
        para_heading = get_element_text(para_elem, "heading", "")

        # Normalize heading
        para_heading = normalize_text(para_heading) if para_heading else ""

        para_text = ""
        try:
            para_text = extract_text_from_element(para_elem, exclude_non_statutory=True)
        except Exception as e:
            logger.debug(f"Error extracting paragraph text: {e}")

        # Normalize text
        para_text = normalize_text(para_text) if para_text else ""

        if not para_text and not para_heading:
            return None

        # Normalize clause number
        clause_num = normalize_clause_number(para_num)

        return {
            "number": clause_num,
            "title": para_heading,
            "text": para_text
        }
    except Exception as e:
        logger.debug(f"Error parsing paragraph: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def parse_section(section_elem: ET.Element, title_num: int, chapter_num: str = "",
                 subchapter_num: str = "", part_num: str = "") -> Optional[Dict[str, Any]]:
    """Parse a single section element from USLM XML, excluding non-statutory content."""
    section_num = ""
    section_title = ""

    try:
        # Skip if this is a non-statutory section
        if is_non_statutory_element(section_elem):
            return None

        # Get section number
        try:
            section_num = get_element_text(section_elem, "num", "")
            # Clean section number: remove "Section", "§", quotes, and trailing periods
            if section_num:
                section_num = str(section_num).strip()
                # Remove leading/trailing quotes
                section_num = section_num.strip('"\'')
                # Remove "Section" prefix (case insensitive, may appear multiple times)
                while True:
                    new_num = re.sub(r'^(section|Section|SECTION)\s+', '', section_num, flags=re.IGNORECASE).strip()
                    if new_num == section_num:
                        break
                    section_num = new_num
                # Remove § symbols
                section_num = re.sub(r'§+\s*', '', section_num)
                # Remove trailing periods and other punctuation
                section_num = section_num.rstrip('.')
                section_num = section_num.strip()
        except Exception as e:
            logger.warning(f"Error extracting section number for Title {title_num}: {e}")
            section_num = ""

        # Get section heading/title
        try:
            section_title = get_element_text(section_elem, "heading", "")
            section_title = normalize_text(section_title) if section_title else ""
        except Exception as e:
            logger.warning(f"Error extracting section title for Title {title_num}, Section {section_num}: {e}")
            section_title = ""

        # Get section text content - try multiple approaches (excluding non-statutory content)
        section_text = ""

        # First, try to get text from <text> element
        try:
            text_elem = find_element_with_namespace(section_elem, "text")
            if text_elem is not None and not is_non_statutory_element(text_elem):
                section_text = extract_text_from_element(text_elem, exclude_non_statutory=True)
        except Exception as e:
            logger.debug(f"Error extracting text element for Title {title_num}, Section {section_num}: {e}")

        # If no text element, try to get content from paragraph elements
        if not section_text:
            try:
                para_elems = findall_elements_with_namespace(section_elem, "p")
                if para_elems:
                    para_texts = []
                    for p in para_elems:
                        # Skip non-statutory paragraphs
                        if is_non_statutory_element(p):
                            continue
                        try:
                            p_text = extract_text_from_element(p, exclude_non_statutory=True)
                            if p_text:
                                para_texts.append(p_text)
                        except Exception as e:
                            logger.debug(f"Error extracting paragraph text: {e}")
                    if para_texts:
                        section_text = " ".join(para_texts)
            except Exception as e:
                logger.debug(f"Error extracting paragraphs for Title {title_num}, Section {section_num}: {e}")

        # If still no text, try to get all text from section element (excluding structural and non-statutory elements)
        if not section_text:
            try:
                # Get text but exclude num, heading, and other structural elements
                text_parts = []
                if section_elem.text:
                    text = section_elem.text.strip()
                    if text:
                        text_parts.append(text)

                for child in section_elem:
                    # Skip non-statutory elements
                    if is_non_statutory_element(child):
                        continue

                    try:
                        tag = child.tag
                        if isinstance(tag, str):
                            local_tag = tag.split('}')[-1] if '}' in tag else tag
                            local_tag_lower = local_tag.lower()

                            # Skip structural and non-statutory elements
                            if local_tag_lower in ['num', 'heading', 'subheading', 'crossheading'] or \
                               local_tag_lower in NON_STATUTORY_ELEMENTS:
                                continue

                            child_text = extract_text_from_element(child, exclude_non_statutory=True)
                            if child_text:
                                text_parts.append(child_text)
                        if child.tail:
                            tail_text = child.tail.strip()
                            if tail_text:
                                text_parts.append(tail_text)
                    except Exception as e:
                        logger.debug(f"Error processing child element in section: {e}")
                        continue

                if text_parts:
                    section_text = " ".join([p for p in text_parts if p]).strip()
            except Exception as e:
                logger.debug(f"Error extracting text from section element for Title {title_num}, Section {section_num}: {e}")

        # Normalize section text
        section_text = normalize_text(section_text) if section_text else ""

        # Skip if no meaningful content
        if not section_text and not section_title:
            logger.debug(f"Skipping section with no content: Title {title_num}, Section {section_num}")
            return None

        # Parse clauses (subsections, paragraphs, subparagraphs)
        clauses = []
        clause_counter = 1

        # Try to find subsections first
        try:
            subsection_elems = findall_elements_with_namespace(section_elem, "subsection")
            if subsection_elems:
                for subsec_elem in subsection_elems:
                    # Skip non-statutory subsections
                    if is_non_statutory_element(subsec_elem):
                        continue
                    try:
                        subsec_data = parse_subsection(subsec_elem)
                        if subsec_data:
                            # Ensure clause has a normalized number
                            if not subsec_data.get("number") or subsec_data.get("number") == "main":
                                # Try to infer from position or use sequential numbering
                                subsec_data["number"] = normalize_clause_number(str(clause_counter))
                                clause_counter += 1
                            clauses.append(subsec_data)
                    except Exception as e:
                        logger.warning(f"Error parsing subsection in Title {title_num}, Section {section_num}: {e}")
                        continue
        except Exception as e:
            logger.debug(f"Error finding subsections for Title {title_num}, Section {section_num}: {e}")

        # If no subsections, try paragraphs
        if not clauses:
            try:
                para_elems = findall_elements_with_namespace(section_elem, "paragraph")
                if para_elems:
                    for para_elem in para_elems:
                        # Skip non-statutory paragraphs
                        if is_non_statutory_element(para_elem):
                            continue
                        try:
                            para_data = parse_paragraph(para_elem)
                            if para_data:
                                # Ensure clause has a normalized number
                                if not para_data.get("number") or para_data.get("number") == "main":
                                    para_data["number"] = normalize_clause_number(str(clause_counter))
                                    clause_counter += 1
                                clauses.append(para_data)
                        except Exception as e:
                            logger.warning(f"Error parsing paragraph in Title {title_num}, Section {section_num}: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Error finding paragraphs for Title {title_num}, Section {section_num}: {e}")

        # If still no clauses, use main text as single clause
        if not clauses and section_text:
            clauses.append({
                "number": "main",
                "title": section_title,
                "text": section_text
            })

        # Ensure we have at least one clause
        if not clauses:
            if section_text:
                clauses.append({
                    "number": "main",
                    "title": section_title,
                    "text": section_text
                })
            else:
                # Even without text, create a clause with just the title (if title exists)
                if section_title:
                    clauses.append({
                        "number": "main",
                        "title": section_title,
                        "text": ""
                    })
                else:
                    # No content at all, skip this section
                    return None

        # Build chapter string (format: just the number, e.g., "1" not "Chapter 1" or "CHAPTER 1—")
        chapter_str = ""
        try:
            if chapter_num:
                chapter_num_str = str(chapter_num).strip()
                # Remove all "Chapter", "CHAPTER" prefixes (case insensitive, may appear multiple times)
                # Handle cases like "Chapter CHAPTER 1--" by removing all instances
                chapter_clean = chapter_num_str
                while True:
                    new_clean = re.sub(r'^(chapter|CHAPTER)\s*', '', chapter_clean, flags=re.IGNORECASE).strip()
                    if new_clean == chapter_clean:
                        break
                    chapter_clean = new_clean

                # Remove everything after and including em-dash, en-dash, or regular dash (including multiple dashes)
                chapter_clean = re.sub(r'[—–\-]+.*$', '', chapter_clean).strip()

                # Extract just digits and optional letters (for cases like "1A")
                match = re.search(r'([0-9]+[A-Za-z]?)', chapter_clean)
                if match:
                    chapter_str = match.group(1)
                else:
                    # If no match, try to extract any number
                    num_match = re.search(r'(\d+)', chapter_clean)
                    if num_match:
                        chapter_str = num_match.group(1)
                    else:
                        chapter_str = chapter_clean if chapter_clean else ""
        except Exception as e:
            logger.debug(f"Error building chapter string: {e}")

        # Format section number with § symbol
        formatted_section_num = format_section_number(section_num) if section_num else ""

        # Convert clauses to sections array - each clause becomes a section entry
        section_array = []
        for clause in clauses:
            # Section number is already cleaned at extraction, but ensure it's just the number
            clean_section_num = section_num.strip() if section_num else ""

            section_array.append({
                "number": clean_section_num,  # Just the section number (e.g., "7"), no § symbol or "Section" prefix
                "title": clause.get("title") or section_title,  # Use clause title or fallback to section title
                "text": clause.get("text", "")
            })

        # Return single section object with sections array (replacing clauses)
        return {
            "article": f"Title {title_num}",
            "chapter": chapter_str,  # Just the number, e.g., "1"
            "section": section_array  # Array of section entries (replaces "clauses")
        }

    except Exception as e:
        logger.error(f"Error parsing section for Title {title_num}, Section {section_num}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

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

        # Extract title number - try multiple approaches with better fallback logic
        title_num = 0
        title_elem = find_element_with_namespace(root, "title")

        # Also check for appendix elements (like usc50A.xml)
        appendix_elem = find_element_with_namespace(root, "appendix")

        # Strategy 1: Extract from title element's num child
        if title_elem is not None:
            title_num_str = get_element_text(title_elem, "num", "")
            if title_num_str:
                try:
                    title_num = int(title_num_str)
                except (ValueError, TypeError):
                    logger.debug(f"Could not parse title number '{title_num_str}' from title element")

        # Strategy 2: Extract from filename (handles various formats)
        if title_num == 0:
            filename = xml_path.stem
            if filename.startswith("usc"):
                # Try multiple extraction strategies
                # Handle formats like: usc1, usc01, usc50, usc50A, usc50a
                try:
                    # Remove 'usc' prefix and any trailing letters (A, a, etc.)
                    num_part = filename[3:].rstrip('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz')
                    if num_part:
                        title_num = int(num_part)
                except (ValueError, TypeError):
                    # Try extracting just digits
                    digits = re.search(r'\d+', filename[3:])
                    if digits:
                        try:
                            title_num = int(digits.group())
                        except (ValueError, TypeError):
                            pass

        # Strategy 3: For appendix files, extract from filename
        if title_num == 0 and appendix_elem is not None:
            filename = xml_path.stem
            if filename.startswith("usc"):
                try:
                    num_part = filename[3:].rstrip('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz')
                    if num_part:
                        title_num = int(num_part)
                except (ValueError, TypeError):
                    digits = re.search(r'\d+', filename[3:])
                    if digits:
                        try:
                            title_num = int(digits.group())
                        except (ValueError, TypeError):
                            pass

        # Final validation
        if title_num == 0:
            logger.error(f"Could not determine title number for {xml_path.name} after all attempts, skipping")
            return []

        if title_num < 1 or title_num > 54:  # US Code has titles 1-54
            logger.warning(f"Title number {title_num} seems out of range (1-54) for {xml_path.name}, but continuing")

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
                    # Clean chapter number: remove "Chapter", "CHAPTER" prefixes and dashes
                    if num:
                        num_str = str(num).strip()
                        # Remove "Chapter" or "CHAPTER" prefixes (case insensitive, may appear multiple times)
                        chapter_clean = num_str
                        while True:
                            new_clean = re.sub(r'^(chapter|CHAPTER)\s*', '', chapter_clean, flags=re.IGNORECASE).strip()
                            if new_clean == chapter_clean:
                                break
                            chapter_clean = new_clean
                        # Remove everything after and including em-dash, en-dash, or regular dash
                        chapter_clean = re.sub(r'[—–\-]+.*$', '', chapter_clean).strip()
                        # Extract just digits and optional letters (for cases like "1A")
                        match = re.search(r'([0-9]+[A-Za-z]?)', chapter_clean)
                        if match:
                            num = match.group(1)
                        else:
                            # If no match, try to extract any number
                            num_match = re.search(r'(\d+)', chapter_clean)
                            if num_match:
                                num = num_match.group(1)
                            else:
                                num = chapter_clean if chapter_clean else ""
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

        # Start traversal from root or title/appendix element
        start_elem = title_elem if title_elem is not None else (appendix_elem if appendix_elem is not None else root)
        initial_hierarchy = {"chapter": "", "subchapter": "", "part": "", "subpart": ""}
        traverse_with_hierarchy(start_elem, initial_hierarchy)

        # Find all sections - try multiple namespace approaches
        section_elems = findall_elements_with_namespace(root, "section")

        if not section_elems:
            # Try finding within title element
            if title_elem is not None:
                section_elems = findall_elements_with_namespace(title_elem, "section")
            elif appendix_elem is not None:
                # Appendices might not have sections, skip them
                logger.info(f"Appendix file {xml_path.name} has no sections, skipping")
                return []

        logger.info(f"Found {len(section_elems)} sections in Title {title_num}")

        # Process sections
        skipped_count = 0
        processed_count = 0
        for section_elem in section_elems:
            # Get hierarchy for this section
            section_id = id(section_elem)
            hierarchy = hierarchy_map.get(section_id, {
                "chapter": "",
                "subchapter": "",
                "part": "",
                "subpart": ""
            })

            try:
                parsed_sections = parse_section(section_elem, title_num,
                                             hierarchy.get("chapter", ""),
                                             hierarchy.get("subchapter", ""),
                                             hierarchy.get("part", ""))

                # parse_section now returns a single section object with "section" array (replacing clauses)
                if parsed_sections and isinstance(parsed_sections, dict):
                    # Validate required fields
                    required_fields = ["article", "section"]
                    missing_fields = [field for field in required_fields if field not in parsed_sections]
                    if missing_fields:
                        logger.warning(f"Section missing required fields {missing_fields} in Title {title_num}, skipping")
                        skipped_count += 1
                        continue

                    # Validate sections array (replaces clauses)
                    if not isinstance(parsed_sections.get("section"), list):
                        logger.warning(f"Section 'section' field is not a list in Title {title_num}, skipping")
                        skipped_count += 1
                        continue

                    if len(parsed_sections.get("section", [])) == 0:
                        logger.warning(f"Section has no sections in Title {title_num}, skipping")
                        skipped_count += 1
                        continue

                    # Validate each section entry
                    valid_sections = []
                    for sec in parsed_sections.get("section", []):
                        if not isinstance(sec, dict):
                            logger.debug("Skipping invalid section entry (not a dict)")
                            continue
                        # Sections should have at least text or title
                        if not sec.get("text") and not sec.get("title"):
                            logger.debug("Skipping section entry with no text or title")
                            continue
                        valid_sections.append(sec)

                    if not valid_sections:
                        logger.warning(f"Section has no valid section entries in Title {title_num}, skipping")
                        skipped_count += 1
                        continue

                    # Update with validated sections
                    parsed_sections["section"] = valid_sections
                    sections.append(parsed_sections)
                    processed_count += 1

                    # Log progress for large titles
                    if processed_count % 100 == 0:
                        logger.debug(f"Processed {processed_count} sections in Title {title_num}")
                elif parsed_sections is None:
                    skipped_count += 1
                else:
                    logger.warning(f"Unexpected return type from parse_section in Title {title_num}: {type(parsed_sections)}")
                    skipped_count += 1
            except Exception as e:
                logger.error(f"Unexpected error processing section in Title {title_num}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                skipped_count += 1
                continue

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid sections in Title {title_num} (found {len(section_elems)} total, processed {processed_count})")
        else:
            logger.info(f"Successfully parsed {len(sections)} sections from Title {title_num} (all {len(section_elems)} sections processed)")

        if len(sections) == 0:
            logger.warning(f"No valid sections extracted from Title {title_num} (found {len(section_elems)} sections in XML)")

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
    parser = argparse.ArgumentParser(description="Convert US Code XML files to JSON format")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing XML files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (default: us_code.json in input directory)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads for parallel processing (default: 4)")

    args = parser.parse_args()

    # Set up paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_dir / "us_code.json"

    # Parse XML files
    sections = parse_all_titles(input_dir, num_workers=args.workers)

    if not sections:
        logger.error("No sections parsed. Check XML files and parsing logic.")
        return 1

    # Create JSON output
    create_json_output(sections, output_path)

    logger.info("Conversion complete!")
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

