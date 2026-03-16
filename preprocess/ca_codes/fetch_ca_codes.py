# fetch_ca_codes.py
"""
Fetch all California State Codes from leginfo.legislature.ca.gov and populate ca_code.json.
Downloads and processes all 29 major California codes from the official California legislative website.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_ca_codes")

# Thread-safe locks for shared resources
progress_lock = Lock()
save_lock = Lock()
rate_limit_lock = Lock()
last_request_time = [0.0]  # Use list to allow modification in nested functions

# California Legislative Information Base URL
LEGINFO_BASE = "https://leginfo.legislature.ca.gov"
LEGINFO_CODES_URL = f"{LEGINFO_BASE}/faces/codes.xhtml"
LEGINFO_SECTION_URL = f"{LEGINFO_BASE}/faces/codes_displaySection.xhtml"

# All 29 major California codes with abbreviations
CA_CODES = {
    "Civil Code": "CIV",
    "Code of Civil Procedure": "CCP",
    "Commercial Code": "COM",
    "Corporations Code": "CORP",
    "Education Code": "EDC",
    "Elections Code": "ELEC",
    "Evidence Code": "EVID",
    "Family Code": "FAM",
    "Financial Code": "FIN",
    "Fish and Game Code": "FGC",
    "Food and Agricultural Code": "FAC",
    "Government Code": "GOV",
    "Harbors and Navigation Code": "HNC",
    "Health and Safety Code": "HSC",
    "Insurance Code": "INS",
    "Labor Code": "LAB",
    "Military and Veterans Code": "MVC",
    "Penal Code": "PEN",
    "Probate Code": "PROB",
    "Public Contract Code": "PCC",
    "Public Resources Code": "PRC",
    "Public Utilities Code": "PUC",
    "Revenue and Taxation Code": "RTC",
    "Streets and Highways Code": "SHC",
    "Unemployment Insurance Code": "UIC",
    "Vehicle Code": "VEH",
    "Water Code": "WAT",
    "Welfare and Institutions Code": "WIC",
    "Business and Professions Code": "BPC"
}

def get_headers(referer: str = None) -> Dict[str, str]:
    """Get HTTP headers for requests. More browser-like to avoid 403 errors."""
    if referer is None:
        referer = "https://www.google.com/"

    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none" if "google" in referer else "same-origin",
        "Sec-Fetch-User": "?1",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Referer": referer,
        "DNT": "1"
    }

def fetch_with_retry(url: str, params: Optional[Dict] = None, retries: int = None, min_delay: float = 0.5, max_retries: int = None, session: Optional[requests.Session] = None) -> Optional[requests.Response]:
    """Fetch URL with retry logic and rate limiting. Will retry indefinitely until success.
    Uses session for JSF pages to maintain state."""
    # If retries is None, retry indefinitely
    infinite_retry = (retries is None)
    if retries is None:
        retries = 999999  # Very large number for infinite retry
    if max_retries is None:
        max_retries = retries

    # Use session to maintain state and cookies
    if session is None:
        session = requests.Session()
        # Start with Google referer to look like natural navigation
        session.headers.update(get_headers("https://www.google.com/"))

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
            # For Justia/FindLaw, visit homepage first to establish session and avoid 403
            if attempt == 0:
                if "justia" in url:
                    try:
                        logger.debug("Visiting Justia homepage first to establish session...")
                        session.get("https://law.justia.com/", timeout=(30, 60))
                        session.headers.update(get_headers("https://law.justia.com/"))
                        time.sleep(2)
                    except Exception as e:
                        logger.debug(f"Homepage visit failed (non-critical): {e}")
                elif "findlaw" in url:
                    try:
                        logger.debug("Visiting FindLaw homepage first to establish session...")
                        session.get("https://codes.findlaw.com/", timeout=(30, 60))
                        session.headers.update(get_headers("https://codes.findlaw.com/"))
                        time.sleep(2)
                    except Exception as e:
                        logger.debug(f"Homepage visit failed (non-critical): {e}")

            # Justia/FindLaw are usually fast, use standard timeouts
            timeout_val = (30, 60)
            response = session.get(url, params=params, timeout=timeout_val)
            response.raise_for_status()
            if attempt > 0:
                logger.info(f"Successfully fetched {url} on attempt {attempt + 1}")
            return response
        except requests.exceptions.Timeout as e:
            # Timeout errors - exponential backoff with longer waits
            wait_time = min((2 ** attempt) * 5, 300)  # Cap at 5 minutes
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
            if infinite_retry:
                logger.info(f"Will keep retrying indefinitely. Attempt {attempt + 1}/{max_retries if max_retries < 999999 else '∞'}")
            time.sleep(wait_time)
            attempt += 1
        except requests.exceptions.ConnectionError as e:
            # Connection errors - exponential backoff
            wait_time = min((2 ** attempt) * 3, 300)  # Cap at 5 minutes
            logger.warning(f"Connection error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
            if infinite_retry:
                logger.info(f"Will keep retrying indefinitely. Attempt {attempt + 1}/{max_retries if max_retries < 999999 else '∞'}")
            time.sleep(wait_time)
            attempt += 1
        except requests.exceptions.HTTPError as e:
            # HTTP errors - retry for 5xx, 429 (rate limit), and 403
            status_code = e.response.status_code if e.response else 0
            if status_code >= 500 or status_code in [403, 429]:
                wait_time = min((2 ** attempt) * 5, 300)  # Longer wait for API rate limits
                if status_code == 429:
                    logger.warning(f"HTTP 429 Rate Limited on attempt {attempt + 1} for {url}. Waiting longer...")
                elif status_code == 403:
                    logger.warning(f"HTTP 403 Forbidden on attempt {attempt + 1} for {url}. Retrying...")
                else:
                    logger.warning(f"HTTP {status_code} error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
            else:
                logger.error(f"HTTP {status_code if e.response else 'unknown'} error for {url}: {e}")
                return None
        except Exception as e:
            wait_time = min((2 ** attempt) * 2, 120)
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying in {wait_time}s...")
            if infinite_retry:
                logger.info(f"Will keep retrying indefinitely. Attempt {attempt + 1}/{max_retries if max_retries < 999999 else '∞'}")
            time.sleep(wait_time)
            attempt += 1

    if not infinite_retry:
        logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

def get_section_url(code_abbrev: str, section_num: str) -> str:
    """Generate URL for a specific section on leginfo.legislature.ca.gov."""
    # URL format: https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?sectionNum=1.&lawCode=CIV
    # Section numbers need to end with a period
    section_num_formatted = section_num if section_num.endswith('.') else f"{section_num}."
    return f"{LEGINFO_SECTION_URL}?sectionNum={section_num_formatted}&lawCode={code_abbrev}"

def fetch_section_detail(code_abbrev: str, section_num: str, min_delay: float = 0.5, session: Optional[requests.Session] = None, source: str = None) -> Optional[Dict[str, Any]]:
    """Fetch detailed information for a specific section from leginfo.legislature.ca.gov."""
    # Rate limiting
    time.sleep(min_delay)

    url = get_section_url(code_abbrev, section_num)

    # Use provided session or create new one
    if session is None:
        session = requests.Session()
        session.headers.update(get_headers())

    try:
        response = fetch_with_retry(url, session=session, min_delay=min_delay)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Find the section content - leginfo uses specific divs/classes
        section_text = ""
        section_title = f"Section {section_num}"

        # Try to find section content in various possible locations
        # Common patterns on leginfo.legislature.ca.gov
        content_selectors = [
            "div.sectionContent",
            "div#sectionContent",
            "div.section",
            "div.content",
            "div.mainContent",
            "div.bodytext",
            "div[class*='section']",
            "div[class*='content']",
            "div[class*='text']",
            "article",
            "main",
            "div.rich-panel-body",  # JSF RichFaces panel
            "div.rich-panel",  # JSF RichFaces panel
        ]

        section_content = None
        for selector in content_selectors:
            candidates = soup.select(selector)
            for candidate in candidates:
                text = candidate.get_text(strip=True)
                # Skip navigation/UI elements
                if any(keyword in text.lower() for keyword in ["select code", "keyword", "search", "navigation", "menu", "cookie"]):
                    continue
                # Must have substantial content
                if len(text) > 100:
                    section_content = candidate
                    break
            if section_content:
                break

        if section_content:
            # Extract text, preserving structure
            section_text = section_content.get_text(separator="\n", strip=True)
        else:
            # Fallback: get all text from body, excluding navigation
            body = soup.find("body")
            if body:
                # Remove navigation elements
                for element in body.find_all(["nav", "form", "header", "footer", "script", "style"]):
                    element.decompose()
                section_text = body.get_text(separator="\n", strip=True)

        # Try to extract section title
        title_elem = soup.find(["h1", "h2", "h3", "h4"], class_=re.compile(r"title|heading", re.I))
        if not title_elem:
            title_elem = soup.find("span", class_=re.compile(r"title|heading", re.I))
        if not title_elem:
            # Look for "Section X" in the text
            section_match = re.search(rf"Section\s+{re.escape(section_num)}[\.:]?\s*(.+?)(?:\n|$)", section_text, re.IGNORECASE)
            if section_match:
                section_title = section_match.group(1).strip()[:200]  # Limit title length
        else:
            section_title = title_elem.get_text(strip=True)

        # Clean up text
        section_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', section_text)
        section_text = re.sub(r'[ \t]+', ' ', section_text)
        section_text = section_text.strip()

        # Clean up title
        section_title = re.sub(r'\s+', ' ', section_title).strip()

        # Remove common navigation patterns
        patterns_to_remove = [
            r"Code:\s*Select Code.*?Keyword\(s\):",
            r"Article:\s*Section:",
        ]
        for pattern in patterns_to_remove:
            section_text = re.sub(pattern, "", section_text, flags=re.IGNORECASE | re.DOTALL)
        section_text = re.sub(r'\s+', ' ', section_text).strip()

        # Final validation - must have meaningful content
        if not section_text or len(section_text) < 20:
            logger.warning(f"Section {section_num} from {code_abbrev} has insufficient content")
            return None

        return {
            "text": section_text,
            "title": section_title,
            "section_num": section_num
        }
    except Exception as e:
        logger.warning(f"Error fetching section {section_num} from {code_abbrev}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

        # Remove select elements and their parent containers (these are the code selection dropdowns)
        for select in soup.find_all("select"):
            # Remove the select and its parent container if it's a form element
            parent = select.find_parent(["form", "div", "span"])
            if parent:
                parent.decompose()
            else:
                select.decompose()

        # Remove input elements (search boxes, etc.)
        for input_elem in soup.find_all("input", type=re.compile(r"text|search|submit", re.I)):
            parent = input_elem.find_parent(["form", "div", "span"])
            if parent and any(kw in parent.get_text().lower() for kw in ["select", "code", "keyword"]):
                parent.decompose()
            else:
                input_elem.decompose()

        # Remove elements with common navigation/form classes
        for element in soup.find_all(class_=re.compile(r"nav|menu|form|search|header|footer|sidebar|breadcrumb|filter|dropdown", re.I)):
            element.decompose()

        # Remove elements with IDs that suggest navigation/forms
        for element in soup.find_all(id=re.compile(r"nav|menu|form|search|header|footer|sidebar|breadcrumb|filter|dropdown", re.I)):
            element.decompose()

        # Remove any div/span that contains "Select Code" or "Keyword" text
        for element in soup.find_all(["div", "span", "p"]):
            text = element.get_text(strip=True)
            if any(kw in text.lower() for kw in ["select code", "keyword(s)", "code: select"]):
                element.decompose()

        # Try multiple selectors to find section content - prioritize content areas
        section_content = None
        for selector in [
            "div.sectionContent",
            "div#sectionContent",
            "div.section",
            "div.content",
            "div.mainContent",
            "div.bodytext",
            "div[class*='section']",
            "div[class*='content']",
            "div[class*='text']",
            "article",
            "main"
        ]:
            candidates = soup.select(selector)
            for candidate in candidates:
                # Check if this looks like actual content (not navigation)
                text = candidate.get_text(strip=True)
                # Skip if it looks like navigation/search form
                if any(keyword in text.lower() for keyword in ["select code", "keyword", "search", "navigation", "menu"]):
                    continue
                # Skip if too short (likely navigation)
                if len(text) < 50:
                    continue
                section_content = candidate
                break
            if section_content:
                break

        if not section_content:
            # Fallback: look for main content area, excluding navigation
            # Try to find the largest text block that looks like legal content
            candidates = []
            for elem in soup.find_all(["div", "article", "section", "main"]):
                text = elem.get_text(strip=True)
                # Skip navigation/UI elements
                if any(keyword in text.lower() for keyword in ["select code", "keyword", "search", "cookie", "privacy", "terms"]):
                    continue
                # Must have substantial content
                if len(text) > 200:
                    # Check if it contains legal text patterns
                    if re.search(r'(section|subsection|paragraph|subdivision)', text, re.IGNORECASE):
                        candidates.append((elem, len(text)))

            # Sort by length and pick the largest
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                section_content = candidates[0][0]

        if section_content:
            # Extract text, preserving some structure
            section_text = section_content.get_text(separator=" ", strip=True)
            # Clean up excessive whitespace
            section_text = re.sub(r'\s+', ' ', section_text).strip()

            # Remove common navigation/search text patterns
            patterns_to_remove = [
                r"Code:\s*Select Code.*?Keyword\(s\):",
                r"Article:\s*Section:",
                r"Code:\s*Select Code\s+All.*?Section:",
            ]
            for pattern in patterns_to_remove:
                section_text = re.sub(pattern, "", section_text, flags=re.IGNORECASE | re.DOTALL)

            section_text = re.sub(r'\s+', ' ', section_text).strip()
        else:
            # Last resort: get all text from body, but filter out navigation
            body = soup.find("body")
            if body:
                # Remove navigation elements again
                for element in body.find_all(["nav", "form", "header", "footer"]):
                    element.decompose()
                section_text = body.get_text(separator=" ", strip=True)
                section_text = re.sub(r'\s+', ' ', section_text).strip()

                # Remove navigation patterns
                patterns_to_remove = [
                    r"Code:\s*Select Code.*?Keyword\(s\):",
                    r"Article:\s*Section:",
                ]
                for pattern in patterns_to_remove:
                    section_text = re.sub(pattern, "", section_text, flags=re.IGNORECASE | re.DOTALL)
                section_text = re.sub(r'\s+', ' ', section_text).strip()
            else:
                section_text = ""

        # Validate that we got actual content, not just navigation
        if any(keyword in section_text.lower() for keyword in ["select code", "keyword(s):", "code: select code"]):
            logger.warning(f"Section {section_num} from {code_abbrev} appears to contain navigation text - trying alternative extraction")

            # Try to find the actual section content using more specific selectors
            # Look for divs that contain section numbers or legal text
            content_candidates = soup.find_all(["div", "p", "span"], string=re.compile(rf"Section\s+{re.escape(section_num)}", re.I))
            for candidate in content_candidates:
                parent = candidate.find_parent(["div", "article", "section"])
                if parent:
                    text = parent.get_text(separator=" ", strip=True)
                    # Check if this looks like real legal content
                    if len(text) > 100 and not any(kw in text.lower() for kw in ["select code", "keyword"]):
                        section_text = text
                        break

            # If still navigation text, try to find content after navigation patterns
            if any(keyword in section_text.lower() for keyword in ["select code", "keyword(s):"]):
                # Look for text that comes after common navigation patterns
                patterns = [
                    r"Keyword\(s\):\s*(.+?)(?=Section\s+\d+|$)",
                    r"Section:\s*(.+?)(?=Code:|$)",
                    r"(?:Section\s+\d+[\.:]?\s*)(.+?)(?=Code:|Select|$)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, section_text, re.IGNORECASE | re.DOTALL)
                    if match and len(match.group(1).strip()) > 50:
                        section_text = match.group(1).strip()
                        break

                # Final check - if still navigation, return None
                if any(keyword in section_text.lower() for keyword in ["select code", "keyword(s):", "code: select code"]):
                    logger.warning(f"Section {section_num} from {code_abbrev} - could not extract real content, skipping")
                    return None

        # Try to extract section title
        title_elem = soup.find(["h1", "h2", "h3", "h4"], class_=re.compile(r"title|heading", re.I))
        if not title_elem:
            # Look for title in common locations
            title_elem = soup.find("span", class_=re.compile(r"title|heading", re.I))
        if not title_elem:
            # Try finding title in section content
            if section_content:
                title_elem = section_content.find(["h1", "h2", "h3", "h4"])

        section_title = title_elem.get_text(strip=True) if title_elem else f"Section {section_num}"

        # Clean up title - remove navigation text
        section_title = re.sub(r'Code:.*?Section:', '', section_title, flags=re.IGNORECASE)
        section_title = re.sub(r'\s+', ' ', section_title).strip()

        # Final validation - must have meaningful content
        if len(section_text) < 20:
            logger.warning(f"Section {section_num} from {code_abbrev} has insufficient content ({len(section_text)} chars)")
            return None

        return {
            "text": section_text,
            "title": section_title,
            "section_num": section_num
        }
    except Exception as e:
        logger.warning(f"Error parsing section {section_num} from {code_abbrev}: {e}")
        return None

def process_section(code_name: str, code_abbrev: str, section_num: str, hierarchy: Dict[str, Dict[str, str]],
                   min_delay: float = 0.5, session: Optional[requests.Session] = None, source: str = None) -> Optional[Dict[str, Any]]:
    """Process a single section and return its data. Uses LegiScan API."""
    # Fetch section detail
    section_detail = fetch_section_detail(code_abbrev, section_num, min_delay=min_delay, session=session, source=source)

    # Get hierarchy for this section
    section_hierarchy = hierarchy.get(section_num, {})

    if section_detail:
        section_title = section_detail.get("title", f"Section {section_num}")
        section_text = section_detail.get("text", "")

        # Validate we have meaningful content
        if not section_text or len(section_text.strip()) < 20:
            logger.warning(f"  Section {code_name} {section_num} has no meaningful content - skipping")
            return None
    else:
        logger.warning(f"  Could not fetch detail for {code_name} Section {section_num} - skipping")
        return None

    return {
        "code": code_name,
        "division": section_hierarchy.get("division", ""),
        "part": section_hierarchy.get("part", ""),
        "chapter": section_hierarchy.get("chapter", ""),
        "section": f"Section {section_num}",
        "title": section_title,
        "clauses": [
            {
                "number": 1,
                "title": section_title,
                "text": section_text
            }
        ]
    }

def discover_section_numbers(code_abbrev: str, session: Optional[requests.Session] = None) -> List[str]:
    """Discover all section numbers for a code by trying common patterns.

    Since leginfo.legislature.ca.gov doesn't provide a simple list endpoint,
    we'll try to discover sections by attempting common section number patterns.
    """
    section_numbers = []

    # Common section number ranges - most codes have sections numbered 1-10000+
    # We'll try a smart approach: start with 1 and increment, stopping when we get consistent failures
    logger.info(f"Discovering section numbers for {code_abbrev}...")

    if session is None:
        session = requests.Session()
        session.headers.update(get_headers())

    consecutive_failures = 0
    max_consecutive_failures = 50  # Stop after 50 consecutive failures
    current_section = 1
    max_sections = 50000  # Safety limit

    while current_section <= max_sections and consecutive_failures < max_consecutive_failures:
        section_num = str(current_section)
        url = get_section_url(code_abbrev, section_num)

        try:
            response = session.get(url, timeout=10)
            # Check if we got a valid section page (not error page)
            if response.status_code == 200:
                # Check if page contains actual section content (not error message)
                if "section" in response.text.lower() and len(response.text) > 1000:
                    section_numbers.append(section_num)
                    consecutive_failures = 0
                    if len(section_numbers) % 100 == 0:
                        logger.info(f"  Found {len(section_numbers)} sections so far (checked up to {current_section})...")
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
        except Exception as e:
            consecutive_failures += 1

        current_section += 1

        # Small delay to avoid overwhelming the server
        if current_section % 10 == 0:
            time.sleep(0.1)

    logger.info(f"Discovered {len(section_numbers)} sections for {code_abbrev}")
    return section_numbers

def process_code(code_name: str, code_abbrev: str, progress_callback=None, num_workers: int = 4,
                min_delay: float = 0.5) -> List[Dict[str, Any]]:
    """Process a single California code from leginfo.legislature.ca.gov and return all sections using multi-workers."""
    logger.info(f"Processing {code_name} ({code_abbrev}) with {num_workers} workers from leginfo.legislature.ca.gov...")
    sections = []

    # Create session for this code
    code_session = requests.Session()
    code_session.headers.update(get_headers())

    # Discover section numbers
    logger.info(f"Discovering sections for {code_name}...")
    section_numbers = discover_section_numbers(code_abbrev, session=code_session)

    if not section_numbers:
        logger.warning(f"No sections found for {code_name}")
        return sections

    logger.info(f"Found {len(section_numbers)} sections in {code_name}")

    # Create empty hierarchy (we'll extract it from section content if available)
    hierarchy = {section_num: {"division": "", "part": "", "chapter": ""} for section_num in section_numbers}

    # Process sections in parallel using ThreadPoolExecutor
    processed_count = [0]  # Use list for thread-safe counter

    def update_progress():
        with progress_lock:
            processed_count[0] += 1
            current = processed_count[0]
            if current % 50 == 0 or current == len(section_numbers):
                logger.info(f"  Processed {current}/{len(section_numbers)} sections for {code_name}")
                if progress_callback:
                    progress_callback(code_name, current, len(section_numbers))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all section processing tasks
        future_to_section = {
            executor.submit(process_section, code_name, code_abbrev, section_num, hierarchy, min_delay, None, None): section_num
            for section_num in section_numbers
        }

        # Collect results as they complete
        for future in as_completed(future_to_section):
            section_num = future_to_section[future]
            try:
                section_data = future.result()
                if section_data:
                    sections.append(section_data)
                update_progress()
            except Exception as e:
                logger.error(f"Error processing section {section_num} from {code_name}: {e}")
                update_progress()

        # Filter out None results (sections with no valid content)
        sections = [s for s in sections if s is not None]

    logger.info(f"Completed {code_name}: {len(sections)} sections processed")
    return sections

def save_progress(sections: List[Dict[str, Any]], output_path: Path):
    """Save current progress to JSON file (thread-safe)."""
    with save_lock:
        output_data = {
            "data": {
                "california_codes": {
                    "codes": sections
                }
            }
        }

        # Write to temporary file first, then rename (atomic write)
        temp_path = output_path.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        temp_path.replace(output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Progress saved: {len(sections)} sections ({file_size:.2f} MB)")

def main():
    """Main function to fetch all California codes."""
    parser = argparse.ArgumentParser(description="Fetch California State Codes with multi-worker support")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel section fetching (default: 4)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Minimum delay between requests in seconds (default: 0.8, increase if getting timeouts)"
    )
    parser.add_argument(
        "--parallel-codes",
        action="store_true",
        help="Process multiple codes in parallel (uses additional workers)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = base_dir / "Data" / "Knowledge" / "ca_code.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_sections = []
    total_codes = len(CA_CODES)
    num_workers = max(1, args.workers)
    min_delay = max(0.1, args.delay)

    logger.info(f"Starting to fetch all {total_codes} California codes...")
    logger.info(f"Using {num_workers} workers per code with {min_delay}s minimum delay")
    logger.info(f"Fetching from: {LEGINFO_BASE}")
    if args.parallel_codes:
        logger.info("Processing codes in parallel as well")
    logger.info("This will take a significant amount of time (potentially hours).")
    logger.info("Progress will be saved periodically.")

    start_time = time.time()

    if args.parallel_codes:
        # Process codes in parallel
        def process_single_code(code_name: str, code_abbrev: str, code_index: int) -> Tuple[str, List[Dict[str, Any]]]:
            """Process a single code and return its name and sections."""
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing code {code_index}/{total_codes}: {code_name}")
            logger.info(f"{'='*60}")

            try:
                sections = process_code(code_name, code_abbrev, num_workers=num_workers, min_delay=min_delay)
                logger.info(f"Completed {code_name}: {len(sections)} sections")
                return code_name, sections
            except Exception as e:
                logger.error(f"Error processing {code_name}: {e}")
                return code_name, []

        # Use ThreadPoolExecutor to process codes in parallel
        with ThreadPoolExecutor(max_workers=min(4, total_codes)) as executor:
            future_to_code = {
                executor.submit(process_single_code, code_name, code_abbrev, i+1): (code_name, code_abbrev)
                for i, (code_name, code_abbrev) in enumerate(CA_CODES.items())
            }

            for future in as_completed(future_to_code):
                code_name, code_abbrev = future_to_code[future]
                try:
                    name, sections = future.result()
                    all_sections.extend(sections)

                    # Save progress after each code completes
                    save_progress(all_sections, output_path)
                    logger.info(f"Total sections so far: {len(all_sections)}")
                except Exception as e:
                    logger.error(f"Error getting result for {code_name}: {e}")
    else:
        # Process codes sequentially (but sections in parallel)
        for i, (code_name, code_abbrev) in enumerate(CA_CODES.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing code {i}/{total_codes}: {code_name}")
            logger.info(f"{'='*60}")

            try:
                sections = process_code(code_name, code_abbrev, num_workers=num_workers, min_delay=min_delay)
                all_sections.extend(sections)

                # Save progress after each code
                save_progress(all_sections, output_path)

                logger.info(f"Total sections so far: {len(all_sections)}")

                # Rate limiting between codes - longer delay to avoid overwhelming server
                if i < total_codes:
                    wait_time = 15  # Wait 15 seconds between codes to avoid blocking
                    logger.info(f"Waiting {wait_time}s before next code to avoid server overload...")
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error processing {code_name}: {e}")
                logger.error("Continuing with next code...")
                continue

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

