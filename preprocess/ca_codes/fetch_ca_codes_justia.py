#!/usr/bin/env python3
"""
Fetch all California State Codes from Justia (law.justia.com) and populate ca_code.json.
Recursively scrapes all child content (sections, articles, divisions, etc.) from Justia.
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
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_ca_codes_justia")

# Thread-safe locks
progress_lock = Lock()
save_lock = Lock()
rate_limit_lock = Lock()
last_request_time = [0.0]

# Justia base URL
JUSTIA_BASE = "https://law.justia.com/codes/california/2024"

# All California codes with their Justia URL slugs
CA_CODES = {
    "Business and Professions Code": ("BPC", "business-and-professions-code"),
    "Civil Code": ("CIV", "civil-code"),
    "Code of Civil Procedure": ("CCP", "code-of-civil-procedure"),
    "Commercial Code": ("COM", "commercial-code"),
    "Corporations Code": ("CORP", "corporations-code"),
    "Education Code": ("EDC", "education-code"),
    "Elections Code": ("ELEC", "elections-code"),
    "Evidence Code": ("EVID", "evidence-code"),
    "Family Code": ("FAM", "family-code"),
    "Financial Code": ("FIN", "financial-code"),
    "Fish and Game Code": ("FGC", "fish-and-game-code"),
    "Food and Agricultural Code": ("FAC", "food-and-agricultural-code"),
    "Government Code": ("GOV", "government-code"),
    "Harbors and Navigation Code": ("HNC", "harbors-and-navigation-code"),
    "Health and Safety Code": ("HSC", "health-and-safety-code"),
    "Insurance Code": ("INS", "insurance-code"),
    "Labor Code": ("LAB", "labor-code"),
    "Military and Veterans Code": ("MVC", "military-and-veterans-code"),
    "Penal Code": ("PEN", "penal-code"),
    "Probate Code": ("PROB", "probate-code"),
    "Public Contract Code": ("PCC", "public-contract-code"),
    "Public Resources Code": ("PRC", "public-resources-code"),
    "Public Utilities Code": ("PUC", "public-utilities-code"),
    "Revenue and Taxation Code": ("RTC", "revenue-and-taxation-code"),
    "Streets and Highways Code": ("SHC", "streets-and-highways-code"),
    "Unemployment Insurance Code": ("UIC", "unemployment-insurance-code"),
    "Vehicle Code": ("VEH", "vehicle-code"),
    "Water Code": ("WAT", "water-code"),
    "Welfare and Institutions Code": ("WIC", "welfare-and-institutions-code")
}

def get_headers() -> Dict[str, str]:
    """Get HTTP headers for requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://law.justia.com/",
        "DNT": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin"
    }

def fetch_with_retry(url: str, retries: int = 5, min_delay: float = 0.5,
                     session: Optional[requests.Session] = None) -> Optional[requests.Response]:
    """Fetch URL with retry logic and rate limiting."""
    if session is None:
        session = requests.Session()
        session.headers.update(get_headers())

    attempt = 0
    while attempt < retries:
        # Rate limiting
        with rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - last_request_time[0]
            if time_since_last < min_delay:
                time.sleep(min_delay - time_since_last)
            last_request_time[0] = time.time()

        try:
            response = session.get(url, timeout=(30, 60))
            response.raise_for_status()
            if attempt > 0:
                logger.info(f"Successfully fetched {url} on attempt {attempt + 1}")
            return response
        except requests.exceptions.Timeout as e:
            wait_time = min((2 ** attempt) * 2, 60)
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
        except requests.exceptions.ConnectionError as e:
            wait_time = min((2 ** attempt) * 2, 60)
            logger.warning(f"Connection error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code >= 500:
                wait_time = min((2 ** attempt) * 2, 30)
                logger.warning(f"HTTP {e.response.status_code} error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
            else:
                logger.error(f"HTTP {e.response.status_code if e.response else 'unknown'} error for {url}: {e}")
                return None
        except Exception as e:
            wait_time = min((2 ** attempt) * 2, 30)
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1

    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

def get_code_base_url(code_slug: str) -> str:
    """Get the base URL for a code on Justia."""
    return f"{JUSTIA_BASE}/{code_slug}/"

def extract_links_from_page(soup: BeautifulSoup, base_url: str, code_slug: str) -> List[str]:
    """Extract all relevant links (sections, articles, divisions, etc.) from a page."""
    links = set()

    # Find all links that point to sections, articles, divisions, etc.
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if not href:
            continue

        # Make absolute URL
        full_url = urljoin(base_url, href)

        # Normalize URL (remove fragments, trailing slashes)
        full_url = full_url.split('#')[0].rstrip('/')
        base_url_normalized = base_url.rstrip('/')

        # Check if this is a relevant link (section, article, division, part, chapter, etc.)
        if code_slug in full_url and "law.justia.com" in full_url:
            # Match patterns like:
            # - /section-123/
            # - /article-1/
            # - /division-1/
            # - /part-1/
            # - /chapter-1/
            # - /title-1/
            if re.search(r'/(?:section|article|division|part|chapter|title)[_-]?\d', full_url, re.I):
                links.add(full_url)
            # Also match the base code page itself
            elif full_url == base_url_normalized:
                links.add(full_url)

    # Also try to extract section numbers from text (for TOC pages)
    # Look for patterns like "Section 123" or "§ 123" in link text
    for link in soup.find_all("a", href=True):
        link_text = link.get_text(strip=True)
        # Match "Section 123" or "§ 123" patterns
        section_match = re.search(r'(?:Section|§)\s+(\d+(?:\.\d+)*)', link_text, re.I)
        if section_match:
            section_num = section_match.group(1)
            section_url = f"{base_url_normalized}/section-{section_num}/"
            if code_slug in section_url:
                links.add(section_url)

    return sorted(list(links))

def extract_hierarchy_from_url(url: str, code_slug: str) -> Dict[str, str]:
    """Extract hierarchy information (division, part, chapter, etc.) from URL or page structure."""
    hierarchy = {
        "division": "",
        "part": "",
        "chapter": "",
        "article": "",
        "title": ""
    }

    # Try to extract from URL
    url_lower = url.lower()

    # Match patterns in URL
    div_match = re.search(r'/division[_-]?(\d+[a-z]?)', url_lower)
    if div_match:
        hierarchy["division"] = f"Division {div_match.group(1).upper()}"

    part_match = re.search(r'/part[_-]?(\d+[a-z]?)', url_lower)
    if part_match:
        hierarchy["part"] = f"Part {part_match.group(1).upper()}"

    chap_match = re.search(r'/chapter[_-]?(\d+[a-z]?)', url_lower)
    if chap_match:
        hierarchy["chapter"] = f"Chapter {chap_match.group(1).upper()}"

    art_match = re.search(r'/article[_-]?(\d+[a-z]?)', url_lower)
    if art_match:
        hierarchy["article"] = f"Article {art_match.group(1).upper()}"

    title_match = re.search(r'/title[_-]?(\d+[a-z]?)', url_lower)
    if title_match:
        hierarchy["title"] = f"Title {title_match.group(1).upper()}"

    return hierarchy

def extract_hierarchy_from_content(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """Extract hierarchy information from page content (breadcrumbs, headings, etc.)."""
    hierarchy = extract_hierarchy_from_url(url, "")

    # Look for breadcrumbs
    breadcrumbs = soup.find_all(class_=re.compile(r'breadcrumb', re.I))
    for breadcrumb in breadcrumbs:
        text = breadcrumb.get_text()

        div_match = re.search(r'Division\s+(\d+[A-Z]?)', text, re.I)
        if div_match:
            hierarchy["division"] = f"Division {div_match.group(1)}"

        part_match = re.search(r'Part\s+(\d+[A-Z]?)', text, re.I)
        if part_match:
            hierarchy["part"] = f"Part {part_match.group(1)}"

        chap_match = re.search(r'Chapter\s+(\d+[A-Z]?)', text, re.I)
        if chap_match:
            hierarchy["chapter"] = f"Chapter {chap_match.group(1)}"

        art_match = re.search(r'Article\s+(\d+[A-Z]?)', text, re.I)
        if art_match:
            hierarchy["article"] = f"Article {art_match.group(1)}"

        title_match = re.search(r'Title\s+(\d+[A-Z]?)', text, re.I)
        if title_match:
            hierarchy["title"] = f"Title {title_match.group(1)}"

    # Look in headings
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = heading.get_text()

        div_match = re.search(r'Division\s+(\d+[A-Z]?)', text, re.I)
        if div_match:
            hierarchy["division"] = f"Division {div_match.group(1)}"

        part_match = re.search(r'Part\s+(\d+[A-Z]?)', text, re.I)
        if part_match:
            hierarchy["part"] = f"Part {part_match.group(1)}"

        chap_match = re.search(r'Chapter\s+(\d+[A-Z]?)', text, re.I)
        if chap_match:
            hierarchy["chapter"] = f"Chapter {chap_match.group(1)}"

        art_match = re.search(r'Article\s+(\d+[A-Z]?)', text, re.I)
        if art_match:
            hierarchy["article"] = f"Article {art_match.group(1)}"

    return hierarchy

def extract_section_number_from_url(url: str) -> Optional[str]:
    """Extract section number from URL."""
    match = re.search(r'/section[_-]?(\d+(?:\.\d+)*)', url, re.I)
    if match:
        return match.group(1)
    return None

def extract_content_from_page(soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
    """Extract content from a Justia page."""
    # Remove navigation, scripts, styles
    for element in soup.find_all(["nav", "script", "style", "noscript", "header", "footer"]):
        element.decompose()

    # Remove common navigation elements
    for element in soup.find_all(class_=re.compile(r'nav|menu|sidebar|breadcrumb|footer|header', re.I)):
        element.decompose()

    # Try multiple selectors to find main content
    content = None
    for selector in [
        "div.section-content",
        "div#section-content",
        "div.section",
        "div.content",
        "div.main-content",
        "div.main",
        "article",
        "main",
        "div[class*='content']",
        "div[class*='section']"
    ]:
        candidates = soup.select(selector)
        for candidate in candidates:
            text = candidate.get_text(strip=True)
            # Skip if too short or looks like navigation
            if len(text) > 100 and not any(kw in text.lower() for kw in ["select code", "keyword", "search", "navigation"]):
                content = candidate
                break
        if content:
            break

    if not content:
        # Fallback: use body but remove navigation
        body = soup.find("body")
        if body:
            for element in body.find_all(["nav", "header", "footer", "form"]):
                element.decompose()
            content = body

    if not content:
        return None

    # Extract text
    text = content.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()

    # Extract title
    title = None
    for heading in soup.find_all(["h1", "h2", "h3"]):
        heading_text = heading.get_text(strip=True)
        if len(heading_text) > 5 and len(heading_text) < 200:
            title = heading_text
            break

    if not title:
        # Try to extract from URL or content
        section_num = extract_section_number_from_url(url)
        if section_num:
            title = f"Section {section_num}"
        else:
            title = "Untitled"

    # Clean up title
    title = re.sub(r'\s+', ' ', title).strip()

    # Validate content
    if len(text) < 20:
        return None

    return {
        "text": text,
        "title": title
    }

def discover_all_pages(code_slug: str, base_url: str, session: Optional[requests.Session] = None,
                       visited: Optional[Set[str]] = None, max_depth: int = 10) -> Set[str]:
    """Recursively discover all pages (sections, articles, etc.) for a code."""
    if visited is None:
        visited = set()

    if max_depth <= 0:
        return visited

    # Normalize URL
    base_url_normalized = base_url.rstrip('/')

    if base_url_normalized in visited:
        return visited

    # Fetch the page
    response = fetch_with_retry(base_url_normalized + '/', session=session)
    if not response:
        logger.warning(f"Failed to fetch {base_url_normalized}")
        return visited

    visited.add(base_url_normalized)
    logger.debug(f"Discovered: {base_url_normalized}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all relevant links
    links = extract_links_from_page(soup, base_url_normalized, code_slug)

    # Separate structural pages from section pages
    structural_pages = []
    section_pages = []

    for link in links:
        link_normalized = link.rstrip('/')
        if link_normalized in visited:
            continue

        # Check if it's a structural page (division, part, chapter, article, title)
        if re.search(r'/(?:division|part|chapter|article|title)[_-]?\d', link_normalized, re.I):
            structural_pages.append(link_normalized)
        # Check if it's a section page
        elif re.search(r'/section[_-]?\d', link_normalized, re.I):
            section_pages.append(link_normalized)
        # Base page or other content
        else:
            structural_pages.append(link_normalized)

    # Add all section pages directly (no recursion needed)
    for section_url in section_pages:
        visited.add(section_url)
        logger.debug(f"Discovered section: {section_url}")

    # Recursively visit structural pages
    for link in structural_pages:
        if link not in visited and max_depth > 0:
            visited = discover_all_pages(code_slug, link, session, visited, max_depth - 1)

    return visited

def process_page(code_name: str, code_abbrev: str, url: str, session: Optional[requests.Session] = None) -> Optional[Dict[str, Any]]:
    """Process a single page and extract its content."""
    response = fetch_with_retry(url, session=session)
    if not response:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract content
    content_data = extract_content_from_page(soup, url)
    if not content_data:
        return None

    # Extract hierarchy
    hierarchy = extract_hierarchy_from_content(soup, url)

    # Extract section number if applicable
    section_num = extract_section_number_from_url(url)

    # Determine what type of content this is
    if section_num:
        section_label = f"Section {section_num}"
    elif re.search(r'/article-\d', url, re.I):
        section_label = "Article"
    elif re.search(r'/division-\d', url, re.I):
        section_label = "Division"
    elif re.search(r'/part-\d', url, re.I):
        section_label = "Part"
    elif re.search(r'/chapter-\d', url, re.I):
        section_label = "Chapter"
    elif re.search(r'/title-\d', url, re.I):
        section_label = "Title"
    else:
        section_label = "Overview"

    return {
        "code": code_name,
        "code_abbrev": code_abbrev,
        "division": hierarchy.get("division", ""),
        "part": hierarchy.get("part", ""),
        "chapter": hierarchy.get("chapter", ""),
        "article": hierarchy.get("article", ""),
        "title": hierarchy.get("title", ""),
        "section": section_label,
        "section_num": section_num or "",
        "content_title": content_data["title"],
        "url": url,
        "clauses": [
            {
                "number": 1,
                "title": content_data["title"],
                "text": content_data["text"]
            }
        ]
    }

def process_code(code_name: str, code_abbrev: str, code_slug: str, num_workers: int = 4,
                min_delay: float = 0.5) -> List[Dict[str, Any]]:
    """Process a single California code and return all sections."""
    logger.info(f"Processing {code_name} ({code_abbrev}) with {num_workers} workers...")

    base_url = get_code_base_url(code_slug)
    logger.info(f"Base URL: {base_url}")

    # Create session for this code
    code_session = requests.Session()
    code_session.headers.update(get_headers())

    # Discover all pages
    logger.info("Discovering all pages...")
    all_urls = discover_all_pages(code_slug, base_url, session=code_session, max_depth=10)
    logger.info(f"Found {len(all_urls)} pages to process")

    if not all_urls:
        logger.warning(f"No pages found for {code_name}")
        return []

    # Process pages in parallel
    sections = []
    processed_count = [0]

    def update_progress():
        with progress_lock:
            processed_count[0] += 1
            current = processed_count[0]
            if current % 50 == 0 or current == len(all_urls):
                logger.info(f"  Processed {current}/{len(all_urls)} pages for {code_name}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = {
            executor.submit(process_page, code_name, code_abbrev, url, code_session): url
            for url in all_urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                page_data = future.result()
                if page_data:
                    sections.append(page_data)
                update_progress()
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                update_progress()

    logger.info(f"Completed {code_name}: {len(sections)} pages processed")
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

        temp_path = output_path.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        temp_path.replace(output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Progress saved: {len(sections)} sections ({file_size:.2f} MB)")

def main():
    """Main function to fetch all California codes from Justia."""
    parser = argparse.ArgumentParser(description="Fetch California State Codes from Justia")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel page fetching (default: 4)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Minimum delay between requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Process only a specific code (e.g., 'CIV', 'BPC')"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = base_dir / "Data" / "Knowledge" / "ca_code_justia.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_sections = []

    # Filter codes if specific code requested
    codes_to_process = CA_CODES
    if args.code:
        codes_to_process = {
            name: (abbrev, slug)
            for name, (abbrev, slug) in CA_CODES.items()
            if abbrev.upper() == args.code.upper()
        }
        if not codes_to_process:
            logger.error(f"Code '{args.code}' not found. Available codes: {', '.join(abbrev for _, (abbrev, _) in CA_CODES.items())}")
            return

    total_codes = len(codes_to_process)
    num_workers = max(1, args.workers)
    min_delay = max(0.1, args.delay)

    logger.info(f"Starting to fetch {total_codes} California codes from Justia...")
    logger.info(f"Using {num_workers} workers per code with {min_delay}s minimum delay")
    logger.info("This will take a significant amount of time.")
    logger.info("Progress will be saved periodically.")

    start_time = time.time()

    # Process codes sequentially
    for i, (code_name, (code_abbrev, code_slug)) in enumerate(codes_to_process.items(), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing code {i}/{total_codes}: {code_name} ({code_abbrev})")
        logger.info(f"{'='*60}")

        try:
            sections = process_code(code_name, code_abbrev, code_slug, num_workers=num_workers, min_delay=min_delay)
            all_sections.extend(sections)

            # Save progress after each code
            save_progress(all_sections, output_path)

            logger.info(f"Total sections so far: {len(all_sections)}")

            # Rate limiting between codes
            if i < total_codes:
                wait_time = 10
                logger.info(f"Waiting {wait_time}s before next code...")
                time.sleep(wait_time)

        except Exception as e:
            logger.error(f"Error processing {code_name}: {e}", exc_info=True)
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

