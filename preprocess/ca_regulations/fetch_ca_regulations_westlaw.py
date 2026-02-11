# fetch_ca_regulations_westlaw.py
"""
Fetch comprehensive California Code of Regulations from Westlaw.
Downloads all Titles (1-28) from https://shared-govt.westlaw.com/calregs/
"""
import os
import sys
import json
import logging
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import re
from urllib.parse import urljoin, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_ca_regulations_westlaw")

# Westlaw base URL
WESTLAW_BASE = "https://shared-govt.westlaw.com"
WESTLAW_INDEX = "https://shared-govt.westlaw.com/calregs/Index?transitionType=Default&contextData=%28sc.Default%29"

# All California Code of Regulations Titles (from web search)
CA_REG_TITLES = {
    1: "General Provisions",
    2: "Administration",
    3: "Food and Agriculture",
    4: "Business Regulations",
    5: "Education",
    7: "Harbors and Navigation",
    8: "Industrial Relations",
    9: "Rehabilitative and Developmental Services",
    10: "Investment",
    11: "Law",
    12: "Military and Veterans Affairs",
    13: "Motor Vehicles",
    14: "Natural Resources",
    15: "Crime Prevention and Corrections",
    16: "Professional and Vocational Regulations",
    17: "Public Health",
    18: "Public Revenues",
    19: "Public Safety",
    20: "Public Utilities and Energy",
    21: "Public Works",
    22: "Social Security",
    23: "Waters",
    24: "Building Standards Code",
    25: "Housing and Community Development",
    26: "Toxics",
    27: "Environmental Protection",
    28: "Managed Health Care"
}

# Thread-safe locks
progress_lock = Lock()
save_lock = Lock()
rate_limit_lock = Lock()
last_request_time = [0.0]

def get_headers(referer: str = None) -> Dict[str, str]:
    """Get HTTP headers for requests. Browser-like to avoid blocking."""
    if referer is None:
        referer = WESTLAW_INDEX
    
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin" if referer.startswith(WESTLAW_BASE) else "none",
        "Sec-Fetch-User": "?1",
        "Referer": referer,
        "DNT": "1"
    }

def fetch_with_retry(url: str, session: Optional[requests.Session] = None, retries: int = 3, min_delay: float = 1.0) -> Optional[requests.Response]:
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
            
            # Check for authentication required
            if response.status_code == 401 or response.status_code == 403:
                logger.error(f"Authentication required for {url}. Status: {response.status_code}")
                logger.error("Westlaw may require a subscription or login. Please check if you have access.")
                return None
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 10
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                attempt += 1
                continue
            
            response.raise_for_status()
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
            if status_code >= 500:
                wait_time = min((2 ** attempt) * 5, 300)
                logger.warning(f"HTTP {status_code} error on attempt {attempt + 1} for {url}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempt += 1
            else:
                logger.error(f"HTTP {status_code} error for {url}: {e}")
                return None
        except Exception as e:
            wait_time = min((2 ** attempt) * 2, 120)
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
    
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

def extract_regulation_text(soup: BeautifulSoup) -> str:
    """Extract regulation text from parsed HTML, removing navigation and UI elements."""
    # Remove navigation, headers, footers
    for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # Remove script and style tags
    for element in soup.find_all(['script', 'style']):
        element.decompose()
    
    # Remove common UI elements
    for element in soup.find_all(class_=re.compile(r'nav|menu|header|footer|sidebar|breadcrumb|search|filter', re.I)):
        element.decompose()
    
    # Try to find main content area
    content_selectors = [
        'div.main-content',
        'div.content',
        'div.regulation-content',
        'article',
        'main',
        'div[class*="content"]',
        'div[class*="regulation"]'
    ]
    
    content = None
    for selector in content_selectors:
        candidates = soup.select(selector)
        for candidate in candidates:
            text = candidate.get_text(strip=True)
            if len(text) > 100:  # Must have substantial content
                content = candidate
                break
        if content:
            break
    
    if not content:
        # Fallback: get body text
        body = soup.find('body')
        if body:
            content = body
    
    if content:
        text = content.get_text(separator='\n', strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    return ""

def get_title_url_from_index(title_num: int, session: Optional[requests.Session] = None) -> Optional[str]:
    """Get the correct URL for a title from the index page."""
    if session is None:
        session = requests.Session()
        session.headers.update(get_headers())
    
    response = fetch_with_retry(WESTLAW_INDEX, session=session)
    if not response:
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Look for the link to this specific title
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Check if this is the title we're looking for
        if f"Title {title_num}" in text or f"Title {title_num}." in text:
            full_url = urljoin(WESTLAW_BASE, href)
            return full_url
    
    return None

def fetch_title_regulations(title_num: int, title_name: str, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    """Fetch all regulations for a specific title."""
    logger.info(f"Fetching Title {title_num}: {title_name}...")
    
    if session is None:
        session = requests.Session()
        session.headers.update(get_headers())
    
    regulations = []
    
    # Get the correct title URL from index
    title_url = get_title_url_from_index(title_num, session)
    if not title_url:
        logger.warning(f"Could not find URL for Title {title_num} from index")
        return regulations
    
    # Fetch the title page
    response = fetch_with_retry(title_url, session=session)
    if not response:
        logger.warning(f"Could not access Title {title_num} at {title_url}")
        return regulations
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Look for links to divisions, chapters, articles, or sections
    regulation_links = []
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Check if this looks like a regulation link (not navigation)
        if href.startswith('/calregs/') and not any(skip in href.lower() for skip in ['index', 'home', 'browse/home']):
            full_url = urljoin(WESTLAW_BASE, href)
            # Only include if it has meaningful text
            if text and len(text.strip()) > 3:
                regulation_links.append((full_url, text.strip()))
    
    logger.info(f"Found {len(regulation_links)} potential regulation links for Title {title_num}")
    
    # If we found links, fetch them; otherwise extract from current page
    if regulation_links:
        # Limit to avoid overwhelming (can increase later)
        for reg_url, reg_title in regulation_links[:200]:
            time.sleep(1.0)  # Rate limiting
            
            reg_response = fetch_with_retry(reg_url, session=session)
            if not reg_response:
                continue
            
            reg_soup = BeautifulSoup(reg_response.content, 'html.parser')
            reg_text = extract_regulation_text(reg_soup)
            
            # Clean up the text - remove navigation content
            if reg_text:
                # Remove common navigation patterns
                lines = reg_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip navigation lines
                    if any(skip in line.lower() for skip in ['state government sites', 'alaska case law', 'skip to navigation', 'skip to main content']):
                        continue
                    if len(line) > 5:  # Only keep substantial lines
                        cleaned_lines.append(line)
                
                reg_text = '\n'.join(cleaned_lines)
            
            if reg_text and len(reg_text) > 100:  # Must have substantial content
                # Extract structure from URL or title
                division = ""
                chapter = ""
                article = ""
                section = ""
                
                # Try to parse structure from URL
                url_lower = reg_url.lower()
                if 'division' in url_lower:
                    match = re.search(r'division[\/-]?(\d+)', url_lower, re.I)
                    if match:
                        division = f"Division {match.group(1)}"
                
                if 'chapter' in url_lower:
                    match = re.search(r'chapter[\/-]?(\d+)', url_lower, re.I)
                    if match:
                        chapter = f"Chapter {match.group(1)}"
                
                if 'article' in url_lower:
                    match = re.search(r'article[\/-]?(\d+)', url_lower, re.I)
                    if match:
                        article = f"Article {match.group(1)}"
                
                if 'section' in url_lower:
                    match = re.search(r'section[\/-]?(\d+)', url_lower, re.I)
                    if match:
                        section = f"Section {match.group(1)}"
                
                # Use the link text as title, or construct from structure
                reg_display_title = reg_title
                if not reg_display_title or len(reg_display_title) < 5:
                    parts = [p for p in [division, chapter, article, section] if p]
                    reg_display_title = ' - '.join(parts) if parts else f"Title {title_num}"
                
                regulations.append({
                    "article": f"Title {title_num}",
                    "part": title_name,
                    "section": section,
                    "title": reg_display_title,
                    "clauses": [{
                        "number": 1,
                        "title": reg_display_title,
                        "text": reg_text
                    }]
                })
    else:
        # No links found, try to extract from current page
        content = extract_regulation_text(soup)
        if content and len(content) > 100:
            # Clean up navigation content
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if any(skip in line.lower() for skip in ['state government sites', 'alaska case law', 'skip to navigation']):
                    continue
                if len(line) > 5:
                    cleaned_lines.append(line)
            
            cleaned_content = '\n'.join(cleaned_lines)
            if cleaned_content and len(cleaned_content) > 100:
                regulations.append({
                    "article": f"Title {title_num}",
                    "part": title_name,
                    "section": "",
                    "title": f"Title {title_num}. {title_name}",
                    "clauses": [{
                        "number": 1,
                        "title": f"Title {title_num}",
                        "text": cleaned_content
                    }]
                })
    
    logger.info(f"Found {len(regulations)} regulations for Title {title_num}")
    return regulations

def fetch_all_regulations(num_workers: int = 2, min_delay: float = 1.0) -> List[Dict[str, Any]]:
    """Fetch all California Code of Regulations from Westlaw."""
    logger.info("Starting to fetch California Code of Regulations from Westlaw...")
    logger.info(f"Using {num_workers} workers with {min_delay}s minimum delay")
    logger.warning("Note: Westlaw may require authentication. If you get 401/403 errors, you may need a subscription.")
    
    all_regulations = []
    session = requests.Session()
    session.headers.update(get_headers())
    
    # First, try to access the index page to establish session
    logger.info("Accessing Westlaw index page...")
    index_response = fetch_with_retry(WESTLAW_INDEX, session=session)
    if not index_response:
        logger.error("Could not access Westlaw index page. Check if authentication is required.")
        return all_regulations
    
    logger.info("Successfully accessed Westlaw. Proceeding to fetch regulations...")
    
    # Process titles in parallel
    processed_count = [0]
    
    def update_progress():
        with progress_lock:
            processed_count[0] += 1
            current = processed_count[0]
            if current % 5 == 0 or current == len(CA_REG_TITLES):
                logger.info(f"Processed {current}/{len(CA_REG_TITLES)} titles")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_title = {
            executor.submit(fetch_title_regulations, title_num, title_name, session): (title_num, title_name)
            for title_num, title_name in CA_REG_TITLES.items()
        }
        
        for future in as_completed(future_to_title):
            title_num, title_name = future_to_title[future]
            try:
                regulations = future.result()
                all_regulations.extend(regulations)
                update_progress()
            except Exception as e:
                logger.error(f"Error processing Title {title_num}: {e}")
                update_progress()
    
    logger.info(f"Total regulations fetched: {len(all_regulations)}")
    return all_regulations

def save_regulations(regulations: List[Dict[str, Any]], output_path: Path):
    """Save regulations to JSON file."""
    with save_lock:
        output_data = {
            "data": {
                "california_code_of_regulations": {
                    "regulations": regulations
                }
            }
        }
        
        temp_path = output_path.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        temp_path.replace(output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(regulations)} regulations to {output_path} ({file_size:.2f} MB)")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fetch California Code of Regulations from Westlaw")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker threads (default: 2, use lower to avoid rate limiting)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Minimum delay between requests in seconds (default: 1.0)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = base_dir / "Data" / "Knowledge" / "ca_regulations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("California Code of Regulations Fetcher (Westlaw)")
    logger.info("=" * 60)
    logger.warning("IMPORTANT: Westlaw is a commercial service that may require:")
    logger.warning("  - A subscription or login")
    logger.warning("  - Authentication credentials")
    logger.warning("  - Compliance with terms of service")
    logger.warning("=" * 60)
    
    start_time = time.time()
    
    regulations = fetch_all_regulations(num_workers=args.workers, min_delay=args.delay)
    
    if regulations:
        save_regulations(regulations, output_path)
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        logger.info("=" * 60)
        logger.info("Fetch complete!")
        logger.info(f"Total regulations: {len(regulations)}")
        logger.info(f"Total time: {hours}h {minutes}m {seconds}s")
        logger.info(f"Output file: {output_path}")
        logger.info("=" * 60)
    else:
        logger.error("No regulations were fetched. Check if authentication is required.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

