#!/usr/bin/env python3
"""
Alternative script to fetch California codes using multiple sources.
This script tries alternative sources when leginfo.legislature.ca.gov is blocked.
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
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_ca_codes_alt")

# Alternative sources for California codes
ALTERNATIVE_SOURCES = {
    "california_public_law": "https://california.public.law/codes",
    "justia": "https://law.justia.com/codes/california",
    "findlaw": "https://codes.findlaw.com/ca",
}

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

def get_headers() -> Dict[str, str]:
    """Get HTTP headers for requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/",  # Make it look like we came from Google
    }

def fetch_with_retry(url: str, params: Optional[Dict] = None, retries: int = 3,
                     min_delay: float = 2.0, timeout: tuple = (30, 60)) -> Optional[requests.Response]:
    """Fetch URL with retry logic, longer delays, and session management."""
    session = requests.Session()
    session.headers.update(get_headers())

    for attempt in range(retries):
        try:
            # Longer delays for geo-blocked sites
            if attempt > 0:
                wait_time = min_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.1f}s before retry {attempt + 1}...")
                time.sleep(wait_time)

            response = session.get(url, params=params, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            if attempt < retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            else:
                logger.error(f"Timeout fetching {url} after {retries} attempts")
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                logger.warning(f"Connection error on attempt {attempt + 1} for {url}")
            else:
                logger.error(f"Connection error fetching {url} after {retries} attempts")
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            else:
                logger.error(f"Failed to fetch {url} after {retries} attempts: {e}")

    return None

def try_california_public_law(code_name: str, code_abbrev: str) -> List[Dict[str, Any]]:
    """Try to fetch from california.public.law."""
    logger.info(f"Trying california.public.law for {code_name}...")

    # California Public Law uses different URL patterns
    code_slug = code_name.lower().replace(" ", "-")
    base_url = f"https://california.public.law/codes/{code_slug}"

    sections = []
    # Try to fetch the index page
    response = fetch_with_retry(base_url, timeout=(30, 90))
    if response:
        soup = BeautifulSoup(response.text, "html.parser")
        # Look for section links
        section_links = soup.find_all("a", href=re.compile(r"/codes/.*/section"))
        logger.info(f"Found {len(section_links)} section links")
        # Process a sample first to test
        for link in section_links[:10]:  # Test with first 10
            href = link.get("href", "")
            if href:
                section_url = f"https://california.public.law{href}" if href.startswith("/") else href
                section_data = fetch_section_from_public_law(section_url, code_name)
                if section_data:
                    sections.append(section_data)
                    time.sleep(1)  # Rate limiting

    return sections

def fetch_section_from_public_law(url: str, code_name: str) -> Optional[Dict[str, Any]]:
    """Fetch a section from california.public.law."""
    try:
        response = fetch_with_retry(url, timeout=(30, 60))
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Find section content
        content = soup.find("div", class_=re.compile(r"section|content|text", re.I))
        if not content:
            content = soup.find("article") or soup.find("main")

        if content:
            # Extract section number from URL or content
            section_match = re.search(r"section[_-]?(\d+(?:\.\d+)*)", url, re.I)
            section_num = section_match.group(1) if section_match else "unknown"

            # Extract title
            title_elem = content.find(["h1", "h2", "h3"])
            title = title_elem.get_text(strip=True) if title_elem else f"Section {section_num}"

            # Extract text
            text = content.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

            # Remove navigation/UI elements
            text = re.sub(r'California Public Law.*?Section', 'Section', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > 50:
                return {
                    "code": code_name,
                    "section": f"Section {section_num}",
                    "title": title,
                    "clauses": [{
                        "number": 1,
                        "title": title,
                        "text": text
                    }]
                }
    except Exception as e:
        logger.warning(f"Error fetching from public.law: {e}")

    return None

def create_manual_instructions():
    """Create instructions for manual download when automated fetching fails."""
    instructions = """
# Manual Download Instructions for California Codes

Since automated fetching from leginfo.legislature.ca.gov is blocked from outside California,
here are alternative methods to obtain the data:

## Option 1: Use a VPN/Proxy with California IP
1. Connect to a VPN server located in California
2. Run the original script: `python fetch_ca_codes.py --workers 2 --delay 1.0`

## Option 2: Use Alternative Sources
1. **California Public Law**: https://california.public.law/codes
   - More accessible, but may require different parsing

2. **Justia**: https://law.justia.com/codes/california
   - Good alternative source

3. **FindLaw**: https://codes.findlaw.com/ca
   - Another alternative

## Option 3: Bulk Download (if available)
Check if California provides bulk XML/JSON downloads:
- Contact California Legislative Counsel Office
- Check for official data exports

## Option 4: Use Existing Cached Data
Check the `ca_codes_temp/` directory for cached HTML files that might be parseable.

## Option 5: Try During Off-Peak Hours
The blocking might be less strict during California off-peak hours (late night PST).
"""
    return instructions

def main():
    """Main function with alternative source attempts."""
    parser = argparse.ArgumentParser(description="Fetch California Codes using alternative sources")
    parser.add_argument("--source", choices=["public_law", "justia", "all"], default="all",
                       help="Alternative source to try")
    parser.add_argument("--test", action="store_true",
                       help="Test with just one code (Civil Code)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = base_dir / "Data" / "Knowledge" / "ca_code.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("California Codes Fetch - Alternative Sources")
    logger.info("="*60)
    logger.warning("NOTE: leginfo.legislature.ca.gov appears to be geo-blocked.")
    logger.info("Attempting alternative sources...")
    logger.info("")

    # Test with one code first
    test_codes = [("Civil Code", "CIV")] if args.test else list(CA_CODES.items())

    all_sections = []

    for code_name, code_abbrev in test_codes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {code_name} ({code_abbrev})")
        logger.info(f"{'='*60}")

        sections = []

        if args.source in ["public_law", "all"]:
            sections = try_california_public_law(code_name, code_abbrev)
            if sections:
                logger.info(f"✓ Successfully fetched {len(sections)} sections from california.public.law")
                all_sections.extend(sections)
                continue

        if not sections:
            logger.warning(f"Could not fetch {code_name} from any alternative source")
            logger.info("Consider using a VPN with California IP or manual download")

    # Save results
    if all_sections:
        output_data = {
            "data": {
                "california_codes": {
                    "codes": all_sections
                }
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Saved {len(all_sections)} sections to {output_path}")
    else:
        logger.warning("\n✗ No sections were fetched.")
        logger.info("\n" + create_manual_instructions())

        # Save instructions to file
        instructions_path = script_dir / "CA_CODES_MANUAL_INSTRUCTIONS.md"
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(create_manual_instructions())
        logger.info(f"Manual instructions saved to: {instructions_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


