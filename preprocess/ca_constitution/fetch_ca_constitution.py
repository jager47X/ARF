#!/usr/bin/env python3
"""
Fetch California Constitution from official sources.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_ca_constitution")

# California Constitution sources - try multiple alternatives
CA_CONST_SOURCES = [
    "https://leginfo.legislature.ca.gov/faces/codesTOCSelected.xhtml?tocCode=CONS",
    "https://www.legislature.ca.gov/legislative_and_judicial_branch/constitution.html",
    "https://www.sos.ca.gov/administration/policy-manual/state-constitution",  # Secretary of State
    "https://ballotpedia.org/California_Constitution",  # Ballotpedia has full text
    "https://www.law.cornell.edu/constitution/california",  # Cornell Law School
]

def get_headers() -> Dict[str, str]:
    """Get headers for requests."""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

def fetch_with_retry(url: str, retries: int = 3, delay: float = 3.0) -> Optional[requests.Response]:
    """Fetch URL with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(delay)
            response = requests.get(url, timeout=(60, 120), headers=get_headers())
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                logger.warning(f"403 Forbidden for {url}")
                return None
            elif response.status_code == 404:
                logger.debug(f"404 Not Found for {url}")
                return None
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 10
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.debug(f"HTTP {response.status_code} for {url}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            if attempt < retries - 1:
                wait_time = (2 ** attempt) * 5
                time.sleep(wait_time)
        except Exception as e:
            logger.warning(f"Error fetching {url} (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

def parse_constitution_article(soup: BeautifulSoup, article_url: str) -> Optional[Dict[str, Any]]:
    """Parse a constitution article."""
    try:
        # Remove navigation and UI elements
        for element in soup.find_all(["nav", "header", "footer", "script", "style", "noscript", "form"]):
            element.decompose()

        # Find article content
        article_content = None
        for selector in [
            "div.sectionContent", "div#sectionContent", "div.content",
            "div.mainContent", "div.bodytext", "div[class*='content']",
            "div[class*='text']", "article", "main"
        ]:
            candidates = soup.select(selector)
            for candidate in candidates:
                text = candidate.get_text(strip=True)
                # Skip navigation elements
                if any(kw in text.lower() for kw in ["select code", "keyword", "search", "navigation", "menu", "skip to"]):
                    continue
                if len(text) > 200:  # Substantial content
                    article_content = candidate
                    break
            if article_content:
                break

        if not article_content:
            # Fallback: get all text from body
            body = soup.find("body")
            if body:
                article_content = body

        if not article_content:
            return None

        # Extract article number and title
        article_text = article_content.get_text(separator="\n", strip=True)

        # Try to find article number (e.g., "Article I" or "Article 1")
        article_match = re.search(r'Article\s+([IVX]+|\d+)', article_text[:500], re.IGNORECASE)
        article_num = article_match.group(1) if article_match else ""

        # Extract title (usually first substantial line)
        lines = article_text.split("\n")
        title = ""
        for line in lines[:15]:
            line = line.strip()
            if line and 20 < len(line) < 300:
                if not any(kw in line.lower() for kw in ["article", "section", "constitution", "california"]):
                    title = line
                    break

        # Split into sections/clauses
        clauses = []

        # Try to find sections (e.g., "Section 1", "Sec. 1")
        section_pattern = r'(?:Section|Sec\.?)\s+(\d+(?:\.\d+)?)\s*(.+?)(?=(?:Section|Sec\.?)\s+\d+|$)'
        sections = re.finditer(section_pattern, article_text, re.IGNORECASE | re.DOTALL)

        section_list = list(sections)
        if section_list:
            for idx, section_match in enumerate(section_list, 1):
                section_num = section_match.group(1)
                section_text = section_match.group(2).strip()

                # Extract section title if present
                section_title = ""
                first_line = section_text.split("\n")[0] if section_text else ""
                if first_line and len(first_line) < 200:
                    section_title = first_line

                # Clean up section text
                section_text = re.sub(r'\s+', ' ', section_text).strip()

                if section_text and len(section_text) > 30:
                    clauses.append({
                        "number": idx,
                        "title": section_title or f"Section {section_num}",
                        "text": section_text[:10000]  # Limit length
                    })

        # If no sections found, split into paragraphs
        if not clauses:
            paragraphs = [p.strip() for p in article_text.split("\n\n") if p.strip() and len(p.strip()) > 50]
            for idx, para in enumerate(paragraphs[:30], 1):  # Limit to first 30 paragraphs
                clauses.append({
                    "number": idx,
                    "title": f"Paragraph {idx}",
                    "text": para[:5000]  # Limit length
                })

        # If still no clauses, use entire text
        if not clauses:
            clauses.append({
                "number": 1,
                "title": title or f"Article {article_num}",
                "text": article_text[:20000]
            })

        return {
            "article": f"Article {article_num}" if article_num else "Article",
            "section": "",
            "title": title or f"Article {article_num}" if article_num else "Constitution Article",
            "clauses": clauses
        }
    except Exception as e:
        logger.error(f"Error parsing article from {article_url}: {e}")
        return None

def fetch_constitution_articles() -> List[Dict[str, Any]]:
    """Fetch all California Constitution articles."""
    logger.info("Fetching California Constitution...")
    articles = []

    # Try each source
    for source_url in CA_CONST_SOURCES:
        logger.info(f"Trying source: {source_url}")
        response = fetch_with_retry(source_url, retries=2, delay=3.0)
        if not response:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Find article links
        article_links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            # Look for article links
            if re.search(r'article\s+[ivx\d]+', text, re.IGNORECASE) or re.search(r'article', href, re.IGNORECASE):
                if href.startswith("/") or href.startswith("http"):
                    from urllib.parse import urljoin
                    full_url = urljoin(source_url, href)
                    article_links.append((full_url, text))

        logger.info(f"Found {len(article_links)} article links")

        # If we found links, fetch them
        if article_links:
            for article_url, article_text in article_links[:30]:  # Limit to first 30 articles
                response = fetch_with_retry(article_url, retries=2, delay=2.0)
                if response:
                    article_soup = BeautifulSoup(response.text, "html.parser")
                    parsed = parse_constitution_article(article_soup, article_url)
                    if parsed:
                        articles.append(parsed)

                time.sleep(1.0)  # Rate limiting

        # If we got articles, break
        if articles:
            break

    # If no articles found via links, try parsing the main page directly
    if not articles:
        logger.info("No article links found, trying to parse main page directly...")
        for source_url in CA_CONST_SOURCES:
            try:
                response = fetch_with_retry(source_url, retries=1, delay=3.0)  # Only 1 retry to save time
                if response:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Try to extract all articles from the page
                    # Some sites have all articles on one page
                    article_text = soup.get_text(separator="\n", strip=True)

                    # Split by article markers
                    article_pattern = r'Article\s+([IVX]+|\d+)\s*(.+?)(?=Article\s+[IVX]+|\d+|$)'
                    article_matches = re.finditer(article_pattern, article_text, re.IGNORECASE | re.DOTALL)

                    for match in article_matches:
                        article_num = match.group(1)
                        article_content = match.group(2).strip()

                        if len(article_content) > 100:  # Substantial content
                            # Parse this article
                            clauses = []
                            # Split into sections
                            section_pattern = r'(?:Section|Sec\.?)\s+(\d+(?:\.\d+)?)\s*(.+?)(?=(?:Section|Sec\.?)\s+\d+|$)'
                            sections = re.finditer(section_pattern, article_content, re.IGNORECASE | re.DOTALL)

                            section_list = list(sections)
                            if section_list:
                                for idx, sec_match in enumerate(section_list, 1):
                                    sec_num = sec_match.group(1)
                                    sec_text = sec_match.group(2).strip()
                                    if len(sec_text) > 30:
                                        clauses.append({
                                            "number": idx,
                                            "title": f"Section {sec_num}",
                                            "text": sec_text[:10000]
                                        })

                            # If no sections, use paragraphs
                            if not clauses:
                                paragraphs = [p.strip() for p in article_content.split("\n\n") if p.strip() and len(p.strip()) > 50]
                                for idx, para in enumerate(paragraphs[:30], 1):
                                    clauses.append({
                                        "number": idx,
                                        "title": f"Paragraph {idx}",
                                        "text": para[:5000]
                                    })

                            if clauses:
                                articles.append({
                                    "article": f"Article {article_num}",
                                    "section": "",
                                    "title": f"Article {article_num}",
                                    "clauses": clauses
                                })

                    # If we found articles, break
                    if articles:
                        logger.info(f"Successfully parsed {len(articles)} articles from {source_url}")
                        break
            except Exception as e:
                logger.debug(f"Error parsing {source_url}: {e}")
                continue

    logger.info(f"Fetched {len(articles)} constitution articles")
    return articles

def create_json_output(articles: List[Dict[str, Any]], output_path: Path):
    """Create JSON file in the required format."""
    output_data = {
        "data": {
            "california_constitution": {
                "articles": articles
            }
        }
    }

    logger.info(f"Writing {len(articles)} articles to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"JSON file created: {output_path} ({file_size:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Fetch California Constitution")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    output_path = Path(args.output) if args.output else base_dir / "Data" / "Knowledge" / "ca_constitution.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    articles = fetch_constitution_articles()

    if not articles:
        logger.warning("No articles fetched")
        return 1

    create_json_output(articles, output_path)
    logger.info("Processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

