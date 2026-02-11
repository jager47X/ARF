# fetch_agency_guidance.py
"""
Fetch Agency Guidance documents from official government sources.

Sources:
- USCIS Policy Memoranda
- USCIS Guidance Pages (OPT, H-1B, F-1, STEM OPT, etc.)
- DHS Guidance Documents
- ICE Guidance Documents

Fetches and structures data in the same format as agency_guidance.json
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
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_agency_guidance")

# Official government sources
USCIS_BASE = "https://www.uscis.gov"
USCIS_POLICY_MEMOS = "https://www.uscis.gov/laws-and-policy/policy-memoranda"
USCIS_GUIDANCE_BASE = "https://www.uscis.gov/working-in-the-united-states"
DHS_BASE = "https://www.dhs.gov"
ICE_BASE = "https://www.ice.gov"

# Key immigration topics to fetch
IMMIGRATION_TOPICS = {
    "OPT": {
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
        "title": "Optional Practical Training (OPT) for F-1 Students",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "STEM OPT": {
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
        "title": "STEM OPT Extension",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "H-1B": {
        "url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations-and-fashion-models",
        "title": "H-1B Specialty Occupations",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "F-1 Students": {
        "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors",
        "title": "Students and Exchange Visitors",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "DHS OPT Tip Sheet": {
        "url": "https://www.dhs.gov/publication/f1-OPT-tip-sheet",
        "title": "F-1 OPT Tip Sheet",
        "agency": "Department of Homeland Security",
        "section": "Guidance"
    },
    "L-1 Intracompany Transferees": {
        "url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/l-1-intracompany-transferee",
        "title": "L-1 Intracompany Transferees",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "O-1 Extraordinary Ability": {
        "url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/o-1-extraordinary-ability-or-achievement",
        "title": "O-1 Extraordinary Ability or Achievement",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "E-2 Treaty Investors": {
        "url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/e-2-treaty-investors",
        "title": "E-2 Treaty Investors",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "TN NAFTA Professionals": {
        "url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/tn-nafta-professionals",
        "title": "TN NAFTA Professionals",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "Employment Authorization": {
        "url": "https://www.uscis.gov/working-in-the-united-states/employment-authorization",
        "title": "Employment Authorization",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "Green Card Employment": {
        "url": "https://www.uscis.gov/green-card/green-card-processes-and-procedures/employment-based-immigration",
        "title": "Employment-Based Immigration",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "Family-Based Immigration": {
        "url": "https://www.uscis.gov/family/family-of-us-citizens",
        "title": "Family of U.S. Citizens",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "Naturalization": {
        "url": "https://www.uscis.gov/citizenship/learn-about-citizenship",
        "title": "Naturalization",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "Asylum": {
        "url": "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum",
        "title": "Asylum",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    },
    "DACA": {
        "url": "https://www.uscis.gov/humanitarian/consideration-of-deferred-action-for-childhood-arrivals-daca",
        "title": "Deferred Action for Childhood Arrivals (DACA)",
        "agency": "U.S. Citizenship and Immigration Services",
        "section": "Guidance"
    }
}

# Thread-safe locks
progress_lock = Lock()
save_lock = Lock()
rate_limit_lock = Lock()
last_request_time = [0.0]

def get_headers() -> Dict[str, str]:
    """Get HTTP headers for requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def rate_limit(delay: float = 1.0):
    """Rate limiting to avoid overwhelming servers."""
    with rate_limit_lock:
        elapsed = time.time() - last_request_time[0]
        if elapsed < delay:
            time.sleep(delay - elapsed)
        last_request_time[0] = time.time()

def fetch_with_retry(url: str, retries: int = 3, delay: float = 2.0) -> Optional[requests.Response]:
    """Fetch URL with retries and rate limiting."""
    rate_limit(delay)
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=get_headers(), timeout=30, allow_redirects=True)
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                logger.warning(f"404 Not Found: {url}")
                return None
            else:
                logger.warning(f"Status {response.status_code} for {url}, attempt {attempt + 1}/{retries}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching {url}, attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove common government website boilerplate patterns
    noise_patterns = [
        # Skip to main content and accessibility notices
        r'Skip to main content.*?Here\'s how you know',
        r'Here\'s how you know.*?Official websites use',
        r'An official website of the United States government',
        r'Official websites use \.gov',
        r'A \.gov website belongs to an official government organization',
        r'Secure \.gov websites use HTTPS',
        r'A lock.*?means you\'ve safely connected',
        r'Share sensitive information only on official, secure websites',
        
        # Navigation and menu items
        r'Menu.*?Enter Search Term',
        r'Topics.*?Topics',
        r'Border Security.*?Citizenship and Immigration.*?Cybersecurity',
        r'News.*?News.*?News',
        r'All DHS News.*?Apps.*?Blog',
        r'Comunicados de Prensa.*?Data Events',
        r'Fact Sheets.*?Featured News',
        r'Homeland Security LIVE.*?Media Contacts',
        r'Media Library.*?National Terrorism Advisory System',
        r'Press Releases.*?Publications Library',
        r'Social Media.*?Speeches.*?Subscribe',
        r'Testimony.*?In Focus.*?In Focus',
        r'2025 - The Year In Review',
        r'Worst of the Worst.*?CBP Home',
        r'Cybersecurity.*?Fentanyl',
        r'Independent Review of.*?Attempted Assassination',
        r'Making America Safe Again',
        r'How Do I\?.*?How Do I\?',
        r'Alphabetical Listing At DHS',
        r'For Businesses.*?For Travelers.*?For the Public',
        r'Get Involved.*?Get Involved',
        r'Blue Campaign.*?If You See Something',
        r'Know2Protect.*?Nationwide SAR Initiative',
        r'Ready\.gov.*?Secure Our World',
        r'US Coast Guard Auxiliary',
        r'About DHS.*?About DHS',
        r'Budget & Performance.*?Contact Us',
        r'Employee Resources.*?History',
        r'Homeland Security Careers',
        r'In Memoriam.*?Laws & Regulations',
        r'Leadership.*?Mission.*?Organization',
        r'Site Links.*?Breadcrumb',
        
        # Notices and promotional content
        r'notice.*?Holiday Deal',
        r'CBP Home now offering.*?Stipend',
        r'those who sign up before the end of the year',
        
        # Common footer elements
        r'Sign In.*?Sign In',
        r'Create Account.*?Menu',
        r'Español.*?Multilingual Resources',
        
        # Remove email patterns
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # Remove URLs
        r'https?://[^\s]+',
        
        # Remove common navigation phrases
        r'Home.*?Topics',
        r'Topics.*?Border Security',
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove lines that are just navigation items (short lines with common nav words)
    lines = text.split('\n')
    filtered_lines = []
    nav_keywords = ['menu', 'topics', 'news', 'about', 'contact', 'sign in', 'create account', 
                    'skip to', 'official website', 'breadcrumb', 'home', 'how do i']
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Skip very short lines that are likely navigation
        if len(line_stripped) < 20 and any(keyword in line_stripped.lower() for keyword in nav_keywords):
            continue
        # Skip lines that are mostly navigation keywords
        if sum(1 for keyword in nav_keywords if keyword in line_stripped.lower()) > 2:
            continue
        filtered_lines.append(line_stripped)
    
    text = ' '.join(filtered_lines)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_main_content(soup: BeautifulSoup, url: str) -> str:
    """Extract main content from a government webpage."""
    # Remove unwanted elements first (before finding main content)
    unwanted_selectors = [
        'nav', 'header', 'footer', 'aside',
        'script', 'style', 'noscript',
        '.skip-link', '.skip-to-content',
        '.site-header', '.site-footer',
        '.navigation', '.nav-menu',
        '.breadcrumb', '.breadcrumbs',
        '.social-media', '.social-links',
        '.newsletter', '.subscribe',
        '[role="banner"]', '[role="navigation"]', '[role="complementary"]',
        '.menu', '.menus', '.main-menu',
        '.site-navigation', '.primary-navigation',
        '.secondary-navigation', '.footer-navigation',
        '.utility-menu', '.top-bar',
        '.announcement', '.notice', '.alert',
        '.promo', '.promotion',
    ]
    
    for selector in unwanted_selectors:
        for tag in soup.select(selector):
            tag.decompose()
    
    # Remove elements with common noise text
    noise_text_patterns = [
        'Skip to main content',
        'Here\'s how you know',
        'Official website',
        'Sign In',
        'Create Account',
        'Menu',
        'Topics',
        'How Do I?',
        'Get Involved',
        'About DHS',
        'Breadcrumb',
        'Holiday Deal',
        'CBP Home',
    ]
    
    for pattern in noise_text_patterns:
        for tag in soup.find_all(string=re.compile(pattern, re.IGNORECASE)):
            parent = tag.parent
            if parent:
                # Check if this is likely navigation/footer
                if parent.name in ['nav', 'header', 'footer', 'aside', 'div']:
                    # Check parent classes/ids for navigation indicators
                    classes = parent.get('class', [])
                    ids = parent.get('id', '')
                    nav_indicators = ['nav', 'menu', 'header', 'footer', 'breadcrumb', 'skip']
                    if any(indicator in str(classes).lower() or indicator in str(ids).lower() for indicator in nav_indicators):
                        parent.decompose()
    
    # Try to find main content area
    main_content = None
    
    # Common selectors for government sites (prioritized)
    selectors = [
        'main article',
        'main .field-body',
        'main .field-content',
        'main .node-content',
        'main',
        'article',
        '.main-content article',
        '.main-content .field-body',
        '.main-content',
        '.content article',
        '.content .field-body',
        '.content',
        '#main-content article',
        '#main-content .field-body',
        '#main-content',
        '#content article',
        '#content',
        '.field-body',
        '.field-content',
        '.node-content',
        'div[role="main"] article',
        'div[role="main"]',
    ]
    
    for selector in selectors:
        main_content = soup.select_one(selector)
        if main_content:
            # Verify it has substantial content
            text_length = len(main_content.get_text(strip=True))
            if text_length > 500:  # At least 500 characters
                break
            else:
                main_content = None
    
    # If no main content found, try to remove common navigation/footer elements from body
    if not main_content:
        body = soup.find('body')
        if body:
            # Remove remaining navigation-like elements
            for tag in body.find_all(['nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Remove elements with navigation-like classes/ids
            for tag in body.find_all(class_=re.compile(r'nav|menu|header|footer|breadcrumb|skip|utility', re.I)):
                tag.decompose()
            
            for tag in body.find_all(id=re.compile(r'nav|menu|header|footer|breadcrumb|skip|utility', re.I)):
                tag.decompose()
            
            main_content = body
    
    if main_content:
        # Get text content
        text = main_content.get_text(separator=' ', strip=True)
        return clean_text(text)
    
    # Fallback: get all text (but still cleaned)
    return clean_text(soup.get_text())

def split_into_clauses(text: str, max_clause_length: int = 5000) -> List[Dict[str, Any]]:
    """Split text into clauses for better RAG processing."""
    clauses = []
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_clause = []
    current_length = 0
    clause_num = 1
    
    for para in paragraphs:
        para_length = len(para)
        
        # If paragraph is very long, split it further
        if para_length > max_clause_length:
            # Save current clause if any
            if current_clause:
                clauses.append({
                    "number": clause_num,
                    "title": f"Paragraph {clause_num}",
                    "text": " ".join(current_clause)
                })
                clause_num += 1
                current_clause = []
                current_length = 0
            
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_length + len(sentence) > max_clause_length and current_clause:
                    clauses.append({
                        "number": clause_num,
                        "title": f"Paragraph {clause_num}",
                        "text": " ".join(current_clause)
                    })
                    clause_num += 1
                    current_clause = []
                    current_length = 0
                
                current_clause.append(sentence)
                current_length += len(sentence) + 1
        else:
            # Check if adding this paragraph would exceed limit
            if current_length + para_length > max_clause_length and current_clause:
                clauses.append({
                    "number": clause_num,
                    "title": f"Paragraph {clause_num}",
                    "text": " ".join(current_clause)
                })
                clause_num += 1
                current_clause = []
                current_length = 0
            
            current_clause.append(para)
            current_length += para_length + 2  # +2 for paragraph separator
    
    # Add remaining clause
    if current_clause:
        clauses.append({
            "number": clause_num,
            "title": f"Paragraph {clause_num}",
            "text": " ".join(current_clause)
        })
    
    return clauses if clauses else [{"number": 1, "title": "Paragraph 1", "text": text}]

def fetch_guidance_page(topic_key: str, topic_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch a single guidance page."""
    url = topic_info["url"]
    logger.info(f"Fetching {topic_key}: {url}")
    
    response = fetch_with_retry(url)
    if not response:
        logger.warning(f"Failed to fetch {topic_key}")
        return None
    
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content
        content = extract_main_content(soup, url)
        
        if not content or len(content) < 100:
            logger.warning(f"Insufficient content extracted from {topic_key}")
            return None
        
        # Try to extract date from page
        date_str = "Last Updated: Unknown"
        date_patterns = [
            r'Last Updated:\s*([^\n]+)',
            r'Updated:\s*([^\n]+)',
            r'Date:\s*([^\n]+)',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]
        
        page_text = soup.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                date_str = f"Last Updated: {match.group(1).strip()}"
                break
        
        # Split into clauses
        clauses = split_into_clauses(content)
        
        document = {
            "article": topic_info["agency"],
            "section": topic_info["section"],
            "title": topic_info["title"],
            "date": date_str,
            "document_type": "Guidance",
            "agency": topic_info["agency"],
            "clauses": clauses
        }
        
        logger.info(f"Successfully fetched {topic_key}: {len(clauses)} clauses, {len(content)} chars")
        return document
        
    except Exception as e:
        logger.error(f"Error processing {topic_key}: {e}", exc_info=True)
        return None

def fetch_uscis_policy_memos() -> List[Dict[str, Any]]:
    """Fetch USCIS policy memoranda."""
    logger.info("Fetching USCIS Policy Memoranda...")
    documents = []
    
    # Note: USCIS policy memos page may require more sophisticated scraping
    # For now, we'll focus on the specific guidance pages
    # This can be enhanced later to parse the policy memos listing page
    
    return documents

def fetch_all_guidance() -> List[Dict[str, Any]]:
    """Fetch all guidance documents."""
    all_documents = []
    
    # Fetch specific guidance pages
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fetch_guidance_page, key, info): key
            for key, info in IMMIGRATION_TOPICS.items()
        }
        
        for future in as_completed(futures):
            topic_key = futures[future]
            try:
                doc = future.result()
                if doc:
                    all_documents.append(doc)
            except Exception as e:
                logger.error(f"Error fetching {topic_key}: {e}")
    
    logger.info(f"Fetched {len(all_documents)} guidance documents")
    return all_documents

def merge_with_existing(new_docs: List[Dict[str, Any]], existing_path: str, force_update: bool = False) -> List[Dict[str, Any]]:
    """Merge new documents with existing ones, avoiding duplicates or updating if force_update is True."""
    # Load existing documents
    existing_docs = []
    if os.path.exists(existing_path):
        try:
            with open(existing_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_docs = data.get("data", {}).get("agency_guidance", {}).get("documents", [])
                logger.info(f"Loaded {len(existing_docs)} existing documents")
        except Exception as e:
            logger.warning(f"Could not load existing file: {e}")
    
    # Create dictionary of existing documents by title (lowercase) for easy lookup
    existing_by_title = {doc.get("title", "").lower().strip(): doc for doc in existing_docs}
    
    # Build merged list
    merged_docs = []
    added_count = 0
    updated_count = 0
    
    # First, add/update documents from new_docs
    for new_doc in new_docs:
        title = new_doc.get("title", "").lower().strip()
        if not title:
            continue
            
        if title in existing_by_title:
            if force_update:
                # Replace existing document with new (cleaner) version
                merged_docs.append(new_doc)
                updated_count += 1
                logger.info(f"Updated: {new_doc.get('title', 'Unknown')}")
            else:
                # Keep existing, skip new
                merged_docs.append(existing_by_title[title])
                logger.info(f"Skipping duplicate: {new_doc.get('title', 'Unknown')}")
        else:
            # New document
            merged_docs.append(new_doc)
            existing_by_title[title] = new_doc
            added_count += 1
    
    # Add remaining existing documents that weren't in new_docs
    new_titles = {doc.get("title", "").lower().strip() for doc in new_docs}
    for existing_doc in existing_docs:
        title = existing_doc.get("title", "").lower().strip()
        if title not in new_titles:
            merged_docs.append(existing_doc)
    
    logger.info(f"Added {added_count} new documents, updated {updated_count} documents, total: {len(merged_docs)}")
    return merged_docs

def save_documents(documents: List[Dict[str, Any]], output_path: str):
    """Save documents to JSON file."""
    output_data = {
        "data": {
            "agency_guidance": {
                "documents": documents
            }
        }
    }
    
    # Create backup of existing file
    if os.path.exists(output_path):
        backup_path = output_path + ".backup"
        import shutil
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(documents)} documents to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fetch Agency Guidance documents")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: Data/Knowledge/agency_guidance.json)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing agency_guidance.json file"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=list(IMMIGRATION_TOPICS.keys()) + ["all"],
        default=["all"],
        help="Specific topics to fetch (default: all)"
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Update existing documents with same title (useful for cleaning noise from existing docs)"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        BASE_DIR = Path(__file__).resolve().parents[2]
        output_path = str(BASE_DIR / "Data/Knowledge/agency_guidance.json")
    
    # Determine which topics to fetch
    if "all" in args.topics:
        topics_to_fetch = IMMIGRATION_TOPICS
    else:
        topics_to_fetch = {k: v for k, v in IMMIGRATION_TOPICS.items() if k in args.topics}
    
    logger.info(f"Fetching {len(topics_to_fetch)} guidance topics...")
    
    # Fetch documents
    all_documents = []
    for topic_key, topic_info in topics_to_fetch.items():
        doc = fetch_guidance_page(topic_key, topic_info)
        if doc:
            all_documents.append(doc)
        time.sleep(2)  # Be respectful with rate limiting
    
    if not all_documents:
        logger.error("No documents fetched!")
        return
    
    # Merge with existing if requested
    if args.merge:
        all_documents = merge_with_existing(all_documents, output_path, force_update=args.force_update)
    
    # Save documents
    save_documents(all_documents, output_path)
    logger.info("Done!")

if __name__ == "__main__":
    main()

