"""
Convert USCIS Policy Manual HTML to JSON format
"""
import json
import logging
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_uscis_html_to_json")

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def extract_text_from_element(element) -> str:
    """Extract text from an HTML element, preserving some structure."""
    if element is None:
        return ""
    
    # Get text and preserve line breaks for paragraphs
    text = element.get_text(separator='\n', strip=True)
    # Clean up multiple newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return clean_text(text)

def extract_cfr_references(text: str) -> List[str]:
    """Extract unique CFR references from text.
    Example: "See 8 CFR 216.4(a)(5)(iii)" -> "8 CFR 216.4"
    Example: "8 CFR 274a.12(c)(9)" -> "8 CFR 274a.12"
    """
    references = {}
    # Pattern to match full CFR references with title number: "X CFR Y.Z" or "See X CFR Y.Z"
    # Section can be like: 216.4, 274a.12, 103.2, etc.
    # This will match up to the first parenthesis (if any) for subsections
    pattern_with_title = r'(?:See\s+)?(\d+)\s+CFR\s+([\d\w]+\.\d+(?:\.\d+)?)'
    
    # Find all matches with title number
    matches = re.finditer(pattern_with_title, text, re.IGNORECASE)
    for match in matches:
        title_num = match.group(1)
        section = match.group(2).strip()
        # Remove trailing subsections like (c)(9)
        if '(' in section:
            section = section.split('(')[0].strip()
        ref = f"{title_num} CFR {section}"
        # Use section as key to avoid duplicates, prefer the one with title number
        references[section] = ref
    
    # Also check for references without title number (just "CFR"), but only if not already found
    pattern_no_title = r'(?:See\s+)?CFR\s+([\d\w]+\.\d+(?:\.\d+)?)'
    matches_no_title = re.finditer(pattern_no_title, text, re.IGNORECASE)
    for match in matches_no_title:
        section = match.group(1).strip()
        # Remove trailing subsections
        if '(' in section:
            section = section.split('(')[0].strip()
        # Only add if we don't already have this section with a title number
        if section not in references:
            ref = f"CFR {section}"
            references[section] = ref
    
    return sorted(list(references.values()))

def process_clauses(clauses: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Process clauses to:
    1. Concatenate "Paragraph X" clauses into previous clause
    2. Extract CFR references from footnotes section and all clauses
    3. Return processed clauses and list of unique CFR references
    """
    if not clauses:
        return [], []
    
    processed_clauses = []
    cfr_references = set()
    footnotes_started = False
    
    i = 0
    while i < len(clauses):
        clause = clauses[i].copy()
        title = clause.get("title", "")
        text = clause.get("text", "")
        
        # Check if this is the start of footnotes
        if "footnotes" in title.lower():
            footnotes_started = True
            # Extract references from this and all subsequent clauses
            j = i
            while j < len(clauses):
                ref_text = clauses[j].get("text", "")
                refs = extract_cfr_references(ref_text)
                cfr_references.update(refs)
                j += 1
            # Skip all footnote clauses (don't add them to processed_clauses)
            break
        
        # If footnotes have started, extract references and skip
        if footnotes_started:
            refs = extract_cfr_references(text)
            cfr_references.update(refs)
            i += 1
            continue
        
        # Check if title is "Paragraph X" (case insensitive)
        is_paragraph = re.match(r'^Paragraph\s+\d+$', title, re.IGNORECASE)
        
        if is_paragraph:
            # Concatenate to previous clause if it exists
            if processed_clauses:
                prev_clause = processed_clauses[-1]
                # Append text to previous clause with a space
                prev_clause["text"] = prev_clause["text"] + " " + text
            else:
                # If no previous clause, use the text as the title
                clause["title"] = text[:100] if len(text) > 100 else text
                processed_clauses.append(clause)
        else:
            # Regular clause - keep as is
            processed_clauses.append(clause)
        
        i += 1
    
    # Extract references from all processed clauses (not just footnotes)
    for clause in processed_clauses:
        refs = extract_cfr_references(clause.get("text", ""))
        cfr_references.update(refs)
    
    # Renumber clauses sequentially
    for idx, clause in enumerate(processed_clauses, 1):
        clause["number"] = idx
    
    # References are already formatted (e.g., "8 CFR 274a.12"), just return sorted
    return processed_clauses, sorted(list(cfr_references))

def extract_date_from_article(article) -> Optional[str]:
    """Extract date from article if available."""
    # Look for time elements
    time_elem = article.find('time')
    if time_elem and time_elem.get('datetime'):
        datetime_str = time_elem.get('datetime')
        # Try to parse and format
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return f"Last Updated:{dt.strftime('%m/%d/%Y')}"
        except:
            return f"Last Updated:{datetime_str}"
    
    # Look for current-date div
    current_date = article.find('div', class_='current-date')
    if current_date:
        date_value = current_date.find('span', class_='current-date__value')
        if date_value:
            date_text = clean_text(date_value.get_text())
            if date_text:
                return f"Last Updated:{date_text}"
    
    return None

def parse_chapter_content(body_element) -> List[Dict[str, Any]]:
    """Parse chapter content into clauses."""
    clauses = []
    clause_number = 1
    
    if body_element is None:
        return clauses
    
    # Get all direct children and process them in order
    current_heading = None
    
    for elem in body_element.children:
        if not hasattr(elem, 'name'):
            continue
            
        # Skip script and style tags
        if elem.name in ['script', 'style', 'form']:
            continue
        
        # Handle headings
        if elem.name in ['h2', 'h3', 'h4', 'h5']:
            current_heading = clean_text(elem.get_text())
            continue
        
        # Get text from element
        text = extract_text_from_element(elem)
        if not text or len(text) < 10:  # Skip very short text
            continue
        
        # Use current heading as title if available, otherwise use paragraph number
        if current_heading:
            title = current_heading[:100]
            current_heading = None  # Reset after using
        else:
            title = f"Paragraph {clause_number}"
        
        clauses.append({
            "number": clause_number,
            "title": title,
            "text": text
        })
        clause_number += 1
    
    # If no structured content found, get all text and split into paragraphs
    if not clauses:
        all_text = extract_text_from_element(body_element)
        if all_text and len(all_text) > 10:
            # Split into paragraphs (double newline or after periods)
            paragraphs = []
            current_para = []
            
            for line in all_text.split('\n'):
                line = line.strip()
                if line:
                    current_para.append(line)
                elif current_para:
                    para_text = ' '.join(current_para)
                    if len(para_text) > 10:
                        paragraphs.append(para_text)
                    current_para = []
            
            # Add last paragraph if exists
            if current_para:
                para_text = ' '.join(current_para)
                if len(para_text) > 10:
                    paragraphs.append(para_text)
            
            # If still no paragraphs, split by sentence
            if not paragraphs:
                sentences = re.split(r'[.!?]+\s+', all_text)
                current_para = []
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        current_para.append(sent)
                        if len(' '.join(current_para)) > 200:  # Group sentences into paragraphs
                            paragraphs.append(' '.join(current_para) + '.')
                            current_para = []
                if current_para:
                    paragraphs.append(' '.join(current_para) + '.')
            
            for i, para in enumerate(paragraphs, 1):
                if len(para) > 10:
                    clauses.append({
                        "number": i,
                        "title": f"Paragraph {i}",
                        "text": para
                    })
    
    return clauses

def parse_policy_manual_html(html_path: Path) -> Dict[str, Any]:
    """Parse the USCIS Policy Manual HTML file and convert to JSON format."""
    logger.info(f"Reading HTML file: {html_path}")
    
    # Read HTML file in chunks to handle large files
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    logger.info(f"Parsing HTML content ({len(html_content) / (1024*1024):.2f} MB)...")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    documents = []
    
    # Find all article elements with book-node-depth classes (these are the main sections)
    articles = soup.find_all('article', class_=re.compile(r'book-node-depth-\d+'))
    
    logger.info(f"Found {len(articles)} article sections")
    
    # Get the main date from the document
    main_date = None
    time_elem = soup.find('time', datetime=True)
    if time_elem:
        datetime_str = time_elem.get('datetime')
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            main_date = f"Last Updated:{dt.strftime('%m/%d/%Y')}"
        except:
            main_date = f"Last Updated:{datetime_str}"
    
    if not main_date:
        main_date = "Last Updated:12/22/2025"  # Default from sample
    
    processed_count = 0
    seen_titles = set()  # Track titles to avoid duplicates
    
    for article in articles:
        # Get the heading
        heading = article.find(['h1', 'h2', 'h3'], class_=re.compile(r'book-node-heading'))
        if not heading:
            continue
        
        title = clean_text(heading.get_text())
        if not title or title in ['Policy Manual', 'Search', 'Updates', 'Table of Contents']:
            continue
        
        # Skip if we've already processed this title (avoid duplicates)
        if title in seen_titles:
            continue
        seen_titles.add(title)
        
        # Find the body content - look for field--name-body first
        body_elem = article.find('div', class_=re.compile(r'field--name-body'))
        if not body_elem:
            # Try to find any content div with text-formatted
            body_elem = article.find('div', class_=re.compile(r'text-formatted'))
            if not body_elem:
                # Try clearfix
                body_elem = article.find('div', class_='clearfix')
                if not body_elem:
                    # Try to find any div with substantial text content
                    for div in article.find_all('div', recursive=True):
                        text = extract_text_from_element(div)
                        if text and len(text) > 100:  # Substantial content
                            body_elem = div
                            break
        
        if not body_elem:
            continue
        
        # Extract clauses from content
        clauses = parse_chapter_content(body_elem)
        
        if not clauses:
            # Try to get all text from the article as a fallback
            all_text = extract_text_from_element(article)
            if all_text and len(all_text) > 100:
                # Split into reasonable chunks
                paragraphs = [p.strip() for p in all_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
                if paragraphs:
                    clauses = [{
                        "number": i + 1,
                        "title": f"Paragraph {i + 1}",
                        "text": para
                    } for i, para in enumerate(paragraphs[:20])]  # Limit to 20 paragraphs
        
        if not clauses:
            continue
        
        # Process clauses: concatenate Paragraph X clauses and extract references
        processed_clauses, cfr_refs = process_clauses(clauses)
        
        if not processed_clauses:
            continue
        
        # Extract date for this article
        article_date = extract_date_from_article(article)
        if not article_date:
            article_date = main_date
        
        # Create document (without "article" field, with "references" instead of "agency")
        doc = {
            "title": title,
            "date": article_date,
            "references": cfr_refs if cfr_refs else [],
            "clauses": processed_clauses
        }
        
        documents.append(doc)
        processed_count += 1
        
        if processed_count % 50 == 0:
            logger.info(f"Processed {processed_count} documents...")
    
    logger.info(f"Total documents created: {len(documents)}")
    
    # Create the JSON structure
    result = {
        "data": {
            "uscis_policy": {
                "documents": documents
            }
        }
    }
    
    return result

def main():
    """Main function to convert HTML to JSON."""
    # Set up paths
    knowledge_dir = Path(__file__).parent.parent.parent / "Data" / "Knowledge"
    html_path = knowledge_dir / "Policy Manual _ USCIS.html"
    output_path = knowledge_dir / "uscis_policy.json"
    
    if not html_path.exists():
        logger.error(f"HTML file not found: {html_path}")
        return 1
    
    try:
        # Parse HTML and convert to JSON
        json_data = parse_policy_manual_html(html_path)
        
        # Write JSON file
        logger.info(f"Writing JSON to {output_path}...")
        temp_path = output_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Replace original file
        temp_path.replace(output_path)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Conversion complete! File size: {file_size:.2f} MB")
        logger.info(f"Total documents: {len(json_data['data']['uscis_policy']['documents'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error converting HTML to JSON: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

