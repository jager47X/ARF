#!/usr/bin/env python3
"""
Quick test script to fetch a sample of California codes to verify the process works.
This fetches just a few sections from the Civil Code to test connectivity.
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Base URL for California Legislative Information
CA_LEGINFO_BASE = "https://leginfo.legislature.ca.gov"

def get_headers():
    """Get HTTP headers for requests."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

def fetch_section(code_abbrev: str, section_num: str):
    """Fetch a single section."""
    url = f"{CA_LEGINFO_BASE}/faces/codes_displaySection.xhtml"
    params = {
        "sectionNum": section_num,
        "lawCode": code_abbrev
    }

    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove navigation elements
        for element in soup.find_all(["nav", "form", "header", "footer", "script", "style"]):
            element.decompose()

        # Find content
        content = soup.find("div", class_=re.compile(r"section|content", re.I))
        if not content:
            content = soup.find("body")

        if content:
            text = content.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

            # Remove navigation patterns
            text = re.sub(r'Code:.*?Keyword\(s\):', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'\s+', ' ', text).strip()

            return {
                "code": "Civil Code",
                "section": f"Section {section_num}",
                "title": f"Civil Code Section {section_num}",
                "clauses": [{
                    "number": 1,
                    "title": f"Section {section_num}",
                    "text": text[:2000]  # Limit for test
                }]
            }
    except Exception as e:
        print(f"Error fetching section {section_num}: {e}")
        return None

# Test with a few Civil Code sections
print("Testing California codes fetch...")
sections = []

# Try fetching a few sections from Civil Code
test_sections = ["1", "2", "3", "10", "20"]
for section_num in test_sections:
    print(f"Fetching Civil Code Section {section_num}...")
    section = fetch_section("CIV", section_num)
    if section:
        sections.append(section)
        print(f"  ✓ Successfully fetched Section {section_num}")
    time.sleep(1)  # Rate limiting

# Save test results
script_dir = Path(__file__).resolve().parent
output_path = script_dir.parent.parent / "Data" / "Knowledge" / "ca_code.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    "data": {
        "california_codes": {
            "codes": sections
        }
    }
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✓ Test complete! Fetched {len(sections)} sections")
print(f"✓ Saved to: {output_path}")
print("\nIf this worked, the full fetch_ca_codes.py script should work too.")
print("Run it with: python fetch_ca_codes.py --workers 2 --delay 1.0")


