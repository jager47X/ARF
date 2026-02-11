import requests
from bs4 import BeautifulSoup
from fetch_ca_regulations_westlaw import get_headers, WESTLAW_BASE

# Test Title 4 page
url = f"{WESTLAW_BASE}/calregs/Title/4"
r = requests.get(url, headers=get_headers(), timeout=10)
soup = BeautifulSoup(r.content, 'html.parser')

print("=" * 60)
print("Examining Westlaw Title 4 page structure")
print("=" * 60)

# Find all links
links = soup.find_all('a', href=True)
print(f"\nFound {len(links)} total links")

# Look for regulation-related links
regulation_links = []
for link in links:
    href = link.get('href', '')
    text = link.get_text(strip=True)
    if any(keyword in href.lower() for keyword in ['division', 'chapter', 'article', 'section', 'title']):
        regulation_links.append((href, text))

print(f"\nFound {len(regulation_links)} regulation-related links:")
for href, text in regulation_links[:20]:
    print(f"  {href[:80]:80s} - {text[:50]}")

# Look for main content area
print("\n" + "=" * 60)
print("Looking for main content areas:")
print("=" * 60)

content_selectors = [
    'div.main-content',
    'div.content',
    'article',
    'main',
    'div[class*="content"]',
    'div[class*="regulation"]',
    'div[class*="text"]'
]

for selector in content_selectors:
    elements = soup.select(selector)
    if elements:
        print(f"\nFound {len(elements)} elements with selector '{selector}':")
        for i, elem in enumerate(elements[:3]):
            text = elem.get_text(strip=True)[:200]
            print(f"  Element {i+1}: {text}...")

# Check page title
title = soup.find('title')
if title:
    print(f"\nPage title: {title.get_text()}")

# Check for specific California regulation patterns
print("\n" + "=" * 60)
print("Looking for California regulation patterns:")
print("=" * 60)

body_text = soup.get_text()
if 'Article' in body_text:
    print("Found 'Article' in page text")
if 'Division' in body_text:
    print("Found 'Division' in page text")
if 'Chapter' in body_text:
    print("Found 'Chapter' in page text")
if 'Section' in body_text:
    print("Found 'Section' in page text")

