import requests
from bs4 import BeautifulSoup
from fetch_ca_regulations_westlaw import WESTLAW_INDEX, get_headers

# Test index page
r = requests.get(WESTLAW_INDEX, headers=get_headers(), timeout=10)
soup = BeautifulSoup(r.content, 'html.parser')

print("=" * 60)
print("Examining Westlaw Index page structure")
print("=" * 60)

# Find all links
links = soup.find_all('a', href=True)
print(f"\nFound {len(links)} total links")

# Look for Title links
title_links = []
for link in links:
    href = link.get('href', '')
    text = link.get_text(strip=True)
    if 'title' in text.lower() or 'title' in href.lower():
        title_links.append((href, text))

print(f"\nFound {len(title_links)} title-related links:")
for href, text in title_links[:30]:
    try:
        print(f"  {href[:100]:100s} - {text[:60]}")
    except UnicodeEncodeError:
        print(f"  {href[:100]:100s} - {repr(text[:60])}")

# Look for list items or divs containing title information
print("\n" + "=" * 60)
print("Looking for Title list structure:")
print("=" * 60)

# Check for list items
list_items = soup.find_all('li')
print(f"Found {len(list_items)} list items")
for li in list_items[:10]:
    text = li.get_text(strip=True)
    if 'Title' in text:
        print(f"  {text[:80]}")

# Check for divs with title information
divs = soup.find_all('div')
title_divs = []
for div in divs:
    text = div.get_text(strip=True)
    if 'Title' in text and len(text) < 200:
        title_divs.append(text)

print(f"\nFound {len(title_divs)} divs with Title info:")
for div_text in title_divs[:10]:
    print(f"  {div_text[:80]}")

# Save HTML for inspection
with open('westlaw_index.html', 'w', encoding='utf-8') as f:
    f.write(soup.prettify())
print("\nSaved HTML to westlaw_index.html for inspection")

