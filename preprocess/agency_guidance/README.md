# Agency Guidance Fetch Script

This script fetches agency guidance documents from official government sources (USCIS, DHS, ICE) and enriches the `agency_guidance.json` knowledge base.

## Overview

The script scrapes official guidance pages from government websites and structures them in the same format as the existing `agency_guidance.json` file, making them ready for ingestion into the RAG system.

## Usage

### Basic Usage

Fetch all immigration topics and merge with existing file:

```bash
cd kyr-backend
python services/rag/preprocess/agency_guidance/fetch_agency_guidance.py --merge
```

### Fetch Specific Topics

Fetch only specific topics:

```bash
python services/rag/preprocess/agency_guidance/fetch_agency_guidance.py --topics OPT "STEM OPT" H-1B --merge
```

### Custom Output Path

Save to a different file:

```bash
python services/rag/preprocess/agency_guidance/fetch_agency_guidance.py --output custom_path.json --merge
```

### Fetch Without Merging

Create a new file with only fetched documents:

```bash
python services/rag/preprocess/agency_guidance/fetch_agency_guidance.py --output new_guidance.json
```

## Available Topics

The script can fetch guidance on the following topics:

- **OPT** - Optional Practical Training for F-1 Students
- **STEM OPT** - STEM OPT Extension
- **H-1B** - H-1B Specialty Occupations
- **F-1 Students** - Students and Exchange Visitors
- **L-1 Intracompany Transferees**
- **O-1 Extraordinary Ability**
- **E-2 Treaty Investors**
- **TN NAFTA Professionals**
- **Employment Authorization**
- **Green Card Employment** - Employment-Based Immigration
- **Family-Based Immigration**
- **Naturalization**
- **Asylum**
- **DACA** - Deferred Action for Childhood Arrivals
- **DHS OPT Tip Sheet**

## How It Works

1. **Fetches** guidance pages from official government websites (USCIS, DHS, ICE)
2. **Extracts** main content from HTML pages, removing navigation and boilerplate
3. **Splits** content into clauses for better RAG processing
4. **Structures** data in the same format as existing `agency_guidance.json`
5. **Merges** with existing documents (if `--merge` flag is used), avoiding duplicates
6. **Saves** to JSON file (creates backup of existing file)

## Data Structure

Each document follows this structure:

```json
{
  "article": "U.S. Citizenship and Immigration Services",
  "section": "Guidance",
  "title": "Optional Practical Training (OPT) for F-1 Students",
  "date": "Last Updated: 12/19/2025",
  "document_type": "Guidance",
  "agency": "U.S. Citizenship and Immigration Services",
  "clauses": [
    {
      "number": 1,
      "title": "Paragraph 1",
      "text": "Content text here..."
    }
  ]
}
```

## Rate Limiting

The script includes rate limiting to be respectful of government servers:
- Minimum 2 second delay between requests
- Retry logic with exponential backoff
- Thread-safe request handling

## Next Steps

After fetching new guidance documents:

1. **Review** the generated JSON file to ensure quality
2. **Ingest** into MongoDB using the ingestion script:
   ```bash
   python services/rag/preprocess/agency_guidance/ingest_agency_guidance.py --local --with-embeddings
   ```
3. **Test** queries in the RAG system to verify the new content is searchable

## Adding New Topics

To add new topics, edit `fetch_agency_guidance.py` and add entries to the `IMMIGRATION_TOPICS` dictionary:

```python
"Topic Name": {
    "url": "https://www.uscis.gov/...",
    "title": "Document Title",
    "agency": "U.S. Citizenship and Immigration Services",
    "section": "Guidance"
}
```

## Troubleshooting

### No Content Extracted

If a page shows "Insufficient content extracted", the page structure may have changed. Check the URL manually and update the content extraction logic if needed.

### 404 Errors

Some URLs may change or be removed. Check the official website for updated URLs.

### Rate Limiting

If you encounter rate limiting errors, increase the delay in the `rate_limit()` function.

## Notes

- The script creates a backup of the existing file before overwriting (when using `--merge`)
- Duplicate documents are automatically skipped based on title matching
- Content is cleaned to remove navigation elements and boilerplate text
- Long documents are split into multiple clauses for better RAG processing











































