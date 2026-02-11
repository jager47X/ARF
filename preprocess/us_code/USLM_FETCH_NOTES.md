# US Code USLM Fetch Notes

## Current Status

The `fetch_us_code_uslm.py` script has been improved to properly parse USLM XML format with full hierarchical structure support (chapters, subchapters, parts, etc.).

## Download Issue

The uscode.house.gov website requires JavaScript and session handling to access XML files. Direct HTTP requests return HTML error pages instead of XML files.

## Solutions

### Option 1: Manual Download (Recommended)
1. Visit https://uscode.house.gov/download/download.shtml in a browser
2. Download the XML files for each title (or the complete zip file)
3. Place them in the `usc_uslm_xml_temp` directory
4. Run: `python fetch_us_code_uslm.py --skip-download`

### Option 2: Use Existing XML Files
If you have XML files from a previous download, place them in `usc_uslm_xml_temp` and run:
```bash
python fetch_us_code_uslm.py --skip-download
```

### Option 3: Alternative Source
Check if govinfo.gov provides US Code in USLM format:
- https://www.govinfo.gov/bulkdata/USCODE

## Parser Improvements

The parser now:
- ✅ Properly handles USLM namespaces
- ✅ Extracts hierarchical structure (chapters, subchapters, parts)
- ✅ Parses sections, subsections, paragraphs, and clauses
- ✅ Handles nested text content correctly
- ✅ Outputs JSON in the expected format

## Usage

Once you have XML files:
```bash
cd kyr-backend/services/rag/preprocess
python fetch_us_code_uslm.py --skip-download --parse-workers 4
```

This will parse all XML files and create `us_code.json` with complete US Code data.

