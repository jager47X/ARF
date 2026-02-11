# Preprocess Scripts

Scripts for ingesting and updating embeddings for the RAG system.

## Directory Structure

The preprocess directory is organized by data source, with each source having its own subdirectory containing related fetch and ingest scripts:

```
preprocess/
├── us_code/              # US Code (USC) related scripts
│   ├── fetch_us_code_uslm.py
│   ├── ingest_us_code.py
│   └── convert_usc_xml_to_json.py
├── ca_codes/             # California Codes related scripts
│   ├── fetch_ca_codes.py
│   ├── ingest_ca_codes.py
│   └── fetch_ca_codes_*.py (alternative implementations)
├── ca_constitution/      # California Constitution scripts
│   ├── fetch_ca_constitution.py
│   └── fix_ca_constitution_titles.py
├── ca_regulations/       # California Regulations scripts
│   ├── fetch_ca_regulations_westlaw.py
│   └── fix_ca_regulations_json.py
├── cfr/                  # Code of Federal Regulations scripts
│   ├── fetch_cfr.py
│   ├── ingest_cfr_main.py
│   └── check_cfr_*.py (utility scripts)
├── federal_register/     # Federal Register scripts
│   └── ingest_federal_register.py
├── us_constitution/      # US Constitution scripts
│   ├── ingest_con_law.py
│   ├── ingest_alias_us_con_law.py
│   └── ingest_usc_main.py
├── supreme_court_cases/  # Supreme Court cases scripts
│   ├── ingest_supreme_court_cases.py
│   └── ingest_public_cases.py
└── utils/                # Utility and initialization scripts
    ├── init_empty_collection.py
    ├── init_user_queries.py
    └── verify_final_organization.py
```

## Running the Scripts

**IMPORTANT**: Python cannot import modules with hyphens in the name. Run the scripts directly as Python files.

### Run Scripts Directly (Recommended)

```powershell
# From workspace root (C:\Users\yutto\Documents\micelytech)
cd kyr-backend
python services/rag/preprocess/us_constitution/ingest_con_law.py --production --from-scratch --with-embeddings
```

Or from the workspace root:
```powershell
# From workspace root
python kyr-backend/services/rag/preprocess/us_constitution/ingest_con_law.py --production --from-scratch --with-embeddings
```

The scripts automatically set up the module path internally, so they can be run directly.

## Environment Selection

All scripts support environment selection via command-line flags:

- `--production` - Use `.env.production` file
- `--dev` - Use `.env.dev` file  
- `--local` - Use `.env.local` file
- (no flag) - Auto-detect based on Docker and file existence

## Available Scripts

1. **ingest_con_law.py** - Ingest US Constitution main document
   ```powershell
   cd kyr-backend
   # Ingest from scratch with embeddings (individual mode - faster per item)
   python services/rag/preprocess/us_constitution/ingest_con_law.py --production --from-scratch --with-embeddings
   
   # Ingest from scratch with embeddings (batch mode - slower but more efficient)
   python services/rag/preprocess/us_constitution/ingest_con_law.py --production --from-scratch --with-embeddings --batch-embeddings
   
   # Ingest without dropping existing data
   python services/rag/preprocess/us_constitution/ingest_con_law.py --production --with-embeddings
   ```
   
   Options:
   - `--with-embeddings`: Generate embeddings during ingestion (required for keyword aliases)
   - `--batch-embeddings`: Use batch embedding API (slower but more efficient for large datasets). Default: individual embeddings (faster per item)

2. **ingest_supreme_court_cases.py** - Ingest Supreme Court cases
   ```powershell
   cd kyr-backend
   # Ingest from scratch with embeddings
   python services/rag/preprocess/supreme_court_cases/ingest_supreme_court_cases.py --production --from-scratch --with-embeddings
   
   # Ingest without dropping existing data
   python services/rag/preprocess/supreme_court_cases/ingest_supreme_court_cases.py --production --with-embeddings
   ```

3. **init_user_queries.py** - Initialize User_queries collection from scratch
   ```powershell
   cd kyr-backend
   # Initialize empty collection with indexes
   python services/rag/preprocess/utils/init_user_queries.py --production
   
   # Drop existing and recreate
   python services/rag/preprocess/utils/init_user_queries.py --production --drop-existing
   ```

4. **ingest_alias_us_con_law.py** - Ingest alias mappings for US Constitution
   ```powershell
   cd kyr-backend
   python services/rag/preprocess/us_constitution/ingest_alias_us_con_law.py --production
   ```

5. **init_client_cases.py** - Initialize empty collections for client case database
   ```powershell
   cd kyr-backend
   python services/rag/preprocess/utils/init_client_cases.py --production
   ```
   Creates empty collections:
   - `User_queries` - For storing user query history
   - `client_cases` - For storing client case documents
   
   Options:
   - `--drop-existing` - Drop existing collections before creating (WARNING: deletes all data!)


## Note on Module Names

The directory is named `kyr-backend` but Python module names cannot contain hyphens. **You must run the scripts directly as Python files**, not as modules. The scripts automatically handle the module path setup internally.

**Correct way:**
```powershell
cd kyr-backend
python services/rag/preprocess/us_constitution/ingest_con_law.py --production --from-scratch --with-embeddings
```

**Incorrect way (will fail):**
```powershell
python -m kyr-backend.services.rag.preprocess.us_constitution.ingest_con_law --production  # ❌ ModuleNotFoundError
```

The scripts automatically set up the necessary module paths and aliases internally, so they can be run directly.

## Other Available Scripts

### US Code
- `us_code/fetch_us_code_uslm.py` - Fetch US Code from USLM format
- `us_code/ingest_us_code.py` - Ingest US Code into database
- `us_code/convert_usc_xml_to_json.py` - Convert USC XML to JSON format

### California Codes
- `ca_codes/fetch_ca_codes.py` - Fetch California State Codes
- `ca_codes/ingest_ca_codes.py` - Ingest California Codes into database
- `ca_codes/fetch_ca_codes_*.py` - Alternative fetch implementations

### California Constitution
- `ca_constitution/fetch_ca_constitution.py` - Fetch California Constitution
- `ca_constitution/fix_ca_constitution_titles.py` - Fix constitution titles

### California Regulations
- `ca_regulations/fetch_ca_regulations_westlaw.py` - Fetch CA regulations from Westlaw
- `ca_regulations/fix_ca_regulations_json.py` - Fix regulations JSON format

### Code of Federal Regulations (CFR)
- `cfr/fetch_cfr.py` - Fetch CFR data
- `cfr/ingest_cfr_main.py` - Ingest CFR into database
- `cfr/check_cfr_*.py` - CFR utility scripts

### Federal Register
- `federal_register/ingest_federal_register.py` - Ingest Federal Register documents

### Agency Guidance
- `agency_guidance/fetch_agency_guidance.py` - Fetch agency guidance documents from USCIS, DHS, ICE
- `agency_guidance/ingest_agency_guidance.py` - Ingest agency guidance into database
- `agency_guidance/migrate_agency_guidance.py` - Migrate agency guidance documents

