import argparse
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pymongo import ASCENDING, MongoClient
from pymongo.collation import Collation
from pymongo.errors import DuplicateKeyError

# Setup path for module execution
# The directory is 'kyr-backend' but imports use 'backend'
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create 'backend' -> 'kyr-backend' mapping for imports
# Python can't import modules with hyphens, so we create an alias
import importlib.util
import types

# Set up backend module structure to point to kyr-backend directory
if 'backend' not in sys.modules:
    # Create backend module
    backend_mod = types.ModuleType('backend')
    sys.modules['backend'] = backend_mod

    # Load services
    services_init = backend_dir / 'services' / '__init__.py'
    services_mod = None
    if services_init.exists():
        spec = importlib.util.spec_from_file_location('backend.services', services_init)
        if spec and spec.loader:
            services_mod = importlib.util.module_from_spec(spec)
            sys.modules['backend.services'] = services_mod
            spec.loader.exec_module(services_mod)
            setattr(backend_mod, 'services', services_mod)

            # Also create 'services' module alias (for files using 'from services.rag.config')
            if 'services' not in sys.modules:
                sys.modules['services'] = services_mod

            # Load rag
            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
            rag_mod = None
            if rag_init.exists():
                spec = importlib.util.spec_from_file_location('backend.services.rag', rag_init)
                if spec and spec.loader:
                    rag_mod = importlib.util.module_from_spec(spec)
                    sys.modules['backend.services.rag'] = rag_mod
                    spec.loader.exec_module(rag_mod)
                    setattr(services_mod, 'rag', rag_mod)

                    # Also create 'services.rag' alias
                    sys.modules['services.rag'] = rag_mod

                    # Load config
                    config_file = backend_dir / 'services' / 'rag' / 'config.py'
                    if config_file.exists():
                        spec = importlib.util.spec_from_file_location('backend.services.rag.config', config_file)
                        if spec and spec.loader:
                            config_mod = importlib.util.module_from_spec(spec)
                            sys.modules['backend.services.rag.config'] = config_mod
                            spec.loader.exec_module(config_mod)
                            setattr(rag_mod, 'config', config_mod)

                            # Also create 'services.rag.config' alias
                            sys.modules['services.rag.config'] = config_mod

                            # Load rag_dependencies
                            rag_deps_init = backend_dir / 'services' / 'rag' / 'rag_dependencies' / '__init__.py'
                            if rag_deps_init.exists():
                                spec = importlib.util.spec_from_file_location('backend.services.rag.rag_dependencies', rag_deps_init)
                                if spec and spec.loader:
                                    rag_deps_mod = importlib.util.module_from_spec(spec)
                                    sys.modules['backend.services.rag.rag_dependencies'] = rag_deps_mod
                                    spec.loader.exec_module(rag_deps_mod)
                                    setattr(rag_mod, 'rag_dependencies', rag_deps_mod)

                                    # Also create 'services.rag.rag_dependencies' alias
                                    sys.modules['services.rag.rag_dependencies'] = rag_deps_mod

# Parse arguments before importing config
def parse_args():
    parser = argparse.ArgumentParser(description="Ingest alias map")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (.env.production)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev environment (.env.dev)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local environment (.env.local)"
    )
    return parser.parse_args()

# Load environment based on args
args = parse_args() if __name__ == "__main__" else None
env_override = None
if args:
    if args.production:
        env_override = "production"
    elif args.dev:
        env_override = "dev"
    elif args.local:
        env_override = "local"

# Import config module and load environment
import backend.services.rag.config as config_module

# Now load the correct environment (this will override the default with override=True)
if env_override:
    config_module.load_environment(env_override)
    # Re-read MONGO_URI after loading the correct environment
    import os
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(f"MONGO_URI not found in {config_module._env_file_used}")
else:
    MONGO_URI = config_module.MONGO_URI

# Import other config values
ALIAS_COLL_NAME = config_module.ALIAS_COLL_NAME
COLLECTION = config_module.COLLECTION
from backend.services.rag.rag_dependencies.ai_service import LLM

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Mongo (lazy initialization - only when script is run directly)
# ----------------------------
USC_CONF = COLLECTION["US_CONSTITUTION_SET"]
DB_NAME: str = USC_CONF["db_name"]
MAIN_COLL_NAME: str = USC_CONF["main_collection_name"]

# Lazy initialization - only connect when script is run directly, not on import
_client = None
_db = None
_main_collection = None
_chatgpt = None

def get_main_collection():
    """Get main collection, initializing MongoDB connection if needed"""
    global _client, _db, _main_collection
    if _main_collection is None:
        _client = MongoClient(MONGO_URI)
        _db = _client[DB_NAME]
        _main_collection = _db[MAIN_COLL_NAME]
    return _main_collection

def get_chatgpt():
    """Get LLM instance, initializing if needed"""
    global _chatgpt
    if _chatgpt is None:
        _chatgpt = LLM(config=USC_CONF)
    return _chatgpt

# Note: We're now storing keywords directly in the main collection, not a separate alias collection

# ----------------------------
# Alias Map (embedded here)
# ----------------------------
alias_map = {
    # -------- Preamble / Constitution --------


    # -------- Article I --------
    "The Congress": [
        "legislative powers",
        "congress powers",
        "bicameral legislature"
    ],
    "The House of Representatives": [
        "house of representatives",
        "lower house",
        "apportionment",
        "census",
        "vacancies",
        "impeachment power"
    ],
    "The Senate": [
        "upper house",
        "senate classes",
        "president of the senate",
        "trial of impeachments",
        "impeachment trial"
    ],
    "Elections": [
        "elections clause",
        "time place manner",
        "congress meeting day"
    ],
    "Powers and Duties of Congress": [
        "quorum",
        "journal publication",
        "expel members",
        "adjournment rules",
        "house rules"
    ],
    "Rights and Disabilities of Members": [
        "speech or debate clause",
        "legislative immunity",
        "incompatibility clause",
        "arrest immunity"
    ],
    "Legislative Process": [
        "origination clause",
        "presentment clause",
        "veto",
        "veto override",
        "orders resolutions votes"
    ],
    "Powers of Congress": [
        "enumerated powers",
        "taxing and spending",
        "borrow money",
        "commerce clause",
        "naturalization",
        "bankruptcy",
        "coin money",
        "counterfeiting",
        "post office",
        "patents and copyrights",
        "inferior tribunals",
        "piracies on the high seas",
        "declare war",
        "raise army",
        "maintain navy",
        "militia",
        "district of columbia",
        "necessary and proper clause",
        "elastic clause"
    ],
    "Powers Denied to Congress": [
        "slave trade",
        "habeas corpus",
        "bill of attainder",
        "ex post facto",
        "capitation",
        "export taxes",
        "port preference",
        "appropriations statement",
        "titles of nobility"
    ],
    "Powers Denied to the States": [
        "contracts clause",
        "no treaties by states",
        "coinage prohibition",
        "import export duties",
        "tonnage duties",
        "troops in peacetime",
        "compact clause",
        "state war powers"
    ],

    # -------- Article II --------
    "Executive Power": [
        "presidential powers",
        "electoral college",
        "president qualifications",
        "succession",
        "compensation",
        "presidential oath"
    ],
    "Commander in Chief and Appointments": [
        "commander in chief",
        "pardon power",
        "treaty power",
        "appointments clause",
        "recess appointments"
    ],
    "State of the Union": [
        "state of the union",
        "convene congress",
        "receive ambassadors",
        "take care clause"
    ],
    "Impeachment": [
        "removal from office",
        "high crimes and misdemeanors"
    ],

    # -------- Article III --------
    "Judicial Power": [
        "supreme court",
        "federal jurisdiction",
        "case or controversy clause",
        "original jurisdiction",
        "appellate jurisdiction",
        "trial by jury",
        "jurisdiction of the courts"
    ],
    "Treason": [
        "two witnesses",
        "corruption of blood",
        "treason definition"
    ],

    # -------- Article IV --------
    "Full Faith and Credit": [
        "public acts records",
        "full faith and credit clause"
    ],
    "Privileges and Immunities": [
        "privileges and immunities clause",
        "extradition",
        "fugitive from justice",
        "fugitive slave clause"
    ],
    "Admission of New States": [
        "new states",
        "territory clause",
        "property clause"
    ],
    "Guarantee of Republican Government": [
        "guarantee clause",
        "protection from invasion",
        "domestic violence"
    ],

    # -------- Article V–VII --------
    "Amendment Process": [
        "how to amend",
        "constitutional amendments process"
    ],
    "Supremacy Clause": [
        "supremacy of federal law",
        "debts assumption",
        "oath clause",
        "no religious test"
    ],
    "Ratification": [
        "ratification process"
    ],

    # -------- Bill of Rights (Amendments 1–10) --------
    "First Amendment": [
        "1st amendment",
        "freedom of religion",
        "establishment clause",
        "free exercise",
        "freedom of speech",
        "freedom of the press",
        "right to assemble",
        "right to petition",
        "freedom for individual",
        "Express diversity"
    ],
    "Second Amendment": [
        "2nd amendment",
        "right to bear arms",
        "keep and bear arms",
        "freedom for individual"
    ],
    "Third Amendment": [
        "3rd amendment",
        "quartering of soldiers",
        "no quartering",
        "freedom for individual"
    ],
    "Fourth Amendment": [
        "4th amendment",
        "unreasonable searches",
        "probable cause",
        "warrants",
        "arrest",
        "search and seizure",
        "freedom for individual"
    ],
    "Fifth Amendment": [
        "5th amendment",
        "due process",
        "double jeopardy",
        "self-incrimination",
        "eminent domain",
        "arrest",
        "right to remain silent",
        "miranda rights",
        "takings clause",
        "grand jury",
        "freedom for individual"
    ],
    "Sixth Amendment": [
        "6th amendment",
        "speedy trial",
        "public trial",
         "arrest",
        "impartial jury",
        "vicinage",
        "confront witnesses",
        "compulsory process",
        "right to counsel",
        "notice of accusation",
        "freedom for individual"
    ],
    "Seventh Amendment": [
        "7th amendment",
        "civil jury",
        "trial by jury in civil cases",
        "freedom for individual"
    ],
    "Eighth Amendment": [
        "8th amendment",
        "excessive bail",
        "excessive fines",
        "cruel and unusual punishment",
        "freedom for individual"
    ],
    "Ninth Amendment": [
        "9th amendment",
        "unenumerated rights",
        "rights retained by the people",
        "right to travel",
        "freedom for individual"
    ],
    "Tenth Amendment": [
        "10th amendment",
        "states rights",
        "reserved powers",
        "federalism"
    ],

    # -------- Amendments 11–12 --------
    "11th Amendment": [
        "eleventh amendment",
        "state sovereign immunity",
        "states can't be sued by citizens of other states"
    ],
    "12th Amendment": [
        "twelfth amendment",
        "electoral process",
        "electoral college reform",
        "president vice president voting"
    ],

    # -------- 13th Amendment --------
    "13th Amendment Section 1": [
        "thirteenth amendment",
        "13th amendment",
        "abolish slavery",
        "end of slavery",
        "involuntary servitude"
    ],
    "13th Amendment Section 2": [
        "thirteenth amendment enforcement",
        "13th amendment",
        "enforcement clause"
    ],

    # -------- 14th Amendment --------
    "14th Amendment Section 1": [
        "14th Amendment",
        "gay right",
        "fourteenth amendment",
        "14th amendment",
        "citizenship clause",
        "privileges or immunities",
        "due process clause",
        "equal protection clause",
        "incorporation",
        "civil rights",
        "Express diversity"
    ],
    "14th Amendment Section 2": [
        "14th Amendment",
        "apportionment",
        "penalty for denying vote",
        "representation reduction"
    ],
    "14th Amendment Section 3": [
        "14th Amendment",
        "insurrection disqualification",
        "officer disqualification",
        "section 3 disqualification"
    ],
    "14th Amendment Section 4": [
        "14th Amendment",
        "public debt",
        "confederate debt",
        "validity of public debt"
    ],
    "14th Amendment Section 5": [
        "14th Amendment",
        "enforcement clause",
        "congressional enforcement"
    ],

    # -------- 15th Amendment --------
    "15th Amendment Section 1": [
        "fifteenth amendment",
        "15th amendment",
        "voting rights for race",
        "black voting rights",
        "no race discrimination in voting"
    ],
    "15th Amendment Section 2": [
        "fifteenth amendment",
        "15th amendment",
        "congressional enforcement",
        "enforcement clause"
    ],

    # -------- 16th–17th --------
    "16th Amendment": [
        "sixteenth amendment",
        "income tax",
        "federal income tax"
    ],
    "17th Amendment": [
        "seventeenth amendment",
        "senator elections",
        "direct election of senators",
        "vacancies appointments"
    ],

    # -------- 18th (Prohibition) --------
    "18th Amendment Section 1": [
        "eighteenth amendment",
        "18th amendment",
        "prohibition",
        "alcohol ban",
        "intoxicating liquors"
    ],
    "18th Amendment Section 2": [
        "18th amendment enforcement",
        "concurrent power",
        "enforcement clause"
    ],
    "18th Amendment Section 3": [
        "18th amendment ratification deadline",
        "seven year ratification"
    ],

    # -------- 19th (Women’s suffrage) --------
    "19th Amendment Section 1": [
        "nineteenth amendment",
        "19th amendment",
        "women voting rights",
        "women's suffrage",
        "no sex discrimination in voting"
    ],
    "19th Amendment Section 2": [
        "19th amendment",
        "congressional enforcement",
        "enforcement clause"
    ],

    # -------- 20th (Lame Duck) --------
    "20th Amendment Section 1": [
        "twentieth amendment",
        "20th amendment",
        "term start dates",
        "lame duck amendment"
    ],
    "20th Amendment Section 2": [
        "20th amendment",
        "congressional meetings",
        "session start date"
    ],
    "20th Amendment Section 3": [
        "20th amendment",
        "presidential vacancy before term",
        "president elect fails to qualify"
    ],
    "20th Amendment Section 4": [
        "20th amendment",
        "death of persons in line",
        "contingent election procedures"
    ],
    "20th Amendment Section 5": [
        "20th amendment",
        "effective date of sections 1 and 2"
    ],
    "20th Amendment Section 6": [
        "20th amendment",
        "20th amendment ratification deadline",
        "seven year ratification"
    ],

    # -------- 21st (Repeal of Prohibition) --------
    "21st Amendment Section 1": [
        "twenty-first amendment",
        "21st amendment",
        "repeal prohibition",
        "18th repealed"
    ],
    "21st Amendment Section 2": [
         "21st amendment",
        "transport of liquors",
        "state alcohol control"
    ],
    "21st Amendment Section 3": [
        "21st amendment ratification deadline",
        "conventions ratification"
    ],

    # -------- 22nd (Term limits) --------
    "22nd Amendment Section 1": [
        "twenty-second amendment",
        "22nd amendment",
        "presidential term limits",
        "two term president"
    ],
    "22nd Amendment Section 2": [
        "twenty-second amendment",
        "22nd amendment ratification deadline"
    ],

    # -------- 23rd (DC electors) --------
    "23rd Amendment Section 1": [
        "twenty-third amendment",
        "23rd amendment",
        "dc voting rights",
        "washington dc electors",
        "district electors"
    ],
    "23rd Amendment Section 2": [
        "23rd amendment enforcement",
        "enforcement clause"
    ],

    # -------- 24th (Poll tax) --------
    "24th Amendment Section 1": [
        "twenty-fourth amendment",
        "24th amendment",
        "ban poll tax",
        "no voting tax"
    ],
    "24th Amendment Section 2": [
        "24th amendment enforcement",
        "enforcement clause"
    ],

    # -------- 25th (Succession & Disability) --------
    "25th Amendment Section 1": [
        "twenty-fifth amendment",
        "25th amendment",
        "presidential succession",
        "vp becomes president"
    ],
    "25th Amendment Section 2": [
        "vice presidential vacancy",
        "vp nomination and confirmation"
    ],
    "25th Amendment Section 3": [
        "presidential disability",
        "voluntary transfer of power",
        "acting president"
    ],
    "25th Amendment Section 4": [
        "involuntary transfer of power",
        "cabinet declaration",
        "acting president process"
    ],

    # -------- 26th (Voting age 18) --------
    "26th Amendment Section 1": [
        "twenty-sixth amendment",
        "26th amendment",
        "voting age 18",
        "vote at 18"
    ],
    "26th Amendment Section 2": [
        "26th amendment enforcement",
        "enforcement clause"
    ],

    # -------- 27th (Congressional pay) --------
    "27th Amendment": [
        "twenty-seventh amendment",
        "27th amendment",
        "congressional compensation",
        "salary change next term"
    ]
}
# ----------------------------
# Dedup helpers
# ----------------------------
lock = threading.Lock()

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _preload_existing() -> set[tuple[str, str]]:
    """Preload existing (alias_norm, title_norm) pairs from keywords in main collection."""
    existing_pairs = set()
    try:
        main_coll = get_main_collection()
        cur = main_coll.find({}, {"title": 1, "keywords": 1})
        for doc in cur:
            title = doc.get("title", "")
            title_norm = _norm(title)
            keywords = doc.get("keywords", [])

            for kw in keywords:
                if isinstance(kw, dict):
                    keyword = kw.get("keyword", "")
                    alias_norm = _norm(keyword)
                    if alias_norm and title_norm:
                        existing_pairs.add((alias_norm, title_norm))
    except Exception as e:
        logger.warning(f"Failed to preload existing keywords: {e}")
    return existing_pairs

def _unique_pairs(title_to_aliases: dict[str, list[str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for title, aliases in (title_to_aliases or {}).items():
        if not title or not isinstance(aliases, (list, tuple)):
            continue
        title = title.strip()
        title_norm = _norm(title)
        seen_alias_norms: set[str] = set()
        for alias in aliases:
            alias = (alias or "").strip()
            if not alias:
                continue
            an = _norm(alias)
            if an == title_norm:
                # skip alias identical to title after normalization
                continue
            if an in seen_alias_norms:
                continue
            seen_alias_norms.add(an)
            pairs.append((title, alias))
    return pairs

# Lazy initialization - only preload when script is run directly, not on import
existing: set[tuple[str, str]] = set()              # set of (alias_norm, title_norm)
seen_in_run: set[tuple[str, str]] = set()   # set of (alias_norm, title_norm)

# Only preload if script is run directly (not imported)
if __name__ == "__main__":
    existing = _preload_existing()

# ----------------------------
# Worker
# ----------------------------
def process_alias_with_embedding(title: str, alias: str, embedding: list):
    """
    Process a single alias with pre-generated embedding and add it as a keyword to the main document.
    Keywords are stored as: {keyword: str, embedding: List[float]}
    """
    alias_norm = _norm(alias)
    title_norm = _norm(title)
    key = (alias_norm, title_norm)

    with lock:
        if key in existing or key in seen_in_run:
            logger.info(f"Skip (duplicate): '{alias}' -> '{title}'")
            return None
        seen_in_run.add(key)

    try:
        # Find the main document by title
        main_coll = get_main_collection()
        main_doc = main_coll.find_one({"title": title})
        if not main_doc:
            logger.warning(f"Main document not found for title: '{title}', skipping alias '{alias}'")
            return None

        # Check if keyword already exists
        keywords = main_doc.get("keywords", [])
        existing_keywords = {kw.get("keyword", "").lower(): kw for kw in keywords if isinstance(kw, dict)}

        if alias.lower() in existing_keywords:
            logger.info(f"Skip (exists in DB): '{alias}' -> '{title}'")
            return None

        # Add keyword to the document
        new_keyword = {
            "keyword": alias,
            "embedding": embedding
        }

        # Update document: add keyword to keywords array
        main_coll = get_main_collection()
        main_coll.update_one(
            {"title": title},
            {"$push": {"keywords": new_keyword}}
        )
        logger.info(f"Added keyword: '{alias}' -> '{title}'")
        return alias

    except Exception as e:
        logger.error(f"Failed: '{alias}' -> '{title}': {e}")
        return None

# ----------------------------
# Runner
# ----------------------------
def ingest_alias_map(title_to_aliases: dict[str, list[str]], max_workers: int = 8):
    pairs = _unique_pairs(title_to_aliases)

    # Filter out duplicates and collect valid aliases
    valid_pairs = []
    alias_texts = []
    alias_to_pair = {}  # Map alias text to (title, alias) pair

    for title, alias in pairs:
        alias_norm, title_norm = _norm(alias), _norm(title)
        key = (alias_norm, title_norm)
        with lock:
            if key in existing or key in seen_in_run:
                logger.info(f"Skip (pre-schedule dup): '{alias}' -> '{title}'")
                continue

        # Check if document exists and keyword doesn't exist
        main_coll = get_main_collection()
        main_doc = main_coll.find_one({"title": title})
        if not main_doc:
            logger.warning(f"Main document not found for title: '{title}', skipping alias '{alias}'")
            continue

        keywords = main_doc.get("keywords", [])
        existing_keywords = {kw.get("keyword", "").lower(): kw for kw in keywords if isinstance(kw, dict)}
        if alias.lower() in existing_keywords:
            logger.info(f"Skip (exists in DB): '{alias}' -> '{title}'")
            continue

        valid_pairs.append((title, alias))
        alias_texts.append(alias)
        alias_to_pair[alias] = (title, alias)

    if not alias_texts:
        logger.info("No aliases to process after filtering.")
        return

    # Generate embeddings in batch
    logger.info(f"Generating embeddings for {len(alias_texts)} aliases using batch API...")
    try:
        embedder = get_chatgpt()
        embeddings = embedder.get_openai_embeddings_batch(alias_texts, batch_size=100)
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        return

    # Process aliases with their embeddings
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for i, (title, alias) in enumerate(valid_pairs):
            if i < len(embeddings) and embeddings[i] is not None:
                vec_list = embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i])
                tasks.append(executor.submit(process_alias_with_embedding, title, alias, vec_list))
            else:
                logger.warning(f"Missing embedding for alias '{alias}' -> '{title}'")

        for fut in as_completed(tasks):
            _ = fut.result()

    logger.info("All aliases processed.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ingest_alias_map(alias_map, max_workers=8)
