# config.py
import os
from pathlib import Path

from dotenv import load_dotenv

# Global variable to track if env has been loaded
_env_loaded = False
_env_file_used = None

# Determine which .env file to use (same logic as main config.py)
def get_env_file(env_override: str = None) -> str:
    """Load .env file based on environment override or auto-detection

    Args:
        env_override: Optional environment name ('production', 'dev', 'local')
                     If None, auto-detects based on Docker and file existence
    """
    # Try to find .env files relative to the config file's directory (kyr-backend)
    # or in the project root
    config_dir = Path(__file__).parent.parent.parent  # Goes from config.py -> rag -> services -> kyr-backend
    project_root = config_dir.parent

    if env_override == "production":
        # Try kyr-backend first, then project root
        for base_dir in [config_dir, project_root]:
            env_file = base_dir / ".env.production"
            if env_file.exists():
                return str(env_file)
        # Fallback to .env if .env.production doesn't exist
        return ".env"
    elif env_override == "dev":
        for base_dir in [config_dir, project_root]:
            env_file = base_dir / ".env.dev"
            if env_file.exists():
                return str(env_file)
        return ".env"
    elif env_override == "local":
        for base_dir in [config_dir, project_root]:
            env_file = base_dir / ".env.local"
            if env_file.exists():
                return str(env_file)
        return ".env"

    # Auto-detect (original logic)
    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'

    if in_docker:
        # In Docker, check DOCKER_ENV environment variable to determine which .env file to use
        docker_env = os.environ.get('DOCKER_ENV', '').lower()

        if docker_env == 'local':
            # Dockerfile.local -> use .env.local
            for base_dir in [config_dir, project_root]:
                env_file = base_dir / ".env.local"
                if env_file.exists():
                    return str(env_file)
            return ".env.local"
        elif docker_env == 'dev':
            # Dockerfile.dev -> use .env.dev
            for base_dir in [config_dir, project_root]:
                env_file = base_dir / ".env.dev"
                if env_file.exists():
                    return str(env_file)
            return ".env.dev"
        elif docker_env == 'production':
            # Dockerfile.production -> use .env.production
            for base_dir in [config_dir, project_root]:
                env_file = base_dir / ".env.production"
                if env_file.exists():
                    return str(env_file)
            return ".env.production"
        else:
            # Docker but no DOCKER_ENV set, check for .env file (copied by Dockerfile)
            for base_dir in [config_dir, project_root]:
                env_default = base_dir / ".env"
                if env_default.exists():
                    return str(env_default)
            return ".env"

    # Not in Docker - check in both kyr-backend and project root
    for base_dir in [config_dir, project_root]:
        env_local = base_dir / ".env.local"
        env_default = base_dir / ".env"

        if env_local.exists():
            return str(env_local)
        elif env_default.exists():
            return str(env_default)

    # Final fallback
    return ".env"

def load_environment(env_override: str = None):
    """Load environment variables from the appropriate file

    Args:
        env_override: Optional environment name ('production', 'dev', 'local')
    """
    global _env_loaded, _env_file_used
    env_file = get_env_file(env_override)
    _env_file_used = env_file
    load_dotenv(env_file, override=True)
    _env_loaded = True
    print(f"[RAG Config] Loading environment from: {env_file}")

# Load environment variables from the appropriate file (default behavior)
# Can be overridden by calling load_environment() before importing config values
# Only auto-load if not already loaded (allows scripts to load with override first)
if not _env_loaded:
    load_environment()

# These values are read from environment variables
# If load_environment() is called with override=True after import,
# these will need to be re-read from os.getenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
EMBEDDING_MODEL = "voyage-3-large"
EMBEDDING_DIMENSIONS = 1024
TOP_QUERY_RESULT = 20              # Number of query results at once
QUERY_COLLECTION_NAME = "User_queries"
REASONINGMODEL="gpt-5.2"
REPHRASE_LIMIT=2 # Number of times to rephrase the query if no results are found
KEYWORD_MATCH_SCORE = 0.70  # Fixed score when keyword/alias/semantic/exact match is found (only applies if current score < 0.8)
# For MongoDB connection
# For Dataset
# ------------------------------
# RAG Parameters
# ------------------------------


# Client Case Search Thresholds (separate tuning for case matching)
CLIENT_CASE_THRESHOLDS = {
       "query_search": 0.75,            # Fixed — do not tune automatically
        "alias_search": 0.75, # Alias threshold
        "RAG_SEARCH_min": 0.70, # if lower than this dont even rephrase or search again
        "LLM_VERIFication": 0.75,        # check with LLM verification additionaly
        "RAG_SEARCH": 0.85,# anytime pass here, return as a result
        "confident": 0.85, #saves  the summaries only above this threshold
        "FILTER_GAP": 0.10,  # if there is higer gap comapre to the top result dont show the
        "LLM_SCORE": 0.10,
        "HYBRID_SEMANTIC_WEIGHT": 0.90,  # Weight for semantic scores in hybrid search (70%)
        "HYBRID_BM25_WEIGHT": 0.10,      # Weight for BM25 scores in hybrid search (30%)
}

# Domain-specific thresholds
# Each domain must have its own thresholds defined - no default fallback
# This ensures each domain uses thresholds optimized for its document characteristics
DOMAIN_THRESHOLDS = {
    "us_constitution": {
        "query_search": 0.65,            # Fixed — do not tune automatically
        "alias_search": 0.85, # Alias threshold
        "RAG_SEARCH_min": 0.65, # if lower than this dont even rephrase or search again
        "LLM_VERIFication": 0.70,        # Lowered from 0.70 to 0.45 to allow documents with semantic scores ~0.48-0.50 to be verified
        "RAG_SEARCH": 0.85,# anytime pass here, return as a result
        "confident": 0.85, #saves  the summaries only above this threshold
        "FILTER_GAP": 0.20,  # if there is higer gap comapre to the top result dont show the
        "LLM_SCORE": 0.10,
        "use_mlp_reranker": True,  # Toggle MLP reranking
        "mlp_uncertainty_low": 0.4,  # Below this = reject
        "mlp_uncertainty_high": 0.6,  # Above this = accept
        "mlp_model_path": "models/mlp_reranker.joblib",
    },
    "code_of_federal_regulations": {
        "query_search": 0.65,            # Fixed — do not tune automatically
        "alias_search": 0.85, # Alias threshold
        "RAG_SEARCH_min": 0.65, # if lower than this dont even rephrase or search again
        "LLM_VERIFication": 0.70,        # check with LLM verification additionaly
        "RAG_SEARCH": 0.85,# anytime pass here, return as a result
        "confident": 0.85, #saves  the summaries only above this threshold
        "FILTER_GAP": 0.20,  # if there is higer gap comapre to the top result dont show the
        "LLM_SCORE": 0.10,
        "use_mlp_reranker": True,  # Toggle MLP reranking
        "mlp_uncertainty_low": 0.4,  # Below this = reject
        "mlp_uncertainty_high": 0.6,  # Above this = accept
        "mlp_model_path": "models/mlp_reranker.joblib",
    },
    "us_code": {
        "query_search": 0.65,            # Fixed — do not tune automatically
        "alias_search": 0.85, # Alias threshold
        "RAG_SEARCH_min": 0.65, # if lower than this dont even rephrase or search again
        "LLM_VERIFication": 0.70,        # check with LLM verification additionaly
        "RAG_SEARCH": 0.85,# anytime pass here, return as a result
        "confident": 0.85, #saves  the summaries only above this threshold
        "FILTER_GAP": 0.20,  # if there is higer gap comapre to the top result dont show the
        "LLM_SCORE": 0.10,
        "use_mlp_reranker": True,  # Toggle MLP reranking
        "mlp_uncertainty_low": 0.4,  # Below this = reject
        "mlp_uncertainty_high": 0.6,  # Above this = accept
        "mlp_model_path": "models/mlp_reranker.joblib",
    },
    "uscis_policy": {
        "query_search": 0.65,            # Fixed — do not tune automatically
        "alias_search": 0.85, # Alias threshold
        "RAG_SEARCH_min": 0.65, # if lower than this dont even rephrase or search again
        "LLM_VERIFication": 0.70,        # check with LLM verification additionaly
        "RAG_SEARCH": 0.85,  # Lowered from 0.85 to 0.80 (effective 0.70 after -0.1) to match actual search scores (top results ~0.83)
        "confident": 0.85, #saves  the summaries only above this threshold
        "FILTER_GAP": 0.20,  # if there is higer gap comapre to the top result dont show the
        "LLM_SCORE": 0.10,
        "use_mlp_reranker": True,  # Toggle MLP reranking
        "mlp_uncertainty_low": 0.4,  # Below this = reject
        "mlp_uncertainty_high": 0.6,  # Above this = accept
        "mlp_model_path": "models/mlp_reranker.joblib",
    }
}
#
# ----------
# --------------------
# Base Document Bias Map
# ------------------------------
DOC_BIAS =  {"The U.S. Constitution":-0.2,"Second Amendment":-0.05,"Supremacy Clause":-0.05, "14th Amendment Section 1":-0.05,"14th Amendment Section 2":-0.05,"14th Amendment Section 3":-0.05,"14th Amendment Section 4":-0.05,"14th Amendment Section 5":-0.05,"Third Amendment":-0.1, "Powers of Congress":-0.1, "Privileges and Immunities":-0.1, "Powers denied to the States":-0.1}#"22nd Amendment Section 1":0.02,"The Congress":0.02,"The Senate":0.02,"The House of Representatives": 0.02, "Guarantee of Republican Government":0.01,"elections":-0.01,"privileges and immunities": -0.04,"Powers denied to the States": -0.04,"Second Amendment":0.01,"10th Amendment":0.005,"Ninth Amendment":0.015,"19th Amendment":0.01, "Fourth Amendment" : 0.01 ,"The Congress" : -0.01 ,"Elections":0.01,"Third Amendment" : -0.01, "Sixth Amendment" : 0.01,"First Amendment" : 0.00,  "Fifth Amendment" : 0.01}# "14th Amendment" : -0.04 ,  "Elections": 0.02   "Supremacy Clause" : 0.02 ,"15th Amendment" : 0.02 , "24th Amendment" : 0.02 , "26th Amendment" : 0.02 , "27th Amendment" : 0.02

COLLECTION = {
    "US_CONSTITUTION_SET": {
        "db_name": "public",
        "query_collection_name": QUERY_COLLECTION_NAME,
        "main_collection_name": "us_constitution",
        "cases_collection_name": "supreme_court_cases",
        # MongoDB Atlas Vector Search index names (default: "vector_index")
        "main_vector_index": "vector_index",
        "cases_vector_index": "vector_index",
        "query_vector_index": "vector_index",
        # MongoDB Atlas Full-Text Search index name (for keyword/BM25 search)
        "main_fulltext_index": "fulltext_index",
        "document_type": "US Constitution",
        "unique_index": "title",
        "patterns": r"\b(?:us\s*constitution|u\.s\.\s*constitution|the\s*constitution|constitution|constititution|constititon)\b",
        "bias" : DOC_BIAS,
        "sql_attached": False,
        "use_keyword_matcher": True,  # Enable KeywordMatcher for US Constitution (has structured articles/sections)
        "use_alias_search": True,  # Enable alias search for US Constitution (has aliases/keywords with embeddings)
        "disable_hybrid_search": True,  # Disable BM25/Atlas search, use only semantic search + ABC gates
        "thresholds": DOMAIN_THRESHOLDS["us_constitution"],  # Domain-specific thresholds
        # Field mapping: maps standard field names to collection-specific field names
        "field_mapping": {
            "title": "title",
            "article": "article",
            "section": "section",
            "chapter": None,  # Not used in US Constitution
            "part": None,  # Not used in US Constitution
            "subchapter": None,  # Not used in US Constitution
            "text": ["text", "summary", "content", "body"],
            "nested_text": ["clauses", "sections"]  # Arrays that contain text
        }
    },
    "CFR_SET": {
        "db_name": "public",
        "query_collection_name": QUERY_COLLECTION_NAME,
        "main_collection_name": "code_of_federal_regulations",
        # MongoDB Atlas Vector Search index names (default: "vector_index")
        "main_vector_index": "vector_index",
        "query_vector_index": "vector_index",
        # MongoDB Atlas Full-Text Search index name (for keyword/BM25 search)
        "main_fulltext_index": "fulltext_index",
        "document_type": "Code of Federal Regulations",
        "tag": "Federal Law",  # Tag for grouping search results
        "unique_index": "title",
        "patterns": r"\b(?:cfr|code\s*of\s*federal\s*regulations|federal\s*regulations|title\s*\d+\s*cfr|cfr\s*title\s*\d+)\b",
        "bias": {},
        "sql_attached": False,
        "use_keyword_matcher": False,  # Disable KeywordMatcher for CFR (avoids timeout on large collections)
        "use_alias_search": False,  # Disable alias search for CFR (no aliases/keywords ingested yet)
        "disable_hybrid_search": True,  # Disable BM25/Atlas search, use only semantic search + ABC gates
        "thresholds": DOMAIN_THRESHOLDS["code_of_federal_regulations"],  # Domain-specific thresholds
        # Field mapping: CFR uses article, part, chapter, subchapter, section
        "field_mapping": {
            "title": "title",
            "article": "article",
            "part": "part",
            "chapter": "chapter",
            "subchapter": "subchapter",
            "section": "section",
            "text": ["text", "summary", "content", "body"],
            "nested_text": ["sections"]  # CFR uses sections array
        }
    },
    "US_CODE_SET": {
        "db_name": "public",
        "query_collection_name": QUERY_COLLECTION_NAME,
        "main_collection_name": "us_code",
        # MongoDB Atlas Vector Search index names (default: "vector_index")
        "main_vector_index": "vector_index",
        "query_vector_index": "vector_index",
        # MongoDB Atlas Full-Text Search index name (for keyword/BM25 search)
        "main_fulltext_index": "fulltext_index",
        "document_type": "United States Code",
        "tag": "Federal Law",  # Tag for grouping search results
        "unique_index": "title",
        "patterns": r"\b(?:united\s*states\s*code|u\.s\.\s*code|us\s*code|usc|title\s*\d+\s*usc|usc\s*title\s*\d+)\b",
        "bias": {},
        "sql_attached": False,
        "use_keyword_matcher": False,  # Disable KeywordMatcher for US Code (avoids timeout on large collections)
        "use_alias_search": False,  # Disable alias search for US Code (no aliases/keywords ingested yet)
        "disable_hybrid_search": True,  # Disable BM25/Atlas search, use only semantic search + ABC gates
        "thresholds": DOMAIN_THRESHOLDS["us_code"],  # Domain-specific thresholds
        # Field mapping: US Code uses article, chapter, section, and clauses array
        "field_mapping": {
            "title": "title",
            "article": "article",
            "chapter": "chapter",
            "section": "section",
            "part": None,  # Not used in US Code
            "subchapter": None,  # Not used in US Code
            "text": ["text", "summary", "content", "body"],
            "nested_text": ["clauses"]  # US Code uses clauses array
        }
    },
    "USCIS_POLICY_SET": {
        "db_name": "public",
        "query_collection_name": QUERY_COLLECTION_NAME,
        "main_collection_name": "uscis_policy",
        "main_vector_index": "vector_index",
        "query_vector_index": "vector_index",
        # MongoDB Atlas Full-Text Search index name (for keyword/BM25 search)
        "main_fulltext_index": "fulltext_index",
        "document_type": "USCIS Policy",
        "unique_index": "title",
        "patterns": r"\b(?:uscis\s*policy|policy\s*manual|uscis\s*guidance)\b",
        "bias": {},
        "sql_attached": False,
        "use_keyword_matcher": False,  # Disable KeywordMatcher for USCIS Policy (avoids timeout on large collections)
        "use_alias_search": False,  # Disable alias search for USCIS Policy (no aliases/keywords ingested yet)
        "disable_hybrid_search": True,  # Disable BM25/Atlas search, use only semantic search + ABC gates
        "thresholds": DOMAIN_THRESHOLDS["uscis_policy"],  # Domain-specific thresholds
        # Autoupdate configuration
        "autoupdate_enabled": os.getenv("USCIS_AUTOUPDATE_ENABLED", "false").lower() == "true",  # Default: False, can be overridden by env var
        "autoupdate_url": "https://www.uscis.gov/policy-manual",
        # Field mapping: USCIS Policy uses title, references, and clauses
        "field_mapping": {
            "title": "title",
            "article": None,  # Not used
            "chapter": None,  # Not used
            "section": None,  # Not used
            "part": None,  # Not used
            "subchapter": None,  # Not used
            "text": ["text", "summary", "content", "body"],
            "nested_text": ["clauses"],  # USCIS Policy uses clauses array
            "references": "references"  # CFR references
        }
    },
    "CLIENT_CASES": {
        "db_name": "private",
        "query_collection_name": QUERY_COLLECTION_NAME,
        "main_collection_name": "client_cases",
        # MongoDB Atlas Vector Search index name
        "main_vector_index": "vector_index",
        # MongoDB Atlas Full-Text Search index name (for keyword search)
        "main_fulltext_index": "fulltext_index",
        "document_type": "client_case",
        "unique_index": "title",
        "sql_attached":True,
        "use_keyword_matcher": False,  # Disable KeywordMatcher for Client Cases (not needed for case search)
        "use_alias_search": False,  # Disable alias search for Client Cases (uses SQL path, alias not applicable)
        # Field mapping: Client cases use title (case name) and summary (case summary)
        "field_mapping": {
            "title": "title",  # Case name
            "article": None,  # Not used
            "chapter": None,  # Not used
            "section": None,  # Not used
            "part": None,  # Not used
            "subchapter": None,  # Not used
            "text": ["summary", "text", "content", "body"],  # Case summary
            "nested_text": []  # No nested structures
        }
    }
}

# Domain identifier to collection key mapping
DOMAIN_COLLECTION_MAP = {
    "us_constitution": "US_CONSTITUTION_SET",
    "us_code": "US_CODE_SET",
    "code_of_federal_regulations": "CFR_SET",
    "uscis_policy": "USCIS_POLICY_SET"
}

# Autoupdate configuration
# Controls automatic weekly updates for legal document collections
AUTOUPDATE_CONFIG = {
    "enabled": os.getenv("USCIS_AUTOUPDATE_ENABLED", "false").lower() == "true",  # Default: False, must be explicitly enabled
    "collections": ["USCIS_POLICY_SET"],  # Only USCIS for now
    "check_interval_days": 7,  # Weekly checks
    "last_check_file": "Data/Knowledge/.last_uscis_check"  # Path to file storing last check timestamp
}






