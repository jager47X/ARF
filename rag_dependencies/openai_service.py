import hashlib
import logging
import re
import time
from collections import deque
from threading import Lock
from typing import Optional

import numpy as np
import tiktoken
from openai import OpenAI

# Import from services.rag.config
from services.rag.config import EMBEDDING_MODEL, OPENAI_API_KEY, REASONINGMODEL, VOYAGE_API_KEY

# IMPORTANT: MAX_TOTAL_TOKENS is ONLY for embeddings (Voyage-3-large: 32K tokens)
# LLM operations (GPT-5.2) support 128K tokens and should NOT use this limit
MAX_TOTAL_TOKENS = 32000  # Voyage-3-large supports up to 32,000 tokens per embedding request
# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------

## LLM Service
#llm = LLM()  # Uses OpenAI by default

# ---------------------------
class EmbeddingBackend:
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str):
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        # identical contract as your previous get_openai_embedding
        logger.info("Generating embedding (OpenAI) ...")
        resp = openai_client.embeddings.create(model=self.model, input=text)
        vec = resp.data[0].embedding
        logger.info("Embedding generated (OpenAI).")
        return np.array(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for a batch of texts using OpenAI API"""
        if not texts:
            return []
        logger.info("Generating batch embeddings (OpenAI) for %d texts...", len(texts))
        try:
            resp = openai_client.embeddings.create(model=self.model, input=texts)
            embeddings = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            logger.info("Generated batch embeddings (OpenAI) for %d texts.", len(embeddings))
            return embeddings
        except Exception as e:
            logger.error("Error generating batch embeddings (OpenAI): %s", e)
            raise

class RateLimiter:
    """Simple rate limiter to stay under requests per minute limit"""
    def __init__(self, max_requests_per_minute: int = 1500):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = Lock()
        # Minimum delay between requests: 60 seconds / max_requests = time per request
        # For 1500 req/min: 60/1500 = 0.04 seconds = 40ms per request
        self.min_delay = 60.0 / max_requests_per_minute
        self.last_request_time = 0.0

    def wait_if_needed(self, num_requests: int = 1):
        """Wait if needed to stay under rate limit - MUST be called BEFORE making request"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            while self.request_times and self.request_times[0] < now - 60:
                self.request_times.popleft()

            # Check if we need to wait
            if len(self.request_times) + num_requests > self.max_requests:
                # Calculate how long to wait
                oldest_time = self.request_times[0] if self.request_times else now
                wait_time = 60 - (now - oldest_time) + 0.1  # Add small buffer
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.2f}s to stay under {self.max_requests} req/min")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.request_times and self.request_times[0] < now - 60:
                        self.request_times.popleft()

            # Ensure minimum delay since last request
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_delay * num_requests:
                wait_needed = (self.min_delay * num_requests) - time_since_last
                if wait_needed > 0:
                    time.sleep(wait_needed)
                    now = time.time()

            # Record the requests BEFORE making the API call
            for _ in range(num_requests):
                self.request_times.append(now)
            self.last_request_time = now

# Global rate limiter for Voyage AI (shared across all instances)
# Set to 1500/min to stay safely under 2000/min limit (conservative for parallel requests)
_voyage_rate_limiter = RateLimiter(max_requests_per_minute=1500)

class VoyageEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        try:
            import voyageai
            self.client = voyageai.Client(api_key=api_key or VOYAGE_API_KEY)
            logger.info("VoyageEmbeddingBackend initialized with model: %s", model)
        except ImportError:
            raise ImportError("voyageai package is required. Install it with: pip install voyageai")
        except Exception as e:
            logger.error("Failed to initialize Voyage AI client: %s", e)
            raise

    def embed(self, text: str) -> np.ndarray:
        # Rate limit: 1 request - MUST be called BEFORE making the API call
        _voyage_rate_limiter.wait_if_needed(num_requests=1)
        logger.info("Generating embedding (Voyage AI) ...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Voyage AI API: embed method with model and input
                result = self.client.embed([text], model=self.model, input_type="document")
                vec = result.embeddings[0]
                logger.info("Embedding generated (Voyage AI).")
                return np.array(vec, dtype=np.float32)
            except Exception as e:
                error_str = str(e)
                if "rate limit" in error_str.lower() or "rpm" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        # Wait for rate limiter before retrying
                        _voyage_rate_limiter.wait_if_needed(num_requests=1)
                        continue
                logger.error("Error generating Voyage AI embedding: %s", e)
                raise

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for a batch of texts using Voyage AI API"""
        if not texts:
            return []
        logger.info("Generating batch embeddings (Voyage AI) for %d texts...", len(texts))
        # Rate limit: batch counts as 1 API request (even though it contains multiple texts)
        _voyage_rate_limiter.wait_if_needed(num_requests=1)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Voyage AI API supports batch embedding natively
                result = self.client.embed(texts, model=self.model, input_type="document")
                embeddings = [np.array(vec, dtype=np.float32) for vec in result.embeddings]
                logger.info("Generated batch embeddings (Voyage AI) for %d texts.", len(embeddings))
                return embeddings
            except Exception as e:
                error_str = str(e)
                if "rate limit" in error_str.lower() or "rpm" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        # Wait for rate limiter before retrying
                        _voyage_rate_limiter.wait_if_needed(num_requests=1)
                        continue
                logger.error("Error generating batch embeddings (Voyage AI): %s", e)
                raise

class LocalEmbeddingBackend(EmbeddingBackend):
    """
    Prefer sentence-transformers if available; otherwise deterministic hash embedding.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            logger.info("LocalEmbeddingBackend: loaded '%s'", self.model_name)
        except Exception as e:
            logger.warning("LocalEmbeddingBackend: sentence-transformers unavailable (%s). "
                           "Falling back to deterministic hash embeddings.", e)

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
        # Deterministic pseudo-embedding (stable across runs)
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big", signed=False)
        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-8)
        return v

    def embed(self, text: str) -> np.ndarray:
        if self._st_model is not None:
            try:
                v = self._st_model.encode(text or "", normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(v, dtype=np.float32)
            except Exception as e:
                logger.warning("LocalEmbeddingBackend: encode failed (%s). Using hash embedding.", e)
        return self._hash_embed(text or "")

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for a batch of texts using local model"""
        if not texts:
            return []
        logger.info("Generating batch embeddings (Local) for %d texts...", len(texts))
        if self._st_model is not None:
            try:
                # sentence-transformers encode_batch is more efficient
                embeddings = self._st_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                )
                return [np.asarray(emb, dtype=np.float32) for emb in embeddings]
            except Exception as e:
                logger.warning("LocalEmbeddingBackend: batch encode failed (%s). Using hash embeddings.", e)
        # Fallback to hash embeddings
        return [self._hash_embed(text or "") for text in texts]

# ---------------------------
# Backends: LLM
# ---------------------------
class LLMBackend:
    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.3) -> str:
        raise NotImplementedError

class OpenAILLMBackend(LLMBackend):
    """
    OpenAI LLM backend - supports GPT-5.2 with 128K token context window.
    Input prompts are NOT truncated and can use the full 128K tokens.
    max_tokens parameter is for OUTPUT/completion tokens only.
    """
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.3) -> str:
        # Note: prompt is passed directly without truncation (supports 128K tokens for GPT-5.2)
        resp = openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,  # This is for OUTPUT tokens, not input
            temperature=temperature,
        )
        # Prefer .message.content but guard older SDKs
        text = getattr(resp.choices[0], "message", None)
        text = (text.content if text else getattr(resp.choices[0], "text", "")).strip()
        return text

# ---------------------------
# Main Service
# ---------------------------
class LLM:
    def __init__(
        self,
        *,
        config= None,
    ):
        """
        LLM service using OpenAI backends.
        """
        self.backend_name = "openai"
        self.config=config
        self.document_type=self.config.get("document_type", "unknown") if config else "unknown"
        self.reasoning_model = REASONINGMODEL
        self.embedding_model = EMBEDDING_MODEL
        self.unique_field = "title"

        # Use Voyage AI for embeddings by default
        if self.embedding_model.startswith("voyage"):
            self.emb_backend = VoyageEmbeddingBackend(self.embedding_model, api_key=VOYAGE_API_KEY)
        else:
            self.emb_backend = OpenAIEmbeddingBackend(self.embedding_model)
        self.llm_backend = OpenAILLMBackend(self.reasoning_model)
        logger.info("LLM using backends (embed=%s, llm=%s)", self.embedding_model, self.reasoning_model)

    # ---------------- Core operations (router to backends) ----------------
    def _chat(self, prompt: str, *, max_tokens: int = 220, temperature: float = 0.3) -> str:
        """
        LLM chat operation - supports full 128K tokens (GPT-5.2).
        Note: max_tokens parameter is for OUTPUT tokens, not input.
        Input prompts are NOT truncated and can use the full 128K token context.
        """
        return self.llm_backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def get_openai_embedding(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """
        Kept name for backward compatibility. Routes to OpenAI or Local backend.
        """
        text = self.truncate_text(text, max_tokens=MAX_TOTAL_TOKENS, model=model or self.embedding_model)
        return self.emb_backend.embed(text)

    def get_openai_embeddings_batch(self, texts: list[str], model: Optional[str] = None, batch_size: int = 100) -> list[np.ndarray]:
        """
        Generate embeddings for a batch of texts using batch API.
        Automatically handles truncation and batching for large lists.

        Args:
            texts: List of texts to embed
            model: Optional model override
            batch_size: Maximum number of texts per API call (default 100)

        Returns:
            List of numpy arrays (embeddings)
        """
        if not texts:
            return []

        # Truncate all texts first
        truncated_texts = [
            self.truncate_text(text, max_tokens=MAX_TOTAL_TOKENS, model=model or self.embedding_model)
            for text in texts
        ]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            try:
                if hasattr(self.emb_backend, 'embed_batch'):
                    batch_embeddings = self.emb_backend.embed_batch(batch)
                else:
                    # Fallback to individual calls if batch method not available
                    batch_embeddings = [self.emb_backend.embed(text) for text in batch]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Error in batch embedding (batch %d-%d): %s", i, i + len(batch), e)
                # Fallback to individual calls for this batch
                for text in batch:
                    try:
                        all_embeddings.append(self.emb_backend.embed(text))
                    except Exception as e2:
                        logger.error("Error embedding individual text: %s", e2)
                        # Add zero vector as fallback
                        all_embeddings.append(np.zeros(1024, dtype=np.float32))

        return all_embeddings

    # ---------------- High-level methods (now use _chat + embed router) ----------------
    def insight_explain(self, main_doc: dict, query: str, case_doc: list | None = None,
                            qm=None, db=None, knowledge_id=None, max_tokens: int = 220, language: str = "en") -> str:
            title = (main_doc.get("title") or main_doc.get("article") or main_doc.get("section") or "This provision").strip()
            body  = (main_doc.get("summary") or main_doc.get("text") or "").strip()
            case_names = []
            if isinstance(case_doc, list):
                for c in case_doc:
                    if isinstance(c, dict) and c.get("case"):
                        case_names.append(c["case"])
            cases_line = (f" Cases: {', '.join(case_names[:3])}." if case_names else "")

            # Get document_type from main_doc only - don't fall back to self.document_type (which might be US Constitution)
            # If document_type is missing, leave it empty (None) instead of defaulting
            document_type = main_doc.get("document_type") or None

            # Build domain-specific instruction only if document_type is available
            domain_instruction = ""
            if document_type:
                if language == "es":
                    domain_instruction = f"Sé específico con respecto al dominio {document_type}. Evita jerga legal.\n\n"
                else:
                    domain_instruction = f"Be specific to {document_type} domain. Avoid legalese.\n\n"

            # Language-specific prompts
            if language == "es":
                prompt = (
                    "Escribe 2-4 oraciones concisas en un solo párrafo en español.\n"
                    f"{domain_instruction}"
                    f"Comienza con un resumen del título y luego responde la pregunta de la consulta: \"{title} es \".\n"
                    "Responde la pregunta de la consulta basándote en las (razones) si es posible.\n"
                    f"CONSULTA DEL USUARIO:\n{query}\n\n"
                    f"DOCUMENTO:\n{body[:1200]}\n"
                    f"{cases_line}\n"
                    f"\nCRÍTICO: Solo usa información del TÍTULO y DOCUMENTO proporcionados arriba. "
                    f"No incluyas conocimiento externo, otras leyes, o información que no esté explícitamente "
                    f"establecida en el documento proporcionado. Tu respuesta debe basarse únicamente en el resumen "
                    f"y explicación de {title} y el contenido del DOCUMENTO.\n"
                    "\nIMPORTANTE: Responde completamente en español."
                )
            else:
                prompt = (
                    "Write 2-4 concise sentences as one paragraph.\n"
                    f"{domain_instruction}"
                    f"Begin with summary of title then answer query question. Begin sentence with summary: \"{title} is \".\n"
                    "Answer the question from query based on the reasons if possible.\n"
                    f"USER QUERY:\n{query}\n\n"
                    f"DOCUMENT:\n{body[:1200]}\n"
                    f"{cases_line}\n"
                    f"\nCRITICAL: Only use information from the provided TITLE and DOCUMENT sections above. "
                    f"Do not include any external knowledge, other laws, or information not explicitly stated "
                    f"in the provided document. Your response must be based solely on the summary and explanation "
                    f"of {title} and the content of the DOCUMENT.\n"
                )

            # Try whatever chat function your wrapper has
            try:
                if hasattr(self, "_chat"):
                    return (self._chat(prompt, max_tokens=max_tokens, temperature=0.3) or "").strip()
                if hasattr(self, "chat"):
                    return (self.chat(prompt, max_tokens=max_tokens, temperature=0.3) or "").strip()
                if hasattr(self, "complete_chat"):
                    msgs = [{"role": "user", "content": prompt}]
                    return (self.complete_chat(messages=msgs, max_tokens=max_tokens, temperature=0.3) or "").strip()
            except Exception as e:
                logging.exception("[LLM] insight_explain failed via chat backend: %s", e)

            # Last resort: cheap deterministic line (never empty)
            bits = [f"{title} is related because it frames the relevant powers and limits."]
            if body:
                bits.append(body[:300])
            return " ".join(bits).strip()

    def remove_personal_info(self, text: str) -> str:
        """
        Anonymize PII with semantic labels instead of [REDACTED].
        - PERSON names  -> "a person"
        - Emails        -> "an email"
        - Phone numbers -> "a phone number"
        - Addresses     -> reduced to "State/Prefecture City" if possible, else "a location"
        - Labeled fields (Name/Address/Email/Phone) -> category (with city-level collapse for Address)
        """
        import re
        if not text or len(str(text).strip()) < 1:
            return ""

        # ------------- 1) LLM-guided pass (category replacements) -------------
        prompt = (
            "Sanitize the following text by removing personal identifying information (PII) but "
            "REPLACING it with category phrases instead of [REDACTED].\n\n"
            "IMPORTANT: Only replace SPECIFIC, IDENTIFIABLE personal information. Do NOT replace general terms, "
            "legal terminology, country names, or geographic references.\n\n"
            "Rules - REPLACE ONLY:\n"
            "• Replace any SPECIFIC PERSON NAME (e.g., 'John Smith', 'Maria Garcia') with: a person\n"
            "  DO NOT replace legal terms like 'non-citizens', 'citizens', 'immigrants', 'a person', 'persons'\n"
            "• Replace any SPECIFIC email address (e.g., 'john@example.com') with: an email\n"
            "• Replace any SPECIFIC phone number (e.g., '555-123-4567') with: a phone number\n"
            "• Replace any SPECIFIC street address (e.g., '123 Main St, Vallejo, CA 94590') with only city-level info. "
            "  If you can infer city and state/prefecture, keep only '<State/Prefecture> <City>'. "
            "  If city-level cannot be determined, use: a location\n"
            "  DO NOT replace country names like 'U.S.', 'USA', 'United States', or general phrases like 'inside the u.s.'\n"
            "  DO NOT replace state names like 'California', 'New York', or general geographic references\n"
            "• For labeled fields like 'Name:', 'Address:', 'Email:', 'Phone:' do the same replacements.\n\n"
            "PRESERVE (DO NOT REPLACE):\n"
            "• Country names: 'U.S.', 'USA', 'United States', 'Mexico', 'Canada', etc.\n"
            "• State names: 'California', 'New York', 'Texas', etc.\n"
            "• Legal terms: 'non-citizens', 'citizens', 'immigrants', 'defendants', 'plaintiffs', etc.\n"
            "• General geographic references: 'inside the u.s.', 'in California', 'across the country', etc.\n"
            "• Common phrases and terminology that are not specific personal identifiers\n\n"
            "Examples of what to PRESERVE:\n"
            "- 'what constitutional rights do non-citizens have inside the u.s.?' → keep as-is (no PII)\n"
            "- 'immigrants in California' → keep as-is (no PII)\n"
            "- 'citizens of the United States' → keep as-is (no PII)\n\n"
            "Examples of what to REPLACE:\n"
            "- 'John Smith lives at 123 Main St, Vallejo, CA' → 'a person lives at California Vallejo'\n"
            "- 'Contact: john@example.com' → 'Contact: an email'\n"
            "- 'Phone: 555-123-4567' → 'Phone: a phone number'\n\n"
            "• Do not add explanations. Return only the sanitized text.\n"
            "• If there is no personal info, return the text unchanged.\n\n"
            f"Text:\n{text}\n"
        )

        logger.info("Removing personal info from text (backend=%s).", getattr(self, "backend_name", "openai"))
        cleaned_text = text
        try:
            # prefer your existing chat wrapper
            if hasattr(self, "_chat"):
                cleaned_text = self._chat(prompt, max_tokens=500, temperature=0.1).strip() or text
            elif hasattr(self, "chat"):
                cleaned_text = self.chat(prompt, max_tokens=500, temperature=0.1).strip() or text
            elif hasattr(self, "complete_chat"):
                msgs = [{"role": "user", "content": prompt}]
                cleaned_text = self.complete_chat(messages=msgs, max_tokens=500, temperature=0.1).strip() or text

            # handle any policy-style refusals from your wrapper (if you implemented it)
            if hasattr(self, "_handle_openai_refusal"):
                cleaned_text = self._handle_openai_refusal(cleaned_text) or text
        except Exception as e:
            logger.error("Error removing personal info: %s", e)
            cleaned_text = text

        # ------------- 1.5) Protect whitelisted terms from regex replacement -------------
        # Terms that should NEVER be replaced (legal terms, geographic references, etc.)
        # These are protected from both LLM and regex processing
        PROTECTED_TERMS = [
            # Country names and abbreviations (case-insensitive)
            r'\bU\.S\.\b', r'\bU\.S\.A\.\b', r'\bUSA\b', r'\bUnited States\b',
            # Legal terms (case-insensitive)
            r'\bnon-citizens\b', r'\bnoncitizens\b', r'\bcitizens\b', r'\bimmigrants\b',
            r'\bdefendants\b', r'\bplaintiffs\b', r'\bpersons\b', r'\ba person\b',
            # Common geographic references (preserve as-is)
            r'\binside the u\.s\.\b', r'\bin the u\.s\.\b', r'\bwithin the u\.s\.\b',
            r'\binside the U\.S\.\b', r'\bin the U\.S\.\b', r'\bwithin the U\.S\.\b',
        ]

        # Create placeholders for protected terms
        protected_map = {}
        placeholder_counter = 0
        s_protected = cleaned_text

        for pattern in PROTECTED_TERMS:
            matches = list(re.finditer(pattern, s_protected, re.IGNORECASE))
            for match in reversed(matches):  # Process in reverse to preserve indices
                placeholder = f"__PROTECTED_{placeholder_counter}__"
                protected_map[placeholder] = match.group(0)
                s_protected = s_protected[:match.start()] + placeholder + s_protected[match.end():]
                placeholder_counter += 1

        # ------------- 2) Deterministic regex fallbacks -------------
        s = s_protected

        # (a) Emails -> "an email"
        s = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "an email", s)

        # (b) Phone numbers -> "a phone number"
        # Matches international/US-style patterns loosely: +, (), spaces, dashes, at least 8 digits total
        s = re.sub(r"(?:\+?\d[\d\s().-]{6,}\d)", "a phone number", s)

        # (c) Labeled fields → category
        # Name
        s = re.sub(r"\b(Name|Full Name)\s*[:\-]\s*[^\n,]+", r"\1: a person", s, flags=re.IGNORECASE)
        # Email
        s = re.sub(r"\b(Email|E-mail)\s*[:\-]\s*[^\s]+", r"\1: an email", s, flags=re.IGNORECASE)
        # Phone
        s = re.sub(r"\b(Phone|Tel|Telephone|Mobile)\s*[:\-]\s*[\+\d\s().-]+", r"\1: a phone number", s, flags=re.IGNORECASE)

        # (d) Addresses → attempt to collapse to "<Region> <City>" (US & generic fallback)
        #
        # Pattern 1: US-ish "number street, City, State ZIP"
        #   Capture city/state and drop street + zip.
        def _us_addr_to_city_state(m):
            city = m.group("city").strip()
            state = m.group("state").strip()
            return f"{state} {city}"

        us_addr = re.compile(
            r"""
            \b
            \d{1,6}[\s\-]+
            [A-Za-z0-9.#\-\s]+?             # street name/body
            ,\s*
            (?P<city>[A-Za-z .'\-]+?)       # city
            ,\s*
            (?P<state>[A-Z]{2}|[A-Za-z .'\-]+)  # state or spelled out region
            (?:\s+\d{5}(?:-\d{4})?)?        # optional ZIP
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        )
        s = us_addr.sub(_us_addr_to_city_state, s)

        # Pattern 2: Generic "street-ish then City, Region" without a leading number
        def _generic_addr_to_city_region(m):
            city = (m.group("city") or "").strip()
            region = (m.group("region") or "").strip()
            if city and region:
                return f"{region} {city}"
            return "a location"

        generic_addr = re.compile(
            r"""
            \b
            (?:[A-Za-z0-9.#\-\s]+?,\s*)?    # optional first part (street/building)
            (?P<city>[A-Za-z .'\-]+?)       # city-ish
            ,\s*
            (?P<region>[A-Za-z .'\-]+)      # state/prefecture/region
            (?:\s+\d{3}\-\d{4}|\s+\d{5}(?:-\d{4})?)?  # optional JP/US postal
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        )
        s = generic_addr.sub(_generic_addr_to_city_region, s)

        # Pattern 3: Label-style Address: keep only city-level if present, else "a location"
        def _collapse_labeled_address(m):
            body = m.group("body")
            # Try to find "City, Region" inside the labeled block
            hit = us_addr.search(body) or generic_addr.search(body)
            if hit:
                # Re-run the same collapse by feeding match back into converters
                if hit.re is us_addr:
                    return f"{m.group('label')}: " + _us_addr_to_city_state(hit)
                else:
                    return f"{m.group('label')}: " + _generic_addr_to_city_region(hit)
            return f"{m.group('label')}: a location"

        # Only match explicit "Address:" or "Addr:" labels, not any word followed by colon
        labeled_address = re.compile(
            r"\b(Address|Addr)\s*[:\-]\s*(?P<body>.+?)(?=$|\n)",
            re.IGNORECASE | re.DOTALL,
        )
        def _collapse_labeled_address_safe(m):
            body = m.group("body")
            # Try to find "City, Region" inside the labeled block
            hit = us_addr.search(body) or generic_addr.search(body)
            if hit:
                # Re-run the same collapse by feeding match back into converters
                if hit.re is us_addr:
                    return f"{m.group(1)}: " + _us_addr_to_city_state(hit)
                else:
                    return f"{m.group(1)}: " + _generic_addr_to_city_region(hit)
            return f"{m.group(1)}: a location"
        s = labeled_address.sub(_collapse_labeled_address_safe, s)

        # (e) Aggressive street-only fallback: if something still looks like "123 Foo St/Ave/Blvd..."
        street_fallback = re.compile(
            r"""
            \b
            \d{1,6}\s+[A-Za-z0-9 .'\-#]+?
            \s(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Way|Place|Pl)
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        )
        s = street_fallback.sub("a location", s)

        # ------------- 3) Restore protected terms -------------
        for placeholder, original_term in protected_map.items():
            s = s.replace(placeholder, original_term)

        # ------------- 4) Return -------------
        return s


    def fix_query(self, query: str) -> str:
        if not query or len(str(query).strip()) < 1:
            return ""
        prompt = (
            "The user may have spelling or grammar mistakes. "
            "Return only the corrected sentence, without any explanation or prefix.\n\n"
            "if the input is unable to change, return it unchanged.\n\n"
            f"Query: {query}\n"
        )
        logger.info("Fixing query (backend=%s).", self.backend_name)
        try:
            out = self._chat(prompt, max_tokens=150, temperature=0.1).strip()
            out = self._handle_openai_refusal(out) or query
        except Exception as e:
            logger.error("Error Fixing query: %s", e)
            return query
        out = re.sub(r"^(corrected|fixed|edited|correction|grammar fixed)\s*[:\-]\s*", "", out, flags=re.IGNORECASE).strip()
        return out or query

    def truncate_text(self, text, max_tokens=MAX_TOTAL_TOKENS, model=EMBEDDING_MODEL):
        """
        Truncate text for EMBEDDING operations only (Voyage-3-large: 32K tokens).

        WARNING: Do NOT use this for LLM prompts! GPT-5.2 supports 128K tokens.
        LLM operations should pass prompts directly to _chat() without truncation.
        """
        try:
            encoding = tiktoken.encoding_for_model(model or self.embedding_model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text or "")
        if len(tokens) > max_tokens:
            logger.info("Text is too long (%d tokens). Truncating to %d tokens.", len(tokens), max_tokens)
            tokens = tokens[:max_tokens]
            text = encoding.decode(tokens)
        return text

    def _clean_summary(self, text: str) -> str:
        if not text:
            return ""
        s = str(text)
        s = re.sub(r"```(?:\w+)?\n([\s\S]*?)\n```", r"\1", s)
        s = s.replace('`', '')
        s = re.sub(r"^\s*Summary(?:\s*\(Similarity:\s*[0-9.]+\))?\s*[:\-–—]?\s*", '', s, flags=re.IGNORECASE)
        s = re.sub(r"\bSummary\s*:\s*", '', s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", ' ', s).strip()
        return s

    def rephrase_query(self, query: str, document_type: str, avoid_list: list) -> str:
        if not query or len(str(query).strip()) < 1:
            return ""
        avoid_text = f"\nAvoid using any of the following phrases: {', '.join(avoid_list)}" if avoid_list else ""

        # Generate domain-specific examples based on document_type
        examples = self._get_rephrase_examples(document_type)

        prompt = (
            f"You are an expert on the {document_type}. "
            "Rephrase the user's search query so it matches the language and terminology of the {document_type}, "
            "improving retrieval. Do not add new info or change intent. "
            "Output only the rewritten query.\n\n"
            f"Examples for {document_type}:\n{examples}\n"
            f"{avoid_text}\n\n"
            f"Document Type: {document_type}\n"
            f"Original Query: {query}\n"
            "Rewritten Query:"
        )
        logger.info(f"Rephrasing query for document_type='{document_type}' (backend=%s).", self.backend_name)
        try:
            out = self._chat(prompt, temperature=0.5, max_tokens=150).strip()
            out = self._handle_openai_refusal(out)
            if out is None:
                return None
            if out.lower().startswith("rephrased:"):
                out = out.split(":", 1)[1].strip()
            return out
        except Exception as e:
            logger.error("Error rephrasing query: %s", e)
            return None

    def _get_rephrase_examples(self, document_type: str) -> str:
        """Generate domain-specific examples for query rephrasing based on document_type"""
        document_type_lower = document_type.lower()

        if "constitution" in document_type_lower:
            return (
                "Original: Do illegal immigrants have a consistent right?\n"
                "Rephrased: Do criminals have a constitutional right?\n"
                "Original: What protections do immigrants have under the law?\n"
                "Rephrased: What protections do people have under the law?\n"
                "Original: can i take picture of police?\n"
                "Rephrased: freedom of press against authority?\n"
                "Original: Are immigrants allowed to protest peacefully?\n"
                "Rephrased: Are people allowed to rally peacefully?\n"
                "Original: can they arrest me for no reason?\n"
                "Rephrased: can any person be arrested without cause?\n"
            )
        elif "code" in document_type_lower and "united states" in document_type_lower:
            return (
                "Original: What are the rules for immigration?\n"
                "Rephrased: What are the statutory provisions for immigration?\n"
                "Original: Can I work in the US?\n"
                "Rephrased: What are the employment authorization requirements under United States Code?\n"
                "Original: What happens if I violate immigration law?\n"
                "Rephrased: What are the penalties for immigration violations under federal statute?\n"
                "Original: How do I get a visa?\n"
                "Rephrased: What are the visa application procedures under USC?\n"
            )
        elif "federal regulations" in document_type_lower or "cfr" in document_type_lower:
            return (
                "Original: What are the rules for student visas?\n"
                "Rephrased: What are the regulatory requirements for F-1 student visas under CFR?\n"
                "Original: Can I work while on a student visa?\n"
                "Rephrased: What are the employment authorization regulations for F-1 students?\n"
                "Original: What are the requirements for OPT?\n"
                "Rephrased: What are the Optional Practical Training eligibility requirements under Code of Federal Regulations?\n"
                "Original: How long can I stay on OPT?\n"
                "Rephrased: What is the duration limit for Optional Practical Training under federal regulations?\n"
            )
        elif "agency guidance" in document_type_lower or "guidance" in document_type_lower:
            return (
                "Original: What should I know about student visas?\n"
                "Rephrased: What are the DHS guidance documents for F-1 student visa holders?\n"
                "Original: Can I travel while on OPT?\n"
                "Rephrased: What does agency guidance say about travel authorization during Optional Practical Training?\n"
                "Original: What are the rules for STEM OPT?\n"
                "Rephrased: What are the USCIS guidance documents for STEM Optional Practical Training extension?\n"
                "Original: How do I maintain my status?\n"
                "Rephrased: What are the agency guidance requirements for maintaining F-1 student status?\n"
            )
        else:
            # Generic examples for unknown document types
            return (
                "Original: What are my rights?\n"
                "Rephrased: What legal protections are available?\n"
                "Original: Can I do this?\n"
                "Rephrased: What are the legal requirements and restrictions?\n"
                "Original: What happens if I violate the law?\n"
                "Rephrased: What are the legal consequences and penalties?\n"
            )

    def llm_verification(self, query: str, document: str) -> int:
        prompt = (
            "You are a legal reasoning assistant. Determine how relevant a given document "
            f"under {self.document_type}\n\n"
            f"Query:\n{query}\n\n"
            "Document:\n"
            f"{document}\n\n"
            "Task:\n"
            "- Judge whether the document is directly, indirectly, broadly, or not related to the query.\n"
            "- Identify who is involved (e.g., government, individual, state, or other).\n"
            "- If relevance is unclear, briefly explain why (ambiguity, missing terms, etc.).\n\n"
            "Calibration (0–9):\n"
            "  8–9  Direct match to the legal issue, controlling provision/case, or near-perfect factual/legal overlap.\n"
            "  6–7  Clearly related (same provision/case family or closely adjacent doctrine) but not a direct match.\n"
            "  4–5  Partial/contextual relevance (broader topic, background, or related actors) with some useful signal.\n"
            "  2–3  Tangential (mentions the topic area or actors but weak connection to the specific query).\n"
            "  0–1  Not relevant to the query.\n\n"
            "Tie-break rule: If uncertain between two bins, choose the HIGHER of the two.\n"
            "Ambiguous query rule: If the query is ambiguous but the document shares domain/key terms "
            "(e.g., same Article/Amendment/case line), prefer a mid score (4–6) rather than a low one.\n\n"
            "Output format (must follow exactly):\n"
            "Reasoning: <1-3 concise sentences explaining the connection or lack thereof>\n"
            "Score: <integer 0-9>"
        )

        try:
            text = self._chat(prompt, max_tokens=500, temperature=0.2).strip()
            logger.info("Verification response:\n%s", text)

            # Prefer an explicit 'Score: <digit>' first
            m = re.search(r"Score:\s*(\d)\b", text)
            score = int(m.group(1)) if m else None

            if score is None:
                # Fallback: take the last standalone digit in the text
                candidates = re.findall(r"\b(\d)\b", text)
                if candidates:
                    score = int(candidates[-1])
                else:
                    # Fallback to word numbers if ever produced
                    words = {
                        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
                    }
                    wm = re.search(r"Score:\s*(zero|one|two|three|four|five|six|seven|eight|nine)\b",
                                text, flags=re.I)
                    score = words.get(wm.group(1).lower(), None) if wm else None

            if score is None:
                logger.warning("No valid score detected in LLM output. Defaulting to 5 (neutral).")
                return 5

            # Clamp strictly to 0–9
            return max(0, min(score, 9))

        except Exception as e:
            logger.exception("Error during LLM verification: %s", e)
            # Fail soft, not harsh
            return 5

    def _handle_openai_refusal(self, response_text: str) -> str | None:
        if not response_text:
            return None
        patterns = [
            r"\bsorry[, ]+i\s+cannot\s+help\s+you",
            r"\bsorry[, ]+i\s+can't\s+help\s+you",
            r"\bi\s+cannot\s+assist\s+with\s+that\s+request",
            r"\bi\s+can't\s+assist\s+with\s+that\s+request",
            r"\bi\s+am\s+unable\s+to\s+help\s+with\s+that",
            r"\bi\s+am\s+not\s+able\s+to\s+do\s+that",
            r"\bi\s+cannot\s+fulfill\s+that\s+request",
            r"\bnot\s+ able\s+to\s+provide\s+that\s+information",
            r"\bviolates\s+content\s+policy",
            r"\bi'm\s+sorry[, ]*\s+but\s+i\s+cannot",
        ]
        tl = response_text.strip().lower()
        for p in patterns:
            if re.search(p, tl):
                logger.warning("Refusal detected from backend.")
                return None
        return response_text.strip()

    def check_moderation(self, text: str) -> dict:
        """
        Check if text violates OpenAI moderation policies.

        Args:
            text: Text to check for moderation violations

        Returns:
            dict with keys:
                - flagged: bool - Whether the text was flagged
                - categories: list - List of flagged category names
                - scores: dict - Dictionary of category scores
        """
        try:
            logger.info("[OpenAI][MODERATION] Checking text for moderation violations")
            moderation_response = openai_client.moderations.create(
                input=text,
                model="omni-moderation-latest"
            )
            moderation_result = moderation_response.results[0]

            result = {
                "flagged": moderation_result.flagged,
                "categories": [],
                "scores": {}
            }

            if moderation_result.flagged:
                # Extract flagged categories
                if hasattr(moderation_result, 'categories'):
                    categories = moderation_result.categories
                    category_names = ['hate', 'hate_threatening', 'harassment', 'harassment_threatening',
                                    'self_harm', 'self_harm_intent', 'self_harm_instructions', 'sexual',
                                    'sexual_minors', 'violence', 'violence_graphic']
                    result["categories"] = [
                        cat for cat in category_names
                        if hasattr(categories, cat) and getattr(categories, cat, False)
                    ]

                # Extract category scores
                if hasattr(moderation_result, 'category_scores'):
                    scores = moderation_result.category_scores
                    for cat in category_names:
                        if hasattr(scores, cat):
                            score_value = getattr(scores, cat, 0.0)
                            if isinstance(score_value, (int, float)) and score_value > 0:
                                result["scores"][cat] = score_value

                logger.warning("[OpenAI][MODERATION] Text flagged: %s", result["categories"])
            else:
                logger.info("[OpenAI][MODERATION] Text passed moderation check")

            return result
        except Exception as e:
            logger.exception("[OpenAI][MODERATION][ERROR] Failed to check moderation: %s", e)
            # Fail open - return not flagged if check fails
            return {
                "flagged": False,
                "categories": [],
                "scores": {}
            }

    def check_us_constitution_relevance(self, query: str) -> bool:
        """
        Check if a query is relevant to the US Constitution using LLM.

        Args:
            query: User's query text

        Returns:
            True if the query is relevant to US Constitution, False otherwise
        """
        prompt = f"""You are a legal expert. Determine if the following user query is directly related to the US Constitution, its Articles, Amendments, or constitutional law principles.

Query: "{query}"

Consider:
- Direct references to constitutional provisions (Articles, Amendments, Bill of Rights)
- Constitutional rights (freedom of speech, due process, equal protection, etc.)
- Constitutional powers (separation of powers, federalism, etc.)
- Constitutional interpretation and principles
- Historical constitutional context

NOT relevant:
- General legal questions unrelated to constitutional law
- State law questions (unless involving constitutional issues)
- Criminal or civil law questions (unless constitutional rights are involved)
- Administrative or regulatory questions (unless constitutional)
- General information requests

IMPORTANT: Respond with ONLY the word "YES" if the query is relevant to US Constitution, or ONLY the word "NO" if it is not relevant. Do not include any other text.

Response:"""

        try:
            logger.info("[OpenAI][TOPIC_CHECK] Checking US Constitution relevance for query: %s", query[:100])
            response = self._chat(prompt, max_tokens=20, temperature=0.1).strip().upper()

            # Handle various response formats - check if response contains YES or NO
            if not response:
                logger.warning("[OpenAI][TOPIC_CHECK] Empty response received, failing open")
                return True  # Fail open - assume relevant if we can't determine

            # Remove any punctuation and whitespace, check first word
            response_clean = response.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
            first_word = response_clean.split()[0] if response_clean.split() else ""

            is_relevant = first_word == "YES"
            logger.info("[OpenAI][TOPIC_CHECK] Query relevance: %s (raw response: %r, cleaned: %r, first_word: %r)",
                       is_relevant, response, response_clean, first_word)

            return is_relevant
        except Exception as e:
            logger.exception("[OpenAI][TOPIC_CHECK][ERROR] Failed to check relevance: %s", e)
            # Fail open - assume relevant if check fails (so queries proceed to RAG processing)
            logger.warning("[OpenAI][TOPIC_CHECK] Failing open - assuming query is relevant due to check failure")
            return True

    def generate_general_info(self, query: str, jurisdiction: Optional[str] = None, language: str = "en") -> str:
        """
        Generate general information response for off-topic queries.

        Args:
            query: User's original query
            jurisdiction: Optional jurisdiction/city for location-specific information
            language: Language for response ('en' or 'es')

        Returns:
            General information response text
        """
        is_spanish = language == "es"

        if jurisdiction and jurisdiction.strip():
            if is_spanish:
                prompt = f"""El usuario preguntó: "{query}"

Esta consulta no está directamente relacionada con la Constitución de EE.UU. El usuario está ubicado en o pregunta sobre {jurisdiction.strip()}.

Proporcione una respuesta de información general útil que:
1. Comience DIRECTAMENTE con la información general (PROHIBIDO usar frases como "Gracias por su pregunta", "Agradecemos su consulta", "En respuesta a su pregunta", o cualquier reconocimiento similar)
2. Proporcione información general o orientación relevante para {jurisdiction.strip()} si es posible
3. Sea profesional y útil
4. Mantenga la respuesta concisa (2-3 oraciones)
5. Termine con: "Esta es información general y no constituye asesoramiento legal."

IMPORTANTE: La primera oración debe ser la información general, sin ningún preámbulo.

Respuesta:"""
            else:
                prompt = f"""The user asked: "{query}"

This query is not directly related to the US Constitution. The user is located in or asking about {jurisdiction.strip()}.

Provide a helpful, general information response that:
1. Starts DIRECTLY with the general information (FORBIDDEN to use phrases like "Thank you for your question", "We appreciate your inquiry", "In response to your question", or any similar acknowledgments)
2. Provides general information or guidance relevant to {jurisdiction.strip()} if possible
3. Is professional and helpful
4. Keeps the response concise (2-3 sentences)
5. End with: "This is general information and does not constitute legal advice."

IMPORTANT: The first sentence must be the general information, with no preamble.

Response:"""
        else:
            if is_spanish:
                prompt = f"""El usuario preguntó: "{query}"

Esta consulta no está directamente relacionada con la Constitución de EE.UU. Proporcione una respuesta de información general útil que:
1. Comience DIRECTAMENTE con la información general (PROHIBIDO usar frases como "Gracias por su pregunta", "Agradecemos su consulta", "En respuesta a su pregunta", o cualquier reconocimiento similar)
2. Proporcione información general o orientación si es posible
3. Sea profesional y útil
4. Mantenga la respuesta concisa (2-3 oraciones)
5. Termine con: "Esta es información general y no constituye asesoramiento legal."

IMPORTANTE: La primera oración debe ser la información general, sin ningún preámbulo.

Respuesta:"""
            else:
                prompt = f"""The user asked: "{query}"

This query is not directly related to the US Constitution. Provide a helpful, general information response that:
1. Starts DIRECTLY with the general information (FORBIDDEN to use phrases like "Thank you for your question", "We appreciate your inquiry", "In response to your question", or any similar acknowledgments)
2. Provides general information or guidance if possible
3. Is professional and helpful
4. Keeps the response concise (2-3 sentences)
5. End with: "This is general information and does not constitute legal advice."

IMPORTANT: The first sentence must be the general information, with no preamble.

Response:"""

        try:
            logger.info("[OpenAI][GENERAL_INFO] Generating general info response (jurisdiction=%s, language=%s)", jurisdiction, language)
            response = self._chat(prompt, max_tokens=200, temperature=0.7)
            return response.strip()
        except Exception as e:
            logger.exception("[OpenAI][GENERAL_INFO][ERROR] Failed to generate general info: %s", e)
            raise


# Standalone translation function
def translate_insight(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Translate legal insight text between English and Spanish using OpenAI.

    Args:
        text: The text to translate
        source_lang: Source language code ('en' or 'es')
        target_lang: Target language code ('en' or 'es')

    Returns:
        Translated text or None if translation fails
    """
    if not text or not text.strip():
        return text

    if source_lang == target_lang:
        return text

    # Language direction
    if source_lang == "en" and target_lang == "es":
        direction = "Translate the following English legal text to Spanish"
        instructions = "Maintain the legal terminology and formal tone. Keep the same structure and meaning."
    elif source_lang == "es" and target_lang == "en":
        direction = "Translate the following Spanish legal text to English"
        instructions = "Maintain the legal terminology and formal tone. Keep the same structure and meaning."
    else:
        logger.warning(f"Unsupported language pair: {source_lang} -> {target_lang}")
        return None

    prompt = f"""{direction}.

{instructions}

Text to translate:
{text}

Translation:"""

    try:
        logger.info(f"Translating insight from {source_lang} to {target_lang}")
        response = openai_client.chat.completions.create(
            model=REASONINGMODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
            temperature=0.2,
        )
        translated = response.choices[0].message.content.strip()
        logger.info(f"Translation successful: {source_lang} -> {target_lang}")
        return translated
    except Exception as e:
        logger.error(f"Translation error ({source_lang} -> {target_lang}): {e}")
        return None


def translate_query(text: str, source_lang: str = "auto", target_lang: str = "en") -> Optional[str]:
    """
    Translate query text from Spanish to English for search purposes.

    Args:
        text: The query text to translate
        source_lang: Source language code ('en', 'es', or 'auto' for auto-detect)
        target_lang: Target language code (should be 'en' for search)

    Returns:
        Translated text or original text if translation fails or not needed
    """
    if not text or not text.strip():
        return text

    if source_lang == "en" or target_lang != "en":
        # Already English or not translating to English
        return text

    if source_lang == "auto":
        # Simple heuristic: check for common Spanish words/characters
        text_lower = text.lower()
        spanish_indicators = [
            "qué", "cómo", "dónde", "cuándo", "por qué", "quién", "cuál",
            "derechos", "ley", "caso", "tengo", "puedo", "debo", "necesito",
            "español", "española", "estado", "gobierno"
        ]
        has_spanish_chars = any(char in text for char in "áéíóúñüÁÉÍÓÚÑÜ¿¡")
        has_spanish_indicators = any(indicator in text_lower for indicator in spanish_indicators)

        if not (has_spanish_chars or has_spanish_indicators):
            # Likely English, no translation needed
            return text
        source_lang = "es"

    if source_lang == "es" and target_lang == "en":
        direction = "Translate the following Spanish query to English"
        instructions = "Maintain the legal terminology and meaning. This is a search query, so keep it concise and accurate."
    else:
        logger.warning(f"Unsupported query translation pair: {source_lang} -> {target_lang}")
        return text

    prompt = f"""{direction}.

{instructions}

Query to translate:
{text}

Translation:"""

    try:
        logger.info(f"Translating query from {source_lang} to {target_lang}")
        response = openai_client.chat.completions.create(
            model=REASONINGMODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
            temperature=0.1,
        )
        translated = response.choices[0].message.content.strip()
        logger.info(f"Query translation successful: {source_lang} -> {target_lang}")
        return translated
    except Exception as e:
        logger.error(f"Query translation error ({source_lang} -> {target_lang}): {e}")
        return text  # Return original on error
