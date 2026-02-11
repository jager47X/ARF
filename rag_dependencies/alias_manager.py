# src/rag/alias_manager.py
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import nltk
from nltk.corpus import stopwords
from time import perf_counter

logger = logging.getLogger(__name__)

class AliasManager:
    """
    Loads alias vectors and computes semantic alias matches with cosine similarity.
    Also exposes helpers to normalize references (Amendments, Articles/Sections).
    """

    _ORD_TYPO_RX = re.compile(r"\b(\d+)th\b", re.I)
    _ORD_SUFFIX = {1: "st", 2: "nd", 3: "rd"}

    def __init__(self, db, config: Dict[str, Any]):
        self.db = db
        self.config = config or {}
        self.alias_cache: List[Tuple[str, str, np.ndarray]] = []

        # logging controls
        self.log_top: int = int(self.config.get("alias_log_top", 10))
        self.min_log_score: float = float(self.config.get("alias_min_log_score", 0.25))
        self.log_all_alias_scores: bool = bool(self.config.get("alias_log_all_scores", False))
        self.log_max_dump: int = int(self.config.get("alias_log_max_dump", 100000))

        self._stop: set[str] = set()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[ALIAS][__init__] config: log_top=%d min_log_score=%.3f log_all=%s max_dump=%d",
                self.log_top, self.min_log_score, self.log_all_alias_scores, self.log_max_dump
            )

        t0 = perf_counter()
        self._load_cache()
        t1 = perf_counter()
        logger.info("[ALIAS][__init__] cache load done in %.3f s (size=%d)", (t1 - t0), len(self.alias_cache))

    # ---------- helpers ----------
    def _get_database(self):
        """Return a pymongo Database, supporting wrappers that expose `.db`."""
        db_attr = getattr(self.db, "db", None)
        return db_attr if db_attr is not None else self.db

    # ---------------- internal ----------------
    def _load_cache(self):
        """Load (alias, title, vector) from DB into memory.
        Now reads from main collection (us_constitution) where keywords are stored in 'keywords' array.
        Each keyword has: {keyword: str, embedding: List[float]}
        """
        self.alias_cache.clear()
        try:
            database = self._get_database()  # <- avoid truthiness on Database
            
            # Get main collection name from config
            main_coll_name = self.config.get("main_collection_name", "us_constitution")
            alias_coll_name = self.config.get("alias_collection_name")
            
            # Try main collection first (new structure with keywords array)
            coll = database[main_coll_name]
            
            # Also try legacy Alias_Map collection for backward compatibility
            alias_coll = None
            if alias_coll_name and alias_coll_name in database.list_collection_names():
                alias_coll = database[alias_coll_name]
            
            n_total = n_skipped = n_title = 0
            
            # Load from main collection (keywords stored in 'keywords' array)
            cur = coll.find({}, {"title": 1, "keywords": 1, "aliases": 1, "alias": 1, "embedding": 1, "vector": 1})
            
            for doc in cur:
                title = (doc.get("title") or "").strip() if isinstance(doc.get("title"), str) else None
                if not title:
                    continue
                
                # Load keywords array (new structure)
                keywords = doc.get("keywords", [])
                if keywords:
                    for kw in keywords:
                        if isinstance(kw, dict):
                            keyword = kw.get("keyword", "")
                            vec_raw = kw.get("embedding") or []
                            
                            if not keyword or not vec_raw:
                                continue
                            
                            try:
                                vec = np.array(vec_raw, dtype=float).ravel()
                                if vec.size > 0:
                                    keyword = keyword.strip()
                                    if keyword:
                                        self.alias_cache.append((keyword, title, vec))
                                        n_title += 1
                                        n_total += 1
                                else:
                                    n_skipped += 1
                            except Exception as e:
                                n_skipped += 1
                                logger.debug("[ALIAS][_load_cache] skip malformed keyword vector for %r->%r: %s", keyword, title, e)
                
                # Also support legacy 'aliases' array format (backward compatibility)
                aliases = doc.get("aliases", [])
                if aliases and isinstance(aliases, list):
                    vec_raw = doc.get("embedding") or doc.get("vector") or []
                    if vec_raw:
                        try:
                            vec = np.array(vec_raw, dtype=float).ravel()
                            if vec.size > 0:
                                for alias in aliases:
                                    if alias and isinstance(alias, str):
                                        alias = alias.strip()
                                        if alias:
                                            # Check if already in cache (avoid duplicates with keywords)
                                            if not any(a == alias and t == title for a, t, _ in self.alias_cache):
                                                self.alias_cache.append((alias, title, vec))
                                                n_title += 1
                                                n_total += 1
                        except Exception as e:
                            n_skipped += 1
                            logger.debug("[ALIAS][_load_cache] skip malformed vector for aliases %r->%r: %s", aliases, title, e)
                
                # Also support legacy single 'alias' field (backward compatibility)
                if doc.get("alias") and not keywords and not aliases:
                    alias = doc.get("alias")
                    vec_raw = doc.get("embedding") or doc.get("vector") or []
                    if vec_raw:
                        try:
                            vec = np.array(vec_raw, dtype=float).ravel()
                            if vec.size > 0:
                                alias = alias.strip() if isinstance(alias, str) else str(alias)
                                if alias:
                                    if not any(a == alias and t == title for a, t, _ in self.alias_cache):
                                        self.alias_cache.append((alias, title, vec))
                                        n_title += 1
                                        n_total += 1
                        except Exception as e:
                            n_skipped += 1
                            logger.debug("[ALIAS][_load_cache] skip malformed vector for alias %r->%r: %s", alias, title, e)
            
            # Also load from legacy Alias_Map collection if it exists (for backward compatibility during migration)
            if alias_coll:
                logger.info("[ALIAS][_load_cache] Also loading from legacy Alias_Map collection")
                cur = alias_coll.find({}, {"alias": 1, "title": 1, "vector": 1, "embedding": 1})
                
                for doc in cur:
                    n_total += 1
                    alias = doc.get("alias")
                    title = (doc.get("title") or "").strip() if isinstance(doc.get("title"), str) else None
                    vec_raw = doc.get("embedding") or doc.get("vector") or []
                    
                    try:
                        vec = np.array(vec_raw, dtype=float).ravel()
                        if alias and title and vec.size > 0:
                            # Check if already in cache (avoid duplicates)
                            if not any(a == alias and t == title for a, t, _ in self.alias_cache):
                                self.alias_cache.append((alias, title, vec))
                                n_title += 1
                            else:
                                n_skipped += 1
                        else:
                            n_skipped += 1
                    except Exception as e:
                        n_skipped += 1
                        logger.debug("[ALIAS][_load_cache] skip malformed vector for %r->%r: %s", alias, title, e)

            logger.info(
                "[ALIAS][_load_cache] cached=%d (title=%d) skipped=%d total=%d",
                len(self.alias_cache), n_title, n_skipped, n_total
            )
            if logger.isEnabledFor(logging.DEBUG) and self.alias_cache:
                for i, (a, c, v) in enumerate(self.alias_cache[:3]):
                    logger.debug("[ALIAS][_load_cache] sample#%d alias=%r name=%r dim=%d", i+1, a, c, v.size)
        except Exception as e:
            logger.exception("[ALIAS][_load_cache] Failed to load aliases: %s", e)


    def _fix_ordinal_typos(self, text: str) -> str:
        def _fix(m):
            n = int(m.group(1))
            suf = self._ORD_SUFFIX.get(n % 10, "th")
            if 11 <= (n % 100) <= 13:
                suf = "th"
            return f"{n}{suf}"
        out = self._ORD_TYPO_RX.sub(_fix, text or "")
        if out != text and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][clean] fixed ordinal typo: %r -> %r", text, out)
        return out

    # ---------------- public ----------------
    def clean_query(self, query: str) -> str:
        patt = self.config.get("patterns")
        s0 = query or ""
        s = self._fix_ordinal_typos(s0.lower())  # << make '2th' consistent across the stack

        if patt:
            try:
                s = re.sub(patt, "", s, flags=re.IGNORECASE)
                logger.debug("[ALIAS][clean] applied regex patt=%r", patt)
            except re.error as e:
                logger.warning("[ALIAS][clean] invalid regex in patterns=%r (%s)", patt, e)

        toks_all = re.findall(r"\w+", s)

        # lazy-load stopwords with safe fallback
        if not self._stop:
            try:
                self._stop = set(stopwords.words("english"))
                logger.debug("[ALIAS][clean] NLTK stopwords loaded (size=%d)", len(self._stop))
            except (LookupError, OSError) as e:
                # Try to download stopwords if not available
                try:
                    logger.info("[ALIAS][clean] NLTK stopwords not found, attempting to download...")
                    nltk.download("stopwords", quiet=True)
                    self._stop = set(stopwords.words("english"))
                    logger.info("[ALIAS][clean] NLTK stopwords downloaded and loaded (size=%d)", len(self._stop))
                except Exception as download_error:
                    logger.warning(
                        "[ALIAS][clean] NLTK stopwords unavailable and download failed (%s); using fallback set",
                        download_error
                    )
                    self._stop = {
                        "a","an","the","and","or","but","if","while","is","are","to","of","in","on","for","with","as","by","at","from"
                    }
            except Exception as e:
                logger.warning("[ALIAS][clean] NLTK stopwords unavailable (%s); using fallback set", e)
                self._stop = {
                    "a","an","the","and","or","but","if","while","is","are","to","of","in","on","for","with","as","by","at","from"
                }

        toks = [t for t in toks_all if t not in self._stop]
        out = " ".join(toks)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[ALIAS][clean] raw=%r -> lower+regex=%r | tokens=%d -> %d (removed=%d) | out=%r",
                self._short(s0), self._short(s), len(toks_all), len(toks), max(0, len(toks_all)-len(toks)), self._short(out)
            )
        return out

    def find_semantic_aliases(self, cleaned_query: str, embedder) -> List[Tuple[str, str, float]]:
        t0 = perf_counter()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][find] cleaned_query=%r", self._short(cleaned_query))

        # --- get embedding ---
        try:
            vec = embedder.get_embedding(cleaned_query)
        except Exception as e:
            logger.exception("[ALIAS][find] embedder.get_embedding failed: %s", e)
            return []

        if vec is None:
            logger.info("[ALIAS][find] no embedding returned for %r; skipping", self._short(cleaned_query))
            return []
        if not isinstance(vec, np.ndarray):
            try:
                vec = np.array(vec, dtype=float).ravel()
            except Exception as e:
                logger.info("[ALIAS][find] embedding not array-like for %r (%s); skipping", self._short(cleaned_query), e)
                return []
        if vec.size == 0:
            logger.info("[ALIAS][find] zero-sized embedding for %r; skipping", self._short(cleaned_query))
            return []

        qn = float(np.linalg.norm(vec)) or 1e-9
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][find] emb_dim=%d query_norm=%.6f cache_size=%d", vec.size, qn, len(self.alias_cache))

        # --- cosine similarities ---
        sims: List[Tuple[str, str, float]] = []
        n_err = 0
        for idx, (alias, name, avec) in enumerate(self.alias_cache):
            try:
                avec = np.asarray(avec, dtype=float).ravel()
                if avec.size == 0:
                    continue
                denom = (np.linalg.norm(avec) or 1e-9) * qn
                if denom <= 0:
                    continue
                sim = float(np.dot(vec, avec) / denom)
                if not np.isfinite(sim):
                    continue
                # clamp just in case
                if sim > 1.0: sim = 1.0
                if sim < -1.0: sim = -1.0
                sims.append((alias, name, sim))
            except Exception as e:
                n_err += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[ALIAS][find] skip cache#%d %r->%r due to %s", idx, alias, name, e)

        sims.sort(key=lambda x: x[2], reverse=True)
        t1 = perf_counter()

        # --- stats & dumps ---
        if sims:
            scores = np.array([s for (_a, _n, s) in sims], dtype=float)
            s_min, s_max = float(scores.min()), float(scores.max())
            s_mean, s_std = float(scores.mean()), float(scores.std())
            logger.info(
                "[ALIAS][find] matches=%d errors=%d time=%.3fs score[min=%.3f max=%.3f mean=%.3f std=%.3f]",
                len(sims), n_err, (t1 - t0), s_min, s_max, s_mean, s_std
            )

            dump_level = logging.INFO if self.log_all_alias_scores else logging.DEBUG
            max_dump = int(self.log_max_dump)
            for rank, (a, n, s) in enumerate(sims[:max_dump], start=1):
                logger.log(dump_level, "[ALIAS][ALL] #%04d sim=%.6f alias=%r name=%r", rank, s, a, n)

            # thresholded summary at DEBUG
            if logger.isEnabledFor(logging.DEBUG):
                top = [(a, n, float(s)) for (a, n, s) in sims if s >= self.min_log_score][: self.log_top]
                logger.debug(
                    "[ALIAS][find] top%d (>=%.2f): %s",
                    self.log_top, self.min_log_score,
                    [(a, round(s, 3)) for (a, _n, s) in top]
                )
                if not top:
                    head = [(a, round(s, 3)) for (a, _n, s) in sims[: min(3, len(sims))]]
                    logger.debug("[ALIAS][find] head (no hits >= %.2f): %s", self.min_log_score, head)
        else:
            logger.info("[ALIAS][find] no alias matches computed (cache empty? size=%d)", len(self.alias_cache))

        return sims  # [(alias, canonical_or_title, score)]

    # ---------- reference normalization helpers ----------
    @staticmethod
    def normalize_amendment_title(ref: str) -> Optional[str]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][normalize_amendment_title] in=%r", ref)
        if not ref:
            return None
        s = ref.strip()

        WORD_TO_NUM = {
            "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,
            "eleventh":11,"twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,"seventeenth":17,
            "eighteenth":18,"nineteenth":19,"twentieth":20,"twenty-first":21,"twenty-second":22,"twenty-third":23,
            "twenty-fourth":24,"twenty-fifth":25,"twenty-sixth":26,"twenty-seventh":27
        }
        def to_ord(n:int)->str:
            if 11 <= (n % 100) <= 13: suf="th"
            else: suf={1:"st",2:"nd",3:"rd"}.get(n%10,"th")
            return f"{n}{suf}"

        # Use [a-zA-Z -]+ to explicitly match both uppercase and lowercase letters
        m = re.match(r"(?i)^\s*((?:\d+(?:st|nd|rd|th)?)|(?:[a-zA-Z -]+))\s+Amendment(?:\s*(?:Section|§)\s*(\d+))?\s*$", s)
        if not m:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[ALIAS][normalize_amendment_title] no match for %r", ref)
            return None
        head, sec = m.group(1), m.group(2)
        hl = head.lower().strip()

        # Check if it's a numeric ordinal (e.g., "14th", "5th") or just digits
        if hl.isdigit() or hl.endswith(("st","nd","rd","th")):
            import re as _re
            try:
                num = int(_re.sub(r"(st|nd|rd|th)$","",hl))
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[ALIAS][normalize_amendment_title] cannot parse numeric ordinal from %r", hl)
                return None
        else:
            # Check word ordinals (e.g., "fifth", "sixth", "fourteenth")
            num = WORD_TO_NUM.get(hl)
            if num is None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[ALIAS][normalize_amendment_title] unknown word ordinal %r", hl)
                return None

        base = f"{to_ord(num)} Amendment"
        out = f"{base} Section {sec}" if sec else base
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][normalize_amendment_title] out=%r", out)
        return out

    @staticmethod
    def parse_article_section(ref: str) -> Optional[tuple[str, Optional[str]]]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][parse_article_section] in=%r", ref)
        if not ref:
            return None

        def to_roman(n:int)->str:
            vals=[(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
            out=[]
            for v,sym in vals:
                while n>=v:
                    out.append(sym); n-=v
            return "".join(out)

        m = re.match(r"(?i)^\s*Article\s+([ivxlcdm]+|\d+)(?:\s*(?:Section|§)\s*(\d+))?\s*$", (ref or "").strip())
        if not m:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[ALIAS][parse_article_section] no match for %r", ref)
            return None
        part, sec = m.group(1), m.group(2)
        art = f"Article {part.upper()}" if not part.isdigit() else f"Article {to_roman(int(part))}"
        out = (art, f"Section {sec}" if sec else None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ALIAS][parse_article_section] out=%r", out)
        return out

    @staticmethod
    def _short(s: str, limit: int = 120) -> str:
        if s is None:
            return ""
        return s if len(s) <= limit else (s[: limit - 1] + "…")
