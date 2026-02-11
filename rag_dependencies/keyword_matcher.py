# src/rag/keyword_matcher.py
import re
from typing import Optional, List, Dict, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class KeywordMatcher:
    _ART_RX = re.compile(
        r"(?i)\barticle\s+("
        r"[ivxlcdm]+|\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth"
        r")\b"
    )
    _SEC_RX = re.compile(r"(?i)\b(?:section|§)\s*(\d+)\b")

    _ORD_TYPO_RX = re.compile(r"\b(\d+)th\b", re.I)
    _ORD_SUFFIX = {1: "st", 2: "nd", 3: "rd"}
    _WORD_TO_NUM = {
        "first":1,"one":1,"second":2,"two":2,"third":3,"three":3,"fourth":4,"four":4,"fifth":5,"five":5,
        "sixth":6,"six":6,"seventh":7,"seven":7,"eighth":8,"eight":8,"ninth":9,"nine":9,"tenth":10,"ten":10,
        "eleventh":11,"twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,"seventeenth":17,
        "eighteenth":18,"nineteenth":19,"twentieth":20,
        "twenty first":21,"twenty-first":21,"twenty second":22,"twenty-second":22,
        "twenty third":23,"twenty-third":23,"twenty fourth":24,"twenty-fourth":24,
        "twenty fifth":25,"twenty-fifth":25,"twenty sixth":26,"twenty-sixth":26,
    }

    def __init__(self, db):
        self.db = db

        # --- resolve main collection from MongoManager ---
        # MongoManager always sets self.main, so this should always exist
        self.main = self.db.main

        # --- caches populated below ---
        self._articles: List[str] = []
        self._titles: List[str] = []
        self._title_lc_to_title: Dict[str, str] = {}
        self._article_lc_to_article: Dict[str, str] = {}
        self.article_to_sections: Dict[str, Set[str]] = {}        # exact labels as stored (e.g., "Section 8")
        self._sections_by_article_lc: Dict[str, Set[str]] = {}    # lowercased for quick lookup

        # --- load distincts ---
        # Use aggregation pipeline with parallel execution and index hints for better performance
        try:
            # Parallelize article and title aggregation pipelines
            # MongoDB Atlas can handle concurrent queries, parallelism is controlled by application (ECS)
            def load_articles():
                article_pipeline = [
                    {"$match": {"article": {"$exists": True, "$ne": None, "$ne": ""}}},
                    {"$group": {"_id": "$article"}},
                    {"$project": {"article": "$_id"}}
                ]
                # Use index hint to force MongoDB to use article_idx (loads index into memory cache)
                try:
                    return [doc.get("article", "").strip() 
                           for doc in self.main.aggregate(
                               article_pipeline, 
                               allowDiskUse=True, 
                               maxTimeMS=60000,
                               hint={"article": 1}  # Force index usage - keeps index in MongoDB's cache
                           )
                           if doc.get("article")]
                except Exception:
                    # Fallback if hint fails (index might not exist yet)
                    return [doc.get("article", "").strip() 
                           for doc in self.main.aggregate(article_pipeline, allowDiskUse=True, maxTimeMS=60000)
                           if doc.get("article")]
            
            def load_titles():
                title_pipeline = [
                    {"$match": {"title": {"$exists": True, "$ne": None, "$ne": ""}}},
                    {"$group": {"_id": "$title"}},
                    {"$project": {"title": "$_id"}}
                ]
                # Use index hint to force MongoDB to use title_idx (loads index into memory cache)
                try:
                    return [doc.get("title", "").strip() 
                           for doc in self.main.aggregate(
                               title_pipeline, 
                               allowDiskUse=True, 
                               maxTimeMS=60000,
                               hint={"title": 1}  # Force index usage - keeps index in MongoDB's cache
                           )
                           if doc.get("title")]
                except Exception:
                    # Fallback if hint fails (index might not exist yet)
                    return [doc.get("title", "").strip() 
                           for doc in self.main.aggregate(title_pipeline, allowDiskUse=True, maxTimeMS=60000)
                           if doc.get("title")]
            
            # Execute both aggregations in parallel (MongoDB Atlas handles concurrent queries)
            with ThreadPoolExecutor(max_workers=2) as executor:
                article_future = executor.submit(load_articles)
                title_future = executor.submit(load_titles)
                
                arts = article_future.result()
                tits = title_future.result()
            
            self._articles = [a for a in arts if isinstance(a, str) and a.strip() and a.strip() != "Amendment"]
            self._titles   = [t for t in tits if isinstance(t, str) and t.strip()]
            self._title_lc_to_title    = {t.lower(): t for t in self._titles}
            self._article_lc_to_article = {a.lower(): a for a in self._articles}
        except Exception as e:
            logger.error("[KW] failed to load distinct titles/articles: %s", e)
            self._articles, self._titles = [], []
            self._title_lc_to_title, self._article_lc_to_article = {}, {}

        # --- build article → sections map ---
        # Use aggregation pipeline with index hint for better performance
        try:
            self.article_to_sections = {}
            # Use aggregation to group by article and collect unique sections
            pipeline = [
                {"$match": {"article": {"$exists": True, "$ne": None, "$ne": ""}}},
                {"$group": {
                    "_id": "$article",
                    "sections": {"$addToSet": "$section"}
                }},
                {"$project": {
                    "article": "$_id",
                    "sections": {"$filter": {
                        "input": "$sections",
                        "as": "sec",
                        "cond": {"$and": [
                            {"$ne": ["$$sec", None]},
                            {"$ne": ["$$sec", ""]}
                        ]}
                    }}
                }}
            ]
            # Use index hint to force MongoDB to use article_idx (much faster)
            try:
                cursor = self.main.aggregate(
                    pipeline, 
                    allowDiskUse=True, 
                    maxTimeMS=60000,
                    hint={"article": 1}  # Force index usage
                )
            except Exception:
                # Fallback if hint fails (index might not exist yet)
                cursor = self.main.aggregate(pipeline, allowDiskUse=True, maxTimeMS=60000)
            
            for doc in cursor:
                art = (doc.get("article") or "").strip()
                if not art:
                    continue
                sections = doc.get("sections", [])
                self.article_to_sections[art] = {s.strip() for s in sections if s and s.strip()}
            self._sections_by_article_lc = {
                art.lower(): {s.lower() for s in secs}
                for art, secs in self.article_to_sections.items()
            }
        except Exception as e:
            logger.error("[KW] failed building article→sections map: %s", e)
            self.article_to_sections = {}
            self._sections_by_article_lc = {}

        # --- public surface for UI/debug ---
        self.important_terms = list({*(self._articles + self._titles)})
        logger.info("[KW] cache: articles=%d titles=%d", len(self._articles), len(self._titles))

        # number-word map for alternate matches
        self._number_word_map = {
            "first":"1st","second":"2nd","third":"3rd","fourth":"4th","fifth":"5th","sixth":"6th","seventh":"7th","eighth":"8th","ninth":"9th","tenth":"10th",
            "eleventh":"11th","twelfth":"12th","thirteenth":"13th","fourteenth":"14th","fifteenth":"15th","sixteenth":"16th","seventeenth":"17th","eighteenth":"18th","nineteenth":"19th","twentieth":"20th",
            "one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
            "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20",
            "twenty one":"21","twenty two":"22","twenty three":"23","twenty four":"24","twenty five":"25","twenty six":"26",
        }

    # ---------------- public ----------------
    def find_textual(self, query: str) -> List[str]:
        q_fixed = self._fix_ordinal_typos(query or "")
        lower_q = q_fixed.lower()
        matches: List[str] = []

        # If the entire query looks like a bare ordinal/word/number, guess "Nth Amendment"
        guess = self._maybe_amendment_from_bare(lower_q)
        if guess:
            matches.append(guess)

        # Titles/articles by word-boundary match + simple alternates (word↔number)
        for term in self.important_terms:
            t = (term or "").lower()
            if not t:
                continue
            if re.search(rf"\b{re.escape(t)}\b", lower_q):
                if term not in matches:
                    matches.append(term)
                continue
            # Try simple alternates (e.g., "1st" ↔ "first", "1" ↔ "one")
            # Use word boundaries to prevent partial matches (e.g., "1st" matching "11th")
            for word, number in self._number_word_map.items():
                # Check if number appears as a whole word in the term
                number_pattern = rf"\b{re.escape(number)}\b"
                word_pattern = rf"\b{re.escape(word)}\b"
                
                if re.search(number_pattern, t):
                    # Replace number with word (whole word replacement)
                    alt = re.sub(number_pattern, word, t)
                elif re.search(word_pattern, t):
                    # Replace word with number (whole word replacement)
                    alt = re.sub(word_pattern, number, t)
                else:
                    alt = None
                
                if alt and re.search(rf"\b{re.escape(alt)}\b", lower_q):
                    if term not in matches:
                        matches.append(term)
                    break

        # Article + Section narrowing
        art = self._extract_article(lower_q)          # canonical article from dataset, or None
        secs = self._extract_sections(lower_q)        # list[int]
        if art and secs:
            valid = self._sections_by_article_lc.get(art.lower(), set())
            for n in secs:
                sec_label = f"Section {n}"
                # if we know valid sections for the article, require membership; else accept it
                if not valid or sec_label.lower() in valid:
                    for tok in (art, sec_label, f"{art} {sec_label}"):
                        if tok not in matches:
                            matches.append(tok)

        # dedup preserve order
        out, seen = [], set()
        for m in matches:
            if m not in seen:
                seen.add(m); out.append(m)
        logger.debug("[KW] find_textual hits=%s", out)
        return out

    # ---------------- helpers ----------------
    def _fix_ordinal_typos(self, text: str) -> str:
        def _fix(m):
            n = int(m.group(1))
            suf = "th" if 11 <= (n % 100) <= 13 else self._ORD_SUFFIX.get(n % 10, "th")
            return f"{n}{suf}"
        return self._ORD_TYPO_RX.sub(_fix, text or "")

    def _maybe_amendment_from_bare(self, lower_q: str) -> Optional[str]:
        # Only if there's no explicit "amendment" or "article"
        if re.search(r"\b(amendment|article)\b", lower_q):
            return None
        m = re.fullmatch(r"(?:the\s+)?(\d+(?:st|nd|rd|th)?|\w+(?:[-\s]\w+)*)", lower_q.strip())
        if not m:
            return None
        tok = m.group(1).strip()
        if re.fullmatch(r"\d+(?:st|nd|rd|th)?", tok):
            num = int(re.sub(r"(st|nd|rd|th)$", "", tok))
        else:
            num = self._WORD_TO_NUM.get(tok)
        if not num:
            return None
        suf = "th" if 11 <= (num % 100) <= 13 else self._ORD_SUFFIX.get(num % 10, "th")
        target = f"{num}{suf} Amendment"
        return self._title_lc_to_title.get(target.lower())

    @staticmethod
    def _to_roman(num: int) -> str:
        vals = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
                (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
        out = []
        for v, sym in vals:
            while num >= v:
                out.append(sym); num -= v
        return "".join(out)

    def _extract_article(self, lower_q: str) -> Optional[str]:
        m = self._ART_RX.search(lower_q)
        if not m:
            return None
        token = m.group(1).lower().strip()
        word_to_digit = {
            "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,
            "eleventh":11,"twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,"seventeenth":17,
            "eighteenth":18,"nineteenth":19,"twentieth":20
        }
        candidates: List[str] = []
        if token.isdigit():
            n = int(token)
            candidates.extend((f"Article {self._to_roman(n)}", f"Article {n}"))
        elif re.fullmatch(r"[ivxlcdm]+", token):
            candidates.append(f"Article {token.upper()}")
        else:
            n = word_to_digit.get(token)
            if n:
                candidates.extend((f"Article {self._to_roman(n)}", f"Article {n}"))

        for c in candidates:
            hit = self._article_lc_to_article.get(c.lower())
            if hit:
                return hit
        return None

    def _extract_sections(self, lower_q: str) -> List[int]:
        out: List[int] = []
        for m in self._SEC_RX.finditer(lower_q):
            try:
                out.append(int(m.group(1)))
            except Exception:
                pass
        # dedup preserve order
        seen, dedup = set(), []
        for n in out:
            if n not in seen:
                seen.add(n); dedup.append(n)
        return dedup

    def refresh(self) -> None:
        """Rebuild caches from DB (call if the dataset changes)."""
        # MongoManager always sets self.main, so this should always exist
        self.main = self.db.main

        # reload distincts using aggregation pipeline with parallel execution and index hints
        try:
            def load_articles():
                article_pipeline = [
                    {"$match": {"article": {"$exists": True, "$ne": None, "$ne": ""}}},
                    {"$group": {"_id": "$article"}},
                    {"$project": {"article": "$_id"}}
                ]
                # Use index hint to force MongoDB to use article_idx (loads index into memory cache)
                try:
                    return [doc.get("article", "").strip() 
                           for doc in self.main.aggregate(
                               article_pipeline, 
                               allowDiskUse=True, 
                               maxTimeMS=60000,
                               hint={"article": 1}  # Force index usage - keeps index in MongoDB's cache
                           )
                           if doc.get("article")]
                except Exception:
                    # Fallback if hint fails
                    return [doc.get("article", "").strip() 
                           for doc in self.main.aggregate(article_pipeline, allowDiskUse=True, maxTimeMS=60000)
                           if doc.get("article")]
            
            def load_titles():
                title_pipeline = [
                    {"$match": {"title": {"$exists": True, "$ne": None, "$ne": ""}}},
                    {"$group": {"_id": "$title"}},
                    {"$project": {"title": "$_id"}}
                ]
                # Use index hint to force MongoDB to use title_idx (loads index into memory cache)
                try:
                    return [doc.get("title", "").strip() 
                           for doc in self.main.aggregate(
                               title_pipeline, 
                               allowDiskUse=True, 
                               maxTimeMS=60000,
                               hint={"title": 1}  # Force index usage - keeps index in MongoDB's cache
                           )
                           if doc.get("title")]
                except Exception:
                    # Fallback if hint fails
                    return [doc.get("title", "").strip() 
                           for doc in self.main.aggregate(title_pipeline, allowDiskUse=True, maxTimeMS=60000)
                           if doc.get("title")]
            
            # Execute both aggregations in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                article_future = executor.submit(load_articles)
                title_future = executor.submit(load_titles)
                
                arts = article_future.result()
                tits = title_future.result()
            
            self._articles = [a for a in arts if isinstance(a, str) and a.strip() and a.strip() != "Amendment"]
            self._titles   = [t for t in tits if isinstance(t, str) and t.strip()]
            self._title_lc_to_title    = {t.lower(): t for t in self._titles}
            self._article_lc_to_article = {a.lower(): a for a in self._articles}
        except Exception as e:
            logger.error("[KW] refresh: failed to load distinct titles/articles: %s", e)
            self._articles, self._titles = [], []
            self._title_lc_to_title, self._article_lc_to_article = {}, {}

        # reload map using aggregation pipeline with index hint for better performance
        try:
            self.article_to_sections = {}
            # Use aggregation to group by article and collect unique sections
            pipeline = [
                {"$match": {"article": {"$exists": True, "$ne": None, "$ne": ""}}},
                {"$group": {
                    "_id": "$article",
                    "sections": {"$addToSet": "$section"}
                }},
                {"$project": {
                    "article": "$_id",
                    "sections": {"$filter": {
                        "input": "$sections",
                        "as": "sec",
                        "cond": {"$and": [
                            {"$ne": ["$$sec", None]},
                            {"$ne": ["$$sec", ""]}
                        ]}
                    }}
                }}
            ]
            # Use index hint to force MongoDB to use article_idx (loads index into memory cache)
            try:
                cursor = self.main.aggregate(
                    pipeline, 
                    allowDiskUse=True, 
                    maxTimeMS=60000,
                    hint={"article": 1}  # Force index usage - keeps index in MongoDB's cache
                )
            except Exception:
                # Fallback if hint fails
                cursor = self.main.aggregate(pipeline, allowDiskUse=True, maxTimeMS=60000)
            
            for doc in cursor:
                art = (doc.get("article") or "").strip()
                if not art:
                    continue
                sections = doc.get("sections", [])
                self.article_to_sections[art] = {s.strip() for s in sections if s and s.strip()}
            self._sections_by_article_lc = {
                art.lower(): {s.lower() for s in secs}
                for art, secs in self.article_to_sections.items()
            }
        except Exception as e:
            logger.error("[KW] refresh: failed building article→sections map: %s", e)
            self.article_to_sections = {}
            self._sections_by_article_lc = {}

        self.important_terms = list({*(self._articles + self._titles)})
        logger.info("[KW] refresh complete: articles=%d titles=%d", len(self._articles), len(self._titles))
