# services/rag_dependencies/llm_verifier.py
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Tuple
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.rag.rag_dependencies.ai_service import LLM

logger = logging.getLogger(__name__)
_SCORE_LINE_RX = re.compile(r'(?im)^\s*score\s*:\s*(-?\d{1,2})\s*$')
_JSON_SCORE_KEYS = ("score", "rating", "relevance")


class LLMVerifier:
    """
    Turns LLM judgments into stable multipliers and adjusted scores.
    Deterministic, sequential verification (no threading).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds: Dict[str, float] = config.get("thresholds", {})
        self.openAI = LLM(config=self.config)

    # ---------- extraction helpers ----------
    @staticmethod
    def _extract_score_flag_reason(llm_result) -> int:
        s = unicodedata.normalize("NFKC", str(llm_result or "")).strip()
        logger.debug("[LLM][PARSE] raw_tail=%r", s[-240:])

        # 1) JSON (object OR scalar)
        try:
            obj = json.loads(s)
            # scalar number: "2" -> 2
            if isinstance(obj, (int, float)):
                v = int(obj)
                return max(0, min(9, v))
            # object with key
            if isinstance(obj, dict):
                for key in _JSON_SCORE_KEYS:
                    if key in obj:
                        v = int(float(obj[key]))
                        return max(0, min(9, v))
        except Exception:
            pass

        # 2) Bare numeric string:  "2"
        if re.fullmatch(r"-?\d{1,2}", s):
            v = int(s)
            return max(0, min(9, v))

        # 3) Anchored "Score: N" line
        m = _SCORE_LINE_RX.search(s)
        if m:
            v = int(m.group(1))
            return max(0, min(9, v))

        raise ValueError("No parseable score (JSON scalar/object, numeric string, or 'Score:' line) found.")

    @staticmethod
    def _multiplier(score, *, min_mult=0.50, max_mult=1.50):
        """
        Linear map on 0..9:
          0 -> 0.80x
          9 -> 1.20x
        """
        s = max(0.0, min(9.0, float(score)))
        t = s / 9.0
        return min_mult + (max_mult - min_mult) * t
    def _payload(self, item: dict, item_type: str) -> dict:
        """
        Build an LLM verification payload that is robust to schema differences.
        - If doc_type is 'case' (or item_type says 'case'), include title + summary.
        - Otherwise include title + text-ish content when available.
        """
        # Resolve doc_type defensively
        doc_type = str(item.get("doc_type") or item.get("type") or item_type or "").lower()

        # Resolve a human label/title
        title = (
            item.get("title")
            or item.get("case")         # legacy/case-style key
            or item.get("article")      # constitutional docs
            or item.get("section")
            or "(untitled)"
        )

        # Prefer concise content; fall back to longer body if needed
        summary_or_text = (
            item.get("summary")
            or item.get("text")
            or item.get("body")
            or ""
        )

        if doc_type == "case":
            # ✅ What you asked for: title + summary for cases
            return {"title": title, "summary": summary_or_text}

        # Non-case: still give the verifier some context beyond just a title
        return {"title": title, "text": summary_or_text}

    def verify_one(self, query_text: str, item: dict, base_score: float, item_type: str) -> Tuple[dict, float]:
        adj = base_score
        try:
            payload = self._payload(item, item_type)
            # Derive label from resolved keys (matches _payload logic)
            label = payload.get("title") or item.get("case") or "(untitled)"
            logger.info(f"payload:{payload}")
            logger.debug("[LLM-%s] START label=%r base_score=%.4f", (item_type or "doc").upper(), label, base_score)

            raw = self.openAI.llm_verification(query_text, payload)
            if not raw:
                logger.debug("[LLM-%s] EMPTY raw response; keep base_score=%.4f", (item_type or "doc").upper(), adj)
                return item, adj

            raw_str = str(raw)
            logger.debug("[LLM-%s] RAW_LEN=%d RAW_TAIL=%r", (item_type or "doc").upper(), len(raw_str), raw_str[-240:])

            score = self._extract_score_flag_reason(raw)
            mult = self._multiplier(score)
            adj = min(base_score * mult, 1.0)

            logger.info(
                "[LLM-%s] %s parsed_score=%d → x%.3f ⇒ %.4f (base=%.4f)",
                (item_type or "doc").upper(), label, score, mult, adj, base_score
            )
        except Exception as e:
            try:
                tail = str(raw)[-240:]  # type: ignore[name-defined]
            except Exception:
                tail = "<unavailable>"
            title = (item.get("title") or item.get("case") or "(unknown)")
            logger.warning(
                "[LLM-%s] verify failed for %r: %s; keep %.4f; RAW_TAIL=%r",
                (item_type or "doc").upper(), title, e, adj, tail
            )
        return item, adj
    # ---------- sequential (no workers) ----------
    def verify_many(
        self,
        query_text: str,
        items_with_scores: List[Tuple[dict, float]],
        item_type: str,
    ) -> List[Tuple[dict, float]]:
        """
        Sequential verification (deterministic, thread-safe).
        Returns [(item, adjusted_score)] in the SAME order as input.
        """
        out: List[Tuple[dict, float]] = []
        if not items_with_scores:
            return out

        logger.info("[LLM-%s] VERIFY sequential | candidates=%d",
                    item_type.upper(), len(items_with_scores))

        for item, score in items_with_scores:
            try:
                res = self.verify_one(query_text, item, score, item_type)
                out.append(res)
            except Exception as e:
                title = (item.get("title") or item.get("case") or "(unknown)")
                logger.warning(
                    "[LLM-%s] verify_one failed for %r: %s; keeping base=%.4f",
                    item_type.upper(), title, e, score
                )
                out.append((item, score))
        return out
    
    # ---------- parallel (with workers) ----------
    def verify_many_parallel(
        self,
        query_text: str,
        items_with_scores: List[Tuple[dict, float]],
        item_type: str,
        max_workers: int = 5
    ) -> List[Tuple[dict, float]]:
        """
        Parallel verification using ThreadPoolExecutor for faster processing.
        Returns [(item, adjusted_score)] in the SAME order as input.
        
        Args:
            query_text: User query
            items_with_scores: List of (doc, score) tuples to verify
            item_type: Type of items (e.g., "DOC", "CASE")
            max_workers: Maximum number of parallel threads (default: 5)
        
        Returns:
            List of (doc, adjusted_score) tuples in original order
        """
        if not items_with_scores:
            return []
        
        logger.info("[LLM-%s] VERIFY parallel | candidates=%d | workers=%d",
                   item_type.upper(), len(items_with_scores), max_workers)
        
        # Create a list to store results with their original index
        results = [None] * len(items_with_scores)
        
        def verify_with_index(idx: int, item: dict, score: float):
            """Wrapper to verify a single item and return with its index"""
            try:
                result = self.verify_one(query_text, item, score, item_type)
                return (idx, result)
            except Exception as e:
                title = (item.get("title") or item.get("case") or "(unknown)")
                logger.warning(
                    "[LLM-%s] verify_one failed for %r: %s; keeping base=%.4f",
                    item_type.upper(), title, e, score
                )
                return (idx, (item, score))
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(verify_with_index, idx, item, score): idx
                for idx, (item, score) in enumerate(items_with_scores)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = future_to_idx[future]
                    logger.error("[LLM-%s] Worker failed for index %d: %s", 
                               item_type.upper(), idx, e)
                    # Fall back to original score
                    results[idx] = items_with_scores[idx]
        
        logger.info("[LLM-%s] Parallel verification complete", item_type.upper())
        return results
