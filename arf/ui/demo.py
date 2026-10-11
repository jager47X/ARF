"""A self-contained demo pipeline — no API keys, no database, no model files.

Everything a real deployment would plug in is replaced by a deterministic
stand-in: hashed bag-of-words "embeddings", a dict vector store, a lexical
keyword search, a synonym-based query expander, a hand-written relevance
"MLP", a rule-based "LLM" that answers ``Score: N`` and writes summaries, and
a dict cache.  It exists so the UI can be explored end to end::

    python -m arf.ui --demo
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arf.document import DocumentConfig
from arf.features import FEATURE_NAMES
from arf.pipeline import Pipeline
from arf.triage import Triage

DOCS: List[Dict[str, Any]] = [
    {
        "_id": "us-const-1a",
        "title": "First Amendment",
        "article": "Amendment I",
        "text": "Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise "
        "thereof; or abridging the freedom of speech, or of the press; or the right of the people peaceably to "
        "assemble, and to petition the Government for a redress of grievances.",
    },
    {
        "_id": "us-const-2a",
        "title": "Second Amendment",
        "article": "Amendment II",
        "text": "A well regulated Militia, being necessary to the security of a free State, the right of the people "
        "to keep and bear Arms, shall not be infringed.",
    },
    {
        "_id": "us-const-3a",
        "title": "Third Amendment",
        "article": "Amendment III",
        "text": "No Soldier shall, in time of peace be quartered in any house, without the consent of the Owner, "
        "nor in time of war, but in a manner to be prescribed by law.",
    },
    {
        "_id": "us-const-4a",
        "title": "Fourth Amendment",
        "article": "Amendment IV",
        "text": "The right of the people to be secure in their persons, houses, papers, and effects, against "
        "unreasonable searches and seizures, shall not be violated, and no Warrants shall issue, but upon "
        "probable cause, supported by Oath or affirmation.",
    },
    {
        "_id": "us-const-5a",
        "title": "Fifth Amendment",
        "article": "Amendment V",
        "text": "No person shall be held to answer for a capital crime unless on indictment of a Grand Jury; nor "
        "be twice put in jeopardy; nor be compelled in any criminal case to be a witness against himself, nor be "
        "deprived of life, liberty, or property, without due process of law; nor shall private property be taken "
        "for public use, without just compensation.",
    },
    {
        "_id": "us-const-6a",
        "title": "Sixth Amendment",
        "article": "Amendment VI",
        "text": "In all criminal prosecutions, the accused shall enjoy the right to a speedy and public trial, by an "
        "impartial jury, to be informed of the nature of the accusation, to be confronted with the witnesses "
        "against him, and to have the Assistance of Counsel for his defence.",
    },
    {
        "_id": "us-const-8a",
        "title": "Eighth Amendment",
        "article": "Amendment VIII",
        "text": "Excessive bail shall not be required, nor excessive fines imposed, nor cruel and unusual "
        "punishments inflicted.",
    },
    {
        "_id": "us-const-13a",
        "title": "Thirteenth Amendment",
        "article": "Amendment XIII",
        "section": "1",
        "text": "Neither slavery nor involuntary servitude, except as a punishment for crime whereof the party "
        "shall have been duly convicted, shall exist within the United States.",
    },
    {
        "_id": "us-const-14a",
        "title": "Fourteenth Amendment",
        "article": "Amendment XIV",
        "section": "1",
        "text": "All persons born or naturalized in the United States are citizens of the United States. No State "
        "shall deprive any person of life, liberty, or property, without due process of law; nor deny to any "
        "person within its jurisdiction the equal protection of the laws.",
    },
    {
        "_id": "us-const-15a",
        "title": "Fifteenth Amendment",
        "article": "Amendment XV",
        "section": "1",
        "text": "The right of citizens of the United States to vote shall not be denied or abridged by the United "
        "States or by any State on account of race, color, or previous condition of servitude.",
    },
    {
        "_id": "us-const-16a",
        "title": "Sixteenth Amendment",
        "article": "Amendment XVI",
        "text": "The Congress shall have power to lay and collect taxes on incomes, from whatever source derived, "
        "without apportionment among the several States.",
    },
    {
        "_id": "us-const-19a",
        "title": "Nineteenth Amendment",
        "article": "Amendment XIX",
        "text": "The right of citizens of the United States to vote shall not be denied or abridged by the United "
        "States or by any State on account of sex.",
    },
    {
        "_id": "us-const-a1-s8",
        "title": "Article I, Section 8",
        "article": "Article I",
        "section": "8",
        "text": "The Congress shall have Power To lay and collect Taxes, Duties, Imposts and Excises; To borrow "
        "Money; To regulate Commerce with foreign Nations; To declare War; To raise and support Armies; To "
        "provide and maintain a Navy; To make all Laws which shall be necessary and proper.",
    },
    {
        "_id": "us-const-a2-s1",
        "title": "Article II, Section 1",
        "article": "Article II",
        "section": "1",
        "text": "The executive Power shall be vested in a President of the United States of America. He shall hold "
        "his Office during the Term of four Years, together with the Vice President, chosen for the same Term.",
    },
    {
        "_id": "us-const-a3-s1",
        "title": "Article III, Section 1",
        "article": "Article III",
        "section": "1",
        "text": "The judicial Power of the United States shall be vested in one supreme Court, and in such inferior "
        "Courts as the Congress may from time to time ordain and establish. The Judges shall hold their Offices "
        "during good Behaviour.",
    },
]

SYNONYMS: Dict[str, Sequence[str]] = {
    "speech": ("free speech press religion", "freedom of expression assembly petition"),
    "guns": ("right to keep and bear arms militia",),
    "arms": ("militia bear arms",),
    "search": ("unreasonable searches and seizures warrants probable cause",),
    "warrant": ("searches seizures probable cause oath",),
    "vote": ("right of citizens to vote denied or abridged", "suffrage race color sex"),
    "voting": ("right of citizens to vote denied or abridged",),
    "tax": ("lay and collect taxes on incomes",),
    "taxes": ("collect taxes duties imposts excises",),
    "trial": ("speedy public trial impartial jury counsel", "criminal prosecutions accused witnesses"),
    "jury": ("grand jury indictment trial", "impartial jury speedy public trial"),
    "president": ("executive power vested term of four years",),
    "courts": ("judicial power supreme court judges",),
    "slavery": ("involuntary servitude punishment for crime",),
    "soldiers": ("quartered in any house without consent",),
    "bail": ("excessive bail fines cruel and unusual punishments",),
    "citizens": ("persons born or naturalized equal protection",),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "of",
    "a",
    "an",
    "and",
    "or",
    "to",
    "in",
    "is",
    "what",
    "does",
    "do",
    "about",
    "say",
    "for",
    "on",
    "be",
    "by",
    "who",
    "which",
    "how",
    "when",
    "where",
    "that",
    "this",
    "are",
    "was",
    "it",
    "its",
}
EMBED_DIM = 256


def _stem(token: str) -> str:
    """A deliberately tiny stemmer so "declares" meets "declare", "taxes" meets "tax", "quartered" meets "quartering"."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "ches", "shes", "zes")):
        return token[:-2]
    if token.endswith("ss"):
        return token
    if token.endswith("ing") and len(token) - 3 >= 4:
        return token[:-3]
    if token.endswith("ed") and len(token) - 2 >= 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _tokens(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def hash_embed(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Deterministic bag-of-words embedding: tokens (and bigrams) hashed into *dim* buckets, L2-normalised."""
    vec = [0.0] * dim
    toks = _tokens(text)
    for unit in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]:
        digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=4).digest()
        vec[int.from_bytes(digest, "big") % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _overlap(query: str, doc: Dict[str, Any]) -> float:
    q = set(_tokens(query))
    if not q:
        return 0.0
    d = set(_tokens(f"{doc.get('title', '')} {doc.get('text', '')}"))
    return len(q & d) / len(q)


class DemoStore:
    """In-memory vector store + lexical index + cache, all plain dicts."""

    def __init__(self, docs: Sequence[Dict[str, Any]] = DOCS):
        self.docs: List[Dict[str, Any]] = [dict(d, embedding=hash_embed(f"{d['title']} {d['text']}")) for d in docs]
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.llm_calls = 0

    # -- slots ------------------------------------------------------------

    def search(self, embedding: Sequence[float], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        scored = [(doc, float(sum(a * b for a, b in zip(embedding, doc["embedding"])))) for doc in self.docs]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def keyword(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        phrase = " ".join(_tokens(query))
        if not phrase:
            return []
        hits = []
        for doc in self.docs:
            haystack = " ".join(_tokens(f"{doc['title']} {doc['text']}"))
            if phrase in haystack:
                hits.append((doc, 10.0))  # BM25-ish magnitude, deliberately not cosine-like
        return hits

    def expand(self, query: str) -> List[str]:
        variants: List[str] = []
        for tok in _tokens(query):
            variants.extend(SYNONYMS.get(tok, ()))
        return variants

    def llm(self, query: str, doc: Dict[str, Any]) -> str:
        self.llm_calls += 1
        overlap = _overlap(query, doc)
        score = min(9, int(round(9 * math.sqrt(overlap))))
        return f"Reasoning: {int(overlap * 100)}% of the query terms appear in the text.\nScore: {score}"

    def summarize(self, query: str, doc: Any, context: Any) -> str:
        shared = sorted(set(_tokens(query)) & set(_tokens(f"{doc.title} {doc.content}")))
        if shared:
            return f"{doc.title} matches on: {', '.join(shared)}."
        return f"{doc.title} was retrieved by semantic similarity."

    def cache_lookup(self, key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(key)

    def cache_store(self, key: str, payload: Dict[str, Any]) -> None:
        self.cache.setdefault(key, {})["results"] = payload

    def link(self, source: str, target: str) -> None:
        self.cache.setdefault(source, {})["next"] = target

    def clear_cache(self) -> None:
        self.cache.clear()

    def resolve(self, parent_id: str) -> Optional[Dict[str, Any]]:
        return None


def demo_predict_fn(vectors: List[List[float]]) -> List[float]:
    """A hand-written stand-in for a trained MLP: a logistic blend of three features.

    Weighted towards ``semantic_score`` because the library's feature
    tokenizer does not stem, so ``query_term_coverage`` under-counts
    inflected matches ("quartering" vs "quartered") in this toy corpus.
    """
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    probs = []
    for v in vectors:
        z = (
            8.0 * v[idx["semantic_score"]]
            + 2.0 * v[idx["query_term_coverage"]]
            + 1.5 * v[idx["title_similarity"]]
            - 2.0
        )
        probs.append(1.0 / (1.0 + math.exp(-z)))
    return probs


def build_demo_pipeline(store: Optional[DemoStore] = None, **overrides: Any) -> Pipeline:
    """A fully wired :class:`~arf.pipeline.Pipeline` over :data:`DOCS`.

    Hashed bag-of-words cosine scores are low in absolute terms, so the
    triage thresholds are set for that range (see the README on calibrating
    thresholds to your embedding model).
    """
    store = store or DemoStore()
    settings: Dict[str, Any] = dict(
        doc_config=DocumentConfig(hierarchy=["article", "section", "title"]),
        triage=Triage(min_score=0.06, verify_threshold=0.12, accept_threshold=0.42, gap=0.40),
        search_fn=store.search,
        embed_fn=hash_embed,
        predict_fn=demo_predict_fn,
        predict_zones=(0.30, 0.7),
        llm_fn=store.llm,
        llm_pass_score=4,
        keyword_fn=store.keyword,
        expand_fn=store.expand,
        max_expansions=2,
        summarize_fn=store.summarize,
        cache_lookup=store.cache_lookup,
        cache_store=store.cache_store,
        link_fn=store.link,
        cache_ttl=3600,
        max_rephrase=0,
    )
    settings.update(overrides)
    pipeline = Pipeline(**settings)
    pipeline.demo_store = store  # type: ignore[attr-defined]
    return pipeline
