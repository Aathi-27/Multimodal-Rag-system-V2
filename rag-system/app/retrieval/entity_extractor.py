"""
Entity Extractor - Lightweight keyword / pattern entity extraction.

Extracts named entities (proper nouns, technical terms, acronyms) from
user queries to enable entity-aware chunk injection into the RRF pipeline.

Design:
- No external NLP library required (runs on regex + heuristics)
- Extracts: capitalized multi-word phrases, ALL-CAPS acronyms,
  quoted terms, hyphenated compounds
- Returns a list of entity strings for downstream chunk lookup
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Common English stop-words that should not be treated as entities
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "you", "he", "she", "it", "they", "them",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "and", "or", "but", "nor", "not", "no", "so", "if", "then", "than",
    "to", "of", "in", "for", "on", "at", "by", "with", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "very",
})

# Regex patterns
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_QUOTED_RE = re.compile(r'"([^"]{2,60})"')
_CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_HYPHENATED_RE = re.compile(r"\b[A-Za-z]+-[A-Za-z]+(?:-[A-Za-z]+)*\b")


def extract_entities(query: str) -> list[str]:
    """
    Extract entity-like terms from a query string.

    Returns deduplicated list of entity strings (preserving order).
    """
    entities: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        normalized = term.strip()
        low = normalized.lower()
        if low not in seen and low not in _STOP and len(normalized) > 1:
            seen.add(low)
            entities.append(normalized)

    # 1. Quoted terms (highest priority — explicit user intent)
    for match in _QUOTED_RE.finditer(query):
        _add(match.group(1))

    # 2. ALL-CAPS acronyms (e.g. RAG, BM25, HNSW, GPU)
    for match in _ACRONYM_RE.finditer(query):
        _add(match.group())

    # 3. Capitalized multi-word phrases (e.g. "Reciprocal Rank Fusion")
    for match in _CAPITALIZED_PHRASE_RE.finditer(query):
        _add(match.group())

    # 4. Hyphenated compounds (e.g. "cross-encoder", "bge-reranker-base")
    for match in _HYPHENATED_RE.finditer(query):
        _add(match.group())

    logger.debug("Extracted entities from query: %s → %s", query[:80], entities)
    return entities


def find_entity_chunks(
    entities: list[str],
    bm25_store,
    per_entity_limit: int = 10,
    global_limit: int = 20,
) -> list[dict]:
    """
    Look up chunks that mention the extracted entities.

    Uses BM25 as the lookup engine (fast keyword match).

    Args:
        entities: Extracted entity strings.
        bm25_store: BM25Store instance.
        per_entity_limit: Max chunks per entity query.
        global_limit: Max total injected chunks.

    Returns:
        List of unique chunk dicts tagged with origin="entity".
    """
    if not entities:
        return []

    seen_ids: set[str] = set()
    injected: list[str] = []
    results: list[dict] = []

    for entity in entities:
        hits = bm25_store.search(query=entity, top_k=per_entity_limit)
        for hit in hits:
            cid = hit["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                hit["origin"] = "entity"
                hit["entity_match"] = entity
                results.append(hit)
                injected.append(entity)
            if len(results) >= global_limit:
                break
        if len(results) >= global_limit:
            break

    logger.info(
        "Entity injection: %d entities → %d unique chunks",
        len(entities), len(results),
    )
    return results
