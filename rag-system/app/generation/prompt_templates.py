"""
Prompt Templates - Citation-enforced prompt construction for Qwen2.5 (ChatML).

Handles:
- Context assembly with metadata (source, page, speaker, timestamp)
- Citation format enforcement
- Token budget management
- Dynamic rigor mode: analytical depth for technical queries, concise for simple ones
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class ChunkContext:
    """A retrieved chunk with its metadata for prompt assembly."""
    chunk_id: str
    text: str
    source: str
    modality: str  # "document", "image", "audio"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_index: Optional[int] = None
    speaker: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    reranker_score: float = 0.0


# ── High-rigor prompt: for analytical / technical / "why" queries ────────────

SYSTEM_PROMPT_RIGOROUS = """\
You are a precise technical analyst. Answer using ONLY the provided context.
RULES:
1. Cite every factual claim: [Source: filename, Page: N] or [Speaker: name, Timestamp: MM:SS].
2. Reference exact numbers, thresholds, formulas, and constraints from context.
3. Structure your answer clearly: use headings or numbered points for complex topics.
4. Explain the mechanism, then constraints, then practical implications.
5. If context is insufficient, explicitly state what information is missing. Never fabricate.
6. Provide a thorough, complete answer — do not truncate or omit relevant details.
7. ALWAYS end your response with a brief **Summary:** (1-2 sentences) that captures the key takeaway for the reader.
"""

# ── Standard prompt: factual, for straightforward queries ────────────────────

SYSTEM_PROMPT_CONCISE = """\
You are a knowledgeable assistant. Answer using ONLY the provided context.
RULES:
1. Cite sources: [Source: filename, Page: N] or [Speaker: name, Timestamp: MM:SS].
2. Be direct and well-structured. Use bullet points or short paragraphs.
3. If context is insufficient, say so. Never make up information.
4. Give a complete answer covering all relevant points from the context.
5. ALWAYS end your response with a brief **Summary:** (1-2 sentences) that captures the key takeaway for the reader.\
"""

# ── Rigor detection ──────────────────────────────────────────────────────────

_RIGOR_PATTERN = re.compile(
    r"\b(?:explain|why|how\s+does|how\s+is|ensure|ensures|prevent|prevents|"
    r"deterministic|formula|threshold|mechanism|invariant|guarantee|"
    r"mathematically|constraint|architecture|design\s+decision|trade-?off|"
    r"compare|contrast|analyze|justify|reasoning)\b",
    re.IGNORECASE,
)


def _needs_rigor(query: str) -> bool:
    """Detect whether a query demands analytical depth."""
    return bool(_RIGOR_PATTERN.search(query))


# ── Cached system prompt token counts ────────────────────────────────────────
_system_prompt_token_cache: dict[int, int] = {}


def _get_prompt_overhead(system_prompt: str, query: str, token_counter) -> int:
    """Get the non-context token overhead (system prompt + ChatML tags + query).

    The system prompt portion is cached by identity since it's one of two constants.
    """
    sp_id = id(system_prompt)
    if sp_id not in _system_prompt_token_cache:
        # Count tokens for the static wrapper: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\nCONTEXT:\n
        static_part = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nCONTEXT:\n"
        _system_prompt_token_cache[sp_id] = token_counter(static_part)
    # Add query-specific tokens: \n\nQUESTION: {query}<|im_end|>\n<|im_start|>assistant\n
    query_part = f"\n\nQUESTION: {query}<|im_end|>\n<|im_start|>assistant\n"
    return _system_prompt_token_cache[sp_id] + token_counter(query_part)


def build_citation_tag(chunk: ChunkContext) -> str:
    """Build a citation tag string from chunk metadata."""
    if chunk.modality == "audio" and chunk.speaker:
        ts = chunk.timestamp_start or "00:00"
        return f"[Speaker: {chunk.speaker}, Timestamp: {ts}]"
    elif chunk.modality == "image":
        return f"[Source: {chunk.source}, OCR/Caption]"
    else:
        page = chunk.page_start if chunk.page_start else "N/A"
        return f"[Source: {chunk.source}, Page: {page}]"


def format_context_block(chunks: list[ChunkContext]) -> str:
    """
    Format retrieved chunks into a structured context block.

    Ordering:
    1. Group by document (source)
    2. Sort by reranker score descending
    3. Within same document, preserve logical order (page_start, chunk_index)
    """
    # Group by source
    grouped: dict[str, list[ChunkContext]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source, []).append(chunk)

    # Sort groups by best reranker score in each group (descending)
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: max(c.reranker_score for c in item[1]),
        reverse=True,
    )

    context_parts: list[str] = []
    for source, source_chunks in sorted_groups:
        # Within same document, sort by page then chunk_index
        source_chunks.sort(
            key=lambda c: (c.page_start or 0, c.chunk_index or 0)
        )
        for chunk in source_chunks:
            citation = build_citation_tag(chunk)
            context_parts.append(
                f"--- {citation} ---\n{chunk.text}\n"
            )

    return "\n".join(context_parts)


def build_prompt(
    query: str,
    chunks: list[ChunkContext],
    max_context_tokens: int = 4096,
    token_counter=None,
) -> str:
    """
    Build the full prompt for Qwen2.5 (ChatML) with context and citation instructions.

    Automatically selects high-rigor or concise system prompt based on query intent.

    Args:
        query: User's question
        chunks: Retrieved and reranked chunks
        max_context_tokens: Token budget for context section
        token_counter: Callable that counts tokens in a string

    Returns:
        Complete prompt string in ChatML format
    """
    # Dynamic rigor selection
    system_prompt = SYSTEM_PROMPT_RIGOROUS if _needs_rigor(query) else SYSTEM_PROMPT_CONCISE

    context_block = format_context_block(chunks)

    # Trim context to token budget if counter is available
    if token_counter is not None:
        # Cache the system prompt overhead token count (static per prompt variant)
        prompt_overhead = _get_prompt_overhead(system_prompt, query, token_counter)
        effective_budget = max_context_tokens - prompt_overhead

        lines = context_block.split("\n")
        trimmed_lines: list[str] = []
        current_tokens = 0
        for line in lines:
            line_tokens = token_counter(line)
            if current_tokens + line_tokens > effective_budget:
                break
            trimmed_lines.append(line)
            current_tokens += line_tokens
        context_block = "\n".join(trimmed_lines)

    # Qwen2.5 ChatML format
    prompt = (
        f"<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"Remember: End your answer with a **Summary:** in 1-2 sentences.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt
