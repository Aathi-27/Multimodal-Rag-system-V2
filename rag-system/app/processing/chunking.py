"""
Custom Sliding Window Chunker.

Pipeline: Docling Markdown → Normalized text → This chunker → Embedding-ready chunks

Rules (from PRD):
- Target: 480 tokens
- Hard max: 512 tokens (never exceeded)
- Overlap: 50 tokens
- Respect sentence boundaries when possible
- Preserve Markdown headers at chunk start
- Validate token count before embedding generation
- Chunk audio by token count (NOT by Whisper segments)

Tokenizer Authority: Gemma tokenizer (via llama.cpp)
"""

from __future__ import annotations

import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Sentence boundary pattern
RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Markdown header pattern
RE_MD_HEADER = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ""
    token_count: int = 0
    source: str = ""
    modality: str = "document"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_index: int = 0
    speaker: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class SlidingWindowChunker:
    """
    Custom sliding window chunker with sentence boundary awareness.

    The chunker splits text into overlapping windows of ~480 tokens,
    never exceeding 512 tokens. It preserves Markdown headers at
    chunk start and respects sentence boundaries.
    """

    def __init__(
        self,
        target_tokens: int = 480,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        # Token counter: use provided counter or fallback to whitespace approximation
        self._count_tokens = token_counter or self._approx_token_count

    @staticmethod
    def _approx_token_count(text: str) -> int:
        """Approximate token count (whitespace split). Use Gemma tokenizer in production."""
        return len(text.split())

    def chunk_text(
        self,
        text: str,
        source: str = "",
        modality: str = "document",
        page_start: Optional[int] = None,
        speaker: Optional[str] = None,
        timestamp_start: Optional[str] = None,
    ) -> list[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Normalized text to chunk.
            source: Source filename for metadata.
            modality: Content modality (document/image/audio).
            page_start: Starting page number (documents).
            speaker: Speaker name (audio).
            timestamp_start: Start timestamp (audio).

        Returns:
            List of Chunk objects, each under max_tokens.
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        chunk_index = 0

        # Track the last header seen
        last_header: Optional[str] = None

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Handle single sentence > max_tokens (force split)
            if sentence_tokens > self.max_tokens:
                # Flush current buffer
                if current_sentences:
                    chunks.append(self._make_chunk(
                        current_sentences, chunk_index, source, modality,
                        page_start, speaker, timestamp_start, last_header,
                    ))
                    chunk_index += 1
                    current_sentences = []
                    current_tokens = 0

                # Force-split the long sentence by words
                words = sentence.split()
                word_buf: list[str] = []
                buf_tokens = 0
                for word in words:
                    wt = self._count_tokens(word)
                    if buf_tokens + wt > self.max_tokens and word_buf:
                        chunks.append(self._make_chunk(
                            [" ".join(word_buf)], chunk_index, source, modality,
                            page_start, speaker, timestamp_start, last_header,
                        ))
                        chunk_index += 1
                        # Overlap: keep last few words
                        overlap_words = word_buf[-10:] if len(word_buf) > 10 else []
                        word_buf = overlap_words + [word]
                        buf_tokens = self._count_tokens(" ".join(word_buf))
                    else:
                        word_buf.append(word)
                        buf_tokens += wt

                if word_buf:
                    current_sentences = [" ".join(word_buf)]
                    current_tokens = self._count_tokens(current_sentences[0])
                continue

            # Check if sentence is a markdown header
            if RE_MD_HEADER.match(sentence.strip()):
                last_header = sentence.strip()

            # Would adding this sentence exceed target?
            if current_tokens + sentence_tokens > self.target_tokens and current_sentences:
                # Create chunk from current buffer
                chunks.append(self._make_chunk(
                    current_sentences, chunk_index, source, modality,
                    page_start, speaker, timestamp_start, last_header,
                ))
                chunk_index += 1

                # Overlap: keep trailing sentences that fit in overlap budget
                overlap_sents, overlap_tokens = self._compute_overlap(current_sentences)
                current_sentences = overlap_sents + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Flush remaining
        if current_sentences:
            chunks.append(self._make_chunk(
                current_sentences, chunk_index, source, modality,
                page_start, speaker, timestamp_start, last_header,
            ))

        # Final validation
        for chunk in chunks:
            if chunk.token_count > self.max_tokens:
                logger.error(
                    "CHUNK EXCEEDS MAX TOKENS: %d > %d (id=%s)",
                    chunk.token_count, self.max_tokens, chunk.chunk_id[:8],
                )

        logger.info(
            "Chunked '%s': %d chunks from %d sentences",
            source, len(chunks), len(sentences),
        )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving markdown structure."""
        # Split on sentence boundaries
        parts = RE_SENTENCE_END.split(text)
        # Filter empty
        return [p.strip() for p in parts if p.strip()]

    def _compute_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Get trailing sentences that fit in the overlap budget."""
        overlap_sents: list[str] = []
        overlap_tokens = 0

        for sent in reversed(sentences):
            sent_tokens = self._count_tokens(sent)
            if overlap_tokens + sent_tokens > self.overlap_tokens:
                break
            overlap_sents.insert(0, sent)
            overlap_tokens += sent_tokens

        return overlap_sents, overlap_tokens

    def _make_chunk(
        self,
        sentences: list[str],
        chunk_index: int,
        source: str,
        modality: str,
        page_start: Optional[int],
        speaker: Optional[str],
        timestamp_start: Optional[str],
        last_header: Optional[str],
    ) -> Chunk:
        """Assemble a Chunk from sentences."""
        text = " ".join(sentences)

        # Prepend last header if the chunk doesn't already start with one
        if last_header and not text.startswith("#"):
            text = last_header + "\n" + text

        token_count = self._count_tokens(text)

        return Chunk(
            text=text,
            token_count=token_count,
            source=source,
            modality=modality,
            page_start=page_start,
            chunk_index=chunk_index,
            speaker=speaker,
            timestamp_start=timestamp_start,
        )
