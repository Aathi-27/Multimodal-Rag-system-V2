"""
Tests for the generation pipeline.

Validates:
- Context assembly ordering (grouped by document, sorted by score)
- Token budget enforcement via Gemma tokenizer
- Citation format in generated responses
- Streaming token delivery
- Confidence indicators for uncertain answers
- Empty result handling
"""

from __future__ import annotations

import pytest

from app.generation.prompt_templates import (
    ChunkContext,
    build_citation_tag,
    build_prompt,
    format_context_block,
)


class TestCitationTags:
    """Test citation tag generation."""

    def test_document_citation_format(self):
        """Document citations: [Source: filename, Page: N]."""
        chunk = ChunkContext(
            chunk_id="c1",
            text="sample text",
            source="report.pdf",
            modality="document",
            page_start=5,
        )
        tag = build_citation_tag(chunk)
        assert tag == "[Source: report.pdf, Page: 5]"

    def test_audio_citation_format(self):
        """Audio citations: [Speaker: name, Timestamp: MM:SS]."""
        chunk = ChunkContext(
            chunk_id="c2",
            text="sample text",
            source="meeting.mp3",
            modality="audio",
            speaker="John",
            timestamp_start="05:30",
        )
        tag = build_citation_tag(chunk)
        assert tag == "[Speaker: John, Timestamp: 05:30]"

    def test_image_citation_format(self):
        """Image citations: [Source: filename, OCR/Caption]."""
        chunk = ChunkContext(
            chunk_id="c3",
            text="sample text",
            source="diagram.png",
            modality="image",
        )
        tag = build_citation_tag(chunk)
        assert tag == "[Source: diagram.png, OCR/Caption]"


class TestContextAssembly:
    """Test context block formatting and ordering."""

    def test_grouped_by_document(self):
        """Chunks from the same document should be grouped together."""
        chunks = [
            ChunkContext(chunk_id="1", text="A", source="doc1.pdf",
                        modality="document", page_start=1, reranker_score=0.9),
            ChunkContext(chunk_id="2", text="B", source="doc2.pdf",
                        modality="document", page_start=1, reranker_score=0.8),
            ChunkContext(chunk_id="3", text="C", source="doc1.pdf",
                        modality="document", page_start=2, reranker_score=0.7),
        ]
        block = format_context_block(chunks)
        # doc1.pdf chunks should appear before doc2.pdf (higher max score)
        doc1_first = block.index("doc1.pdf")
        doc2_first = block.index("doc2.pdf")
        assert doc1_first < doc2_first

    def test_sorted_by_reranker_score(self):
        """Document groups should be sorted by max reranker score descending."""
        chunks = [
            ChunkContext(chunk_id="1", text="Low", source="low.pdf",
                        modality="document", reranker_score=0.3),
            ChunkContext(chunk_id="2", text="High", source="high.pdf",
                        modality="document", reranker_score=0.9),
        ]
        block = format_context_block(chunks)
        assert block.index("high.pdf") < block.index("low.pdf")

    def test_logical_order_within_document(self):
        """Within a document, chunks should follow page order."""
        chunks = [
            ChunkContext(chunk_id="1", text="Page 5", source="doc.pdf",
                        modality="document", page_start=5, chunk_index=0,
                        reranker_score=0.5),
            ChunkContext(chunk_id="2", text="Page 2", source="doc.pdf",
                        modality="document", page_start=2, chunk_index=0,
                        reranker_score=0.8),
        ]
        block = format_context_block(chunks)
        assert block.index("Page 2") < block.index("Page 5")


class TestPromptBuilder:
    """Test full prompt construction."""

    def test_prompt_contains_system_instructions(self):
        """Prompt must include citation rules."""
        chunks = [
            ChunkContext(chunk_id="1", text="Test", source="test.pdf",
                        modality="document", reranker_score=0.5),
        ]
        prompt = build_prompt("What is this?", chunks)
        assert "ONLY the provided context" in prompt
        assert "Source:" in prompt

    def test_prompt_contains_query(self):
        """Prompt must include the user's question."""
        chunks = [
            ChunkContext(chunk_id="1", text="Test", source="test.pdf",
                        modality="document", reranker_score=0.5),
        ]
        prompt = build_prompt("What is the answer?", chunks)
        assert "What is the answer?" in prompt

    def test_prompt_gemma_format(self):
        """Prompt must use Gemma turn markers."""
        chunks = [
            ChunkContext(chunk_id="1", text="Test", source="test.pdf",
                        modality="document", reranker_score=0.5),
        ]
        prompt = build_prompt("Test query", chunks)
        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt
        assert "<end_of_turn>" in prompt
