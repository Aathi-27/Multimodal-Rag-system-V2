"""
Custom Exceptions for the RAG system.

Provides clear, traceable error types for each pipeline stage.
All exceptions include traceable IDs where applicable.
"""

from __future__ import annotations

from typing import Optional


class RAGBaseError(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id
        super().__init__(f"[{trace_id or 'no-trace'}] {message}")


# ── Ingestion Errors ──────────────────────────────────────────────────────────


class FileValidationError(RAGBaseError):
    """Raised when an uploaded file fails validation."""
    pass


class UnsupportedFileTypeError(FileValidationError):
    """Raised for unsupported file formats."""
    pass


class FileSizeLimitError(FileValidationError):
    """Raised when file exceeds the maximum upload size."""
    pass


class CorruptedFileError(FileValidationError):
    """Raised when a file is corrupted or cannot be parsed."""
    pass


class PasswordProtectedError(FileValidationError):
    """Raised when a file is password-protected."""
    pass


# ── Processing Errors ─────────────────────────────────────────────────────────


class NormalizationError(RAGBaseError):
    """Raised when text normalization fails."""
    pass


class ChunkingError(RAGBaseError):
    """Raised when chunking produces invalid results."""
    pass


class TokenLimitExceededError(ChunkingError):
    """Raised when a chunk exceeds the 512-token hard limit."""
    pass


class EmptyDocumentError(RAGBaseError):
    """Raised when a document yields no extractable text."""
    pass


# ── Retrieval Errors ──────────────────────────────────────────────────────────


class VectorStoreError(RAGBaseError):
    """Raised for Qdrant connection or query failures."""
    pass


class BM25IndexError(RAGBaseError):
    """Raised for BM25 index load/query failures."""
    pass


class IndexChecksumError(RAGBaseError):
    """Raised when BM25 index checksum doesn't match Qdrant."""
    pass


class RerankerError(RAGBaseError):
    """Raised when the cross-encoder reranker fails."""
    pass


# ── Generation Errors ─────────────────────────────────────────────────────────


class LLMNotLoadedError(RAGBaseError):
    """Raised when LLM is accessed before loading."""
    pass


class LLMGenerationError(RAGBaseError):
    """Raised when LLM generation fails or times out."""
    pass


class ContextBudgetExceededError(RAGBaseError):
    """Raised when context assembly exceeds the token budget."""
    pass


# ── Worker Errors ─────────────────────────────────────────────────────────────


class WorkerTimeoutError(RAGBaseError):
    """Raised when a worker process exceeds its timeout."""
    pass


class WorkerCrashError(RAGBaseError):
    """Raised when a worker process crashes unexpectedly."""
    pass


class OCRError(RAGBaseError):
    """Raised when OCR processing fails."""
    pass


class TranscriptionError(RAGBaseError):
    """Raised when audio transcription fails."""
    pass


# ── Model Errors ──────────────────────────────────────────────────────────────


class ModelNotFoundError(RAGBaseError):
    """Raised when a required model file is not found on disk."""
    pass


class ModelLoadError(RAGBaseError):
    """Raised when a model fails to load into memory."""
    pass
