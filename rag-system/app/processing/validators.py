"""
Validators - Token limit enforcement and chunk validation.

Hard rules:
- No chunk may exceed 512 tokens
- All chunks must have required metadata fields
- Token count must be validated BEFORE embedding generation
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from app.processing.chunking import Chunk
from app.utils.exceptions import TokenLimitExceededError

logger = logging.getLogger(__name__)

# Required metadata fields per modality
REQUIRED_FIELDS = {
    "document": ["source", "modality", "page_start"],
    "image": ["source", "modality"],
    "audio": ["source", "modality", "speaker"],
}


def validate_token_count(
    chunk: Chunk,
    max_tokens: int = 512,
    token_counter: Optional[Callable[[str], int]] = None,
) -> bool:
    """
    Validate that a chunk does not exceed the token limit.

    Args:
        chunk: The chunk to validate.
        max_tokens: Maximum allowed tokens (hard limit).
        token_counter: Function to count tokens. If None, uses chunk.token_count.

    Returns:
        True if valid.

    Raises:
        TokenLimitExceededError: If chunk exceeds limit.
    """
    count = token_counter(chunk.text) if token_counter else chunk.token_count

    if count > max_tokens:
        raise TokenLimitExceededError(
            f"Chunk {chunk.chunk_id[:8]} has {count} tokens "
            f"(max: {max_tokens}). Source: {chunk.source}"
        )
    return True


def validate_metadata(chunk: Chunk) -> list[str]:
    """
    Validate that a chunk has all required metadata fields.

    Returns:
        List of missing field names (empty if all present).
    """
    required = REQUIRED_FIELDS.get(chunk.modality, ["source", "modality"])
    missing = []

    for field_name in required:
        value = getattr(chunk, field_name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)

    if missing:
        logger.warning(
            "Chunk %s missing metadata: %s",
            chunk.chunk_id[:8],
            missing,
        )

    return missing


def validate_chunks(
    chunks: list[Chunk],
    max_tokens: int = 512,
    token_counter: Optional[Callable[[str], int]] = None,
) -> tuple[list[Chunk], list[dict]]:
    """
    Validate a batch of chunks. Returns valid chunks and error reports.

    Args:
        chunks: List of chunks to validate.
        max_tokens: Maximum token limit.
        token_counter: Token counting function.

    Returns:
        (valid_chunks, errors) tuple.
    """
    valid: list[Chunk] = []
    errors: list[dict] = []

    for chunk in chunks:
        chunk_errors = []

        # Validate tokens
        try:
            validate_token_count(chunk, max_tokens, token_counter)
        except TokenLimitExceededError as e:
            chunk_errors.append({"type": "token_limit", "message": str(e)})

        # Validate metadata
        missing = validate_metadata(chunk)
        if missing:
            chunk_errors.append({
                "type": "missing_metadata",
                "fields": missing,
                "chunk_id": chunk.chunk_id,
            })

        # Validate non-empty text
        if not chunk.text or not chunk.text.strip():
            chunk_errors.append({
                "type": "empty_text",
                "chunk_id": chunk.chunk_id,
            })

        if chunk_errors:
            errors.extend(chunk_errors)
        else:
            valid.append(chunk)

    if errors:
        logger.warning(
            "Chunk validation: %d valid, %d errors out of %d total",
            len(valid), len(errors), len(chunks),
        )

    return valid, errors
