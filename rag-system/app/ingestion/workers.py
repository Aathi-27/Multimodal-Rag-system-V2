"""
Workers - Worker pool management and modality routing.

Routes uploaded files to the correct worker pool based on file extension:
- Documents: .pdf, .docx, .pptx → Document worker pool
- Images:    .png, .jpg, .jpeg, .webp → OCR worker pool
- Audio:     .mp3, .wav, .m4a → Audio worker pool

Uses multiprocessing.Process + Queue (NOT ProcessPoolExecutor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.ingestion.task_queue import IngestionTask, Modality, TaskQueue
from app.utils.exceptions import UnsupportedFileTypeError

logger = logging.getLogger(__name__)

# Modality routing by file extension
EXTENSION_MAP: dict[str, Modality] = {
    # Documents
    ".pdf": Modality.DOCUMENT,
    ".docx": Modality.DOCUMENT,
    ".pptx": Modality.DOCUMENT,
    # Images
    ".png": Modality.IMAGE,
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".webp": Modality.IMAGE,
    # Audio
    ".mp3": Modality.AUDIO,
    ".wav": Modality.AUDIO,
    ".m4a": Modality.AUDIO,
    ".ogg": Modality.AUDIO,
    ".flac": Modality.AUDIO,
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())


def detect_modality(filename: str) -> Modality:
    """
    Determine the processing modality based on file extension.

    Args:
        filename: Original filename with extension.

    Returns:
        Modality enum value.

    Raises:
        UnsupportedFileTypeError: If extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    if ext not in EXTENSION_MAP:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return EXTENSION_MAP[ext]


def validate_file(
    filename: str,
    file_size_bytes: int,
    max_size_mb: int = 100,
) -> None:
    """
    Validate an uploaded file before ingestion.

    Checks:
    1. File extension is supported
    2. File size is within limits
    3. Filename is safe (no path traversal)

    Raises:
        UnsupportedFileTypeError: Bad extension
        FileSizeLimitError: File too large
        FileValidationError: Dangerous filename
    """
    from app.utils.exceptions import FileSizeLimitError, FileValidationError

    # Check extension
    detect_modality(filename)

    # Check size
    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise FileSizeLimitError(
            f"File size {file_size_bytes / (1024*1024):.1f}MB exceeds "
            f"limit of {max_size_mb}MB."
        )

    # Sanitize filename (prevent path traversal)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise FileValidationError(
            f"Invalid filename: '{filename}'. Path traversal not allowed."
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.

    Strips path components and replaces unsafe characters.
    """
    # Take only the filename part
    name = Path(filename).name

    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        name = name.replace(char, "_")

    return name
