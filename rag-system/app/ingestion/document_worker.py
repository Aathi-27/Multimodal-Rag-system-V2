"""
Document Worker - Docling-based document conversion to Markdown.

Pipeline:
  Upload file → Docling DocumentConverter → Markdown output
  → Custom normalization → Custom chunking (handled externally)

Supports: PDF, DOCX, PPTX
Docling handles structure extraction. Chunking is NOT done by Docling.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def document_worker_process(
    input_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process for document conversion via Docling.

    Runs in a separate process for full failure isolation.
    Reads tasks from input_queue, posts results to result_queue.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    while True:
        task = input_queue.get()
        if task is None:  # Poison pill
            break

        task_id = task.task_id
        file_path = task.file_path

        try:
            logger.info("[%s] Processing document: %s", task_id[:8], task.original_filename)

            result = convert_document(converter, file_path, task.original_filename)
            result["task_id"] = task_id
            result_queue.put(result)

            logger.info(
                "[%s] Document converted: %d pages, %d chars",
                task_id[:8],
                result.get("page_count", 0),
                len(result.get("markdown", "")),
            )

        except Exception as e:
            logger.error("[%s] Document processing failed: %s", task_id[:8], e)
            result_queue.put({
                "task_id": task_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


def convert_document(
    converter,
    file_path: str,
    original_filename: str,
) -> dict:
    """
    Convert a document to Markdown using Docling.

    Args:
        converter: Docling DocumentConverter instance.
        file_path: Path to the uploaded file.
        original_filename: Original filename for metadata.

    Returns:
        Dict with markdown content and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Convert via Docling
    result = converter.convert(str(path))
    markdown = result.document.export_to_markdown()

    if not markdown or not markdown.strip():
        raise ValueError(f"No text extracted from document: {original_filename}")

    # Extract page count if available
    page_count = 0
    try:
        page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0
    except Exception:
        pass

    return {
        "markdown": markdown,
        "source": original_filename,
        "modality": "document",
        "page_count": page_count,
        "file_path": file_path,
    }
