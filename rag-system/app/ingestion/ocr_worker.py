"""
OCR Worker - EasyOCR text extraction from images.

Pipeline:
  Image → EasyOCR → Text blocks (confidence > 0.7)
  → Optional: MiniCPM-V caption → Separate chunk

Supports: PNG, JPEG, WEBP
Preserves image coordinates for citation linking.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OCR_CONFIDENCE_THRESHOLD = 0.7


def ocr_worker_process(
    input_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process for image OCR via EasyOCR.

    Runs in a separate process for full failure isolation.
    """
    import easyocr

    # Initialize EasyOCR reader (downloads models on first use)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    while True:
        task = input_queue.get()
        if task is None:  # Poison pill
            break

        task_id = task.task_id
        file_path = task.file_path

        try:
            logger.info("[%s] Processing image: %s", task_id[:8], task.original_filename)

            result = extract_text_from_image(reader, file_path, task.original_filename)
            result["task_id"] = task_id
            result_queue.put(result)

            logger.info(
                "[%s] OCR complete: %d text blocks, %d chars",
                task_id[:8],
                result.get("block_count", 0),
                len(result.get("ocr_text", "")),
            )

        except Exception as e:
            logger.error("[%s] OCR processing failed: %s", task_id[:8], e)
            result_queue.put({
                "task_id": task_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


def extract_text_from_image(
    reader,
    file_path: str,
    original_filename: str,
) -> dict:
    """
    Extract text from an image using EasyOCR.

    Args:
        reader: EasyOCR Reader instance.
        file_path: Path to the image file.
        original_filename: Original filename for metadata.

    Returns:
        Dict with OCR text, coordinates, and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    # Run OCR — returns list of (bbox, text, confidence)
    results = reader.readtext(str(path))

    if not results:
        return {
            "ocr_text": "",
            "blocks": [],
            "block_count": 0,
            "source": original_filename,
            "modality": "image",
            "file_path": file_path,
            "low_confidence": True,
        }

    # Filter by confidence threshold
    blocks = []
    text_parts = []

    for bbox, text, confidence in results:
        if confidence >= OCR_CONFIDENCE_THRESHOLD:
            # EasyOCR bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            blocks.append({
                "text": text,
                "confidence": round(float(confidence), 3),
                "coordinates": bbox,
            })
            text_parts.append(text)

    ocr_text = "\n".join(text_parts)

    return {
        "ocr_text": ocr_text,
        "blocks": blocks,
        "block_count": len(blocks),
        "source": original_filename,
        "modality": "image",
        "file_path": file_path,
        "low_confidence": len(blocks) == 0,
    }
