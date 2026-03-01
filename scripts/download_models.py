def download_clip_model() -> None:
    """Download openai/clip-vit-base-patch32 via sentence-transformers."""
    target = MODELS_DIR / "embeddings" / "clip-vit-base-patch32"
    if target.exists() and any(target.iterdir()):
        logger.info("CLIP model already exists at %s", target)
        return

    logger.info("Downloading openai/clip-vit-base-patch32...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("clip-ViT-B-32")
    model.save(str(target))
    logger.info("CLIP model saved to %s", target)
"""
Download Models - Utility script to download all required ML models.

Downloads to the models/ directory (mounted volume in Docker):
- models/llm/gemma-2-9b-it-Q4_K_M.gguf
- models/embeddings/bge-small-en-v1.5/
- models/reranker/bge-reranker-base/
- models/whisper/faster-whisper-small/
- models/ocr/paddleocr/

Usage:
    python scripts/download_models.py [--all | --embedding | --reranker | --llm | --whisper | --ocr]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_embedding_model() -> None:
    """Download BAAI/bge-small-en-v1.5 via sentence-transformers."""
    target = MODELS_DIR / "embeddings" / "bge-small-en-v1.5"
    if target.exists() and any(target.iterdir()):
        logger.info("Embedding model already exists at %s", target)
        return

    logger.info("Downloading BAAI/bge-small-en-v1.5...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    model.save(str(target))
    logger.info("Embedding model saved to %s", target)


def download_reranker_model() -> None:
    """Download BAAI/bge-reranker-base via sentence-transformers."""
    target = MODELS_DIR / "reranker" / "bge-reranker-base"
    if target.exists() and any(target.iterdir()):
        logger.info("Reranker model already exists at %s", target)
        return

    logger.info("Downloading BAAI/bge-reranker-base...")
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-base")
    model.save(str(target))
    logger.info("Reranker model saved to %s", target)


def download_llm_model() -> None:
    """Download Gemma-2-9B GGUF from HuggingFace."""
    target = MODELS_DIR / "llm" / "gemma-2-9b-it-Q4_K_M.gguf"
    if target.exists():
        logger.info("LLM model already exists at %s", target)
        return

    logger.info("Downloading Gemma-2-9B GGUF (Q4_K_M)...")
    logger.info("This is a large file (~5.5GB). Please be patient.")

    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="bartowski/gemma-2-9b-it-GGUF",
            filename="gemma-2-9b-it-Q4_K_M.gguf",
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
        logger.info("LLM model saved to %s", target)
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Install with: pip install huggingface_hub\n"
            "Or download manually from:\n"
            "  https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
        )


def download_whisper_model() -> None:
    """Download faster-whisper small model."""
    target = MODELS_DIR / "whisper" / "faster-whisper-small"
    if target.exists() and any(target.iterdir()):
        logger.info("Whisper model already exists at %s", target)
        return

    logger.info("Downloading faster-whisper small model...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Systran/faster-whisper-small",
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        logger.info("Whisper model saved to %s", target)
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )


def download_ocr_model() -> None:
    """Download PaddleOCR models (auto-downloaded on first use)."""
    target = MODELS_DIR / "ocr" / "paddleocr"
    target.mkdir(parents=True, exist_ok=True)
    logger.info(
        "PaddleOCR models are auto-downloaded on first use. "
        "Directory created at %s", target
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ML models for RAG system")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--embedding", action="store_true", help="Download embedding model")
    parser.add_argument("--reranker", action="store_true", help="Download reranker model")
    parser.add_argument("--llm", action="store_true", help="Download LLM model")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper model")
    parser.add_argument("--ocr", action="store_true", help="Setup OCR model directory")

    parser.add_argument("--clip", action="store_true", help="Download CLIP model")
    args = parser.parse_args()

    # Default to --all if no specific model selected
    download_all = args.all or not any([
        args.embedding, args.reranker, args.llm, args.whisper, args.ocr, args.clip
    ])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    if download_all or args.embedding:
        tasks.append(("Embedding", download_embedding_model))
    if download_all or args.reranker:
        tasks.append(("Reranker", download_reranker_model))
    if download_all or args.llm:
        tasks.append(("LLM", download_llm_model))
    if download_all or args.whisper:
        tasks.append(("Whisper", download_whisper_model))
    if download_all or args.ocr:
        tasks.append(("OCR", download_ocr_model))
    if download_all or args.clip:
        tasks.append(("CLIP", download_clip_model))

    for name, func in tasks:
        logger.info("=" * 60)
        logger.info("Downloading: %s", name)
        logger.info("=" * 60)
        try:
            func()
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)

    logger.info("=" * 60)
    logger.info("Model download complete.")


if __name__ == "__main__":
    main()
