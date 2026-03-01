"""
File Registry — Tracks uploaded files for citation navigation.

Provides a persistent mapping from file_id → file metadata, enabling
the /files/{file_id} endpoint to securely serve original uploaded content.

Storage: JSON file at data/file_registry.json
Schema per entry:
  - file_id       (str)  — upload_id from ingestion
  - file_path     (str)  — absolute path to saved file
  - file_type     (str)  — MIME type
  - file_name     (str)  — original filename
  - modality      (str)  — document | audio | image
  - upload_time   (str)  — ISO timestamp
  - file_size     (int)  — bytes
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MIME type mapping for modalities
_MIME_MAP: dict[str, str] = {
    # Documents
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".wma": "audio/x-ms-wma",
    ".aac": "audio/aac",
}


class FileRegistry:
    """Thread-safe persistent file registry for uploaded content."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._registry_path = data_dir / "file_registry.json"
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r") as f:
                    data = json.load(f)
                self._entries = {e["file_id"]: e for e in data}
                logger.info("File registry loaded: %d entries", len(self._entries))
            except Exception as e:
                logger.warning("Failed to load file registry: %s", e)
                self._entries = {}
        else:
            self._entries = {}

    def _save(self) -> None:
        """Persist registry to disk (must be called under lock)."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_path, "w") as f:
            json.dump(list(self._entries.values()), f, indent=2, default=str)

    # ── Public API ───────────────────────────────────────────────────────

    def register(
        self,
        file_id: str,
        file_path: str,
        file_name: str,
        modality: str,
    ) -> dict:
        """
        Register a newly uploaded file.

        Args:
            file_id: Unique identifier (upload_id).
            file_path: Absolute path where the file is saved.
            file_name: Original filename.
            modality: document | audio | image.

        Returns:
            The registry entry dict.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        mime = _MIME_MAP.get(ext) or mimetypes.guess_type(file_name)[0] or "application/octet-stream"

        entry = {
            "file_id": file_id,
            "file_path": str(path.resolve()),
            "file_name": file_name,
            "file_type": mime,
            "modality": modality,
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "file_size": path.stat().st_size if path.exists() else 0,
        }

        with self._lock:
            self._entries[file_id] = entry
            self._save()

        logger.info("Registered file: %s → %s [%s]", file_id[:8], file_name, modality)
        return entry

    def get(self, file_id: str) -> Optional[dict]:
        """Look up a file by ID. Returns None if not found."""
        with self._lock:
            return self._entries.get(file_id)

    def get_by_name(self, file_name: str) -> Optional[dict]:
        """Look up a file by original filename. Returns first match."""
        with self._lock:
            for entry in self._entries.values():
                if entry["file_name"] == file_name:
                    return entry
        return None

    def list_all(self) -> list[dict]:
        """Return all registry entries."""
        with self._lock:
            return list(self._entries.values())

    def delete(self, file_id: str) -> bool:
        """Remove an entry from the registry."""
        with self._lock:
            if file_id in self._entries:
                del self._entries[file_id]
                self._save()
                return True
        return False

    def count(self) -> int:
        """Number of registered files."""
        with self._lock:
            return len(self._entries)

    def backfill_existing(self, uploads_dir: Path) -> int:
        """
        Scan the uploads directory and register any files not yet tracked.

        File naming convention: {upload_id}_{original_filename}
        Returns the number of newly registered files.
        """
        from app.ingestion.workers import detect_modality

        if not uploads_dir.exists():
            return 0

        added = 0
        for f in uploads_dir.iterdir():
            if not f.is_file():
                continue

            # Parse upload_id from filename: {32-char-hex}_{original}
            name = f.name
            if len(name) > 33 and name[32] == "_":
                upload_id = name[:32]
                original_name = name[33:]
            else:
                continue  # Skip files that don't match the convention

            if upload_id in self._entries:
                continue  # Already registered

            try:
                modality = detect_modality(original_name)
                self.register(upload_id, str(f), original_name, modality.value)
                added += 1
            except Exception:
                continue

        if added:
            logger.info("Backfilled %d existing files into registry.", added)
        return added
