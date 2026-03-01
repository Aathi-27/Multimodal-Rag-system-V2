"""
Index Manager - Versioned index handler for Qdrant and BM25.

Manages index versions with symlink-based switching:
  data/index/v1.0.0/qdrant/
  data/index/v1.0.0/bm25/
  data/index/current -> v1.0.0

Handles:
- Creating new index versions
- Switching active version via symlink
- Listing available versions
- Cleanup of old versions
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages versioned index directories for Qdrant and BM25."""

    CURRENT_LINK = "current"
    QDRANT_DIR = "qdrant"
    BM25_DIR = "bm25"
    METADATA_FILE = "version_metadata.json"

    def __init__(self, index_root: Path) -> None:
        """
        Args:
            index_root: Path to data/index/ directory.
        """
        self._root = index_root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def current_link(self) -> Path:
        return self._root / self.CURRENT_LINK

    @property
    def current_version(self) -> Optional[str]:
        """Get the currently active index version."""
        link = self.current_link
        if link.exists():
            # Resolve symlink or junction target
            target = link.resolve()
            return target.name
        return None

    @property
    def current_qdrant_path(self) -> Optional[Path]:
        """Path to the current Qdrant storage directory."""
        if self.current_link.exists():
            return self.current_link / self.QDRANT_DIR
        return None

    @property
    def current_bm25_path(self) -> Optional[Path]:
        """Path to the current BM25 index directory."""
        if self.current_link.exists():
            return self.current_link / self.BM25_DIR
        return None

    def list_versions(self) -> list[str]:
        """List all available index versions, sorted."""
        versions = []
        for d in self._root.iterdir():
            if d.is_dir() and d.name != self.CURRENT_LINK:
                versions.append(d.name)
        return sorted(versions)

    def create_version(self, version: str) -> Path:
        """
        Create a new index version directory structure.

        Args:
            version: Version string (e.g., "v1.0.0")

        Returns:
            Path to the new version directory.
        """
        version_dir = self._root / version
        if version_dir.exists():
            raise FileExistsError(f"Index version {version} already exists.")

        (version_dir / self.QDRANT_DIR).mkdir(parents=True)
        (version_dir / self.BM25_DIR).mkdir(parents=True)
        logger.info("Created index version: %s", version)
        return version_dir

    def switch_to(self, version: str) -> None:
        """
        Switch the active index to the specified version.
        Updates the 'current' symlink/junction.

        Args:
            version: Version to activate.
        """
        version_dir = self._root / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Index version {version} does not exist.")

        link = self.current_link

        # Remove existing link
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                # On Windows, junctions appear as dirs
                link.unlink()

        # Create symlink (or junction on Windows)
        try:
            link.symlink_to(version_dir, target_is_directory=True)
        except OSError:
            # Windows fallback: use directory junction
            import subprocess
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(version_dir)],
                check=True,
                capture_output=True,
            )

        logger.info("Switched active index to version: %s", version)

    def delete_version(self, version: str) -> None:
        """Delete an index version. Cannot delete the active version."""
        if version == self.current_version:
            raise ValueError("Cannot delete the currently active index version.")

        version_dir = self._root / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Index version {version} does not exist.")

        shutil.rmtree(version_dir)
        logger.info("Deleted index version: %s", version)

    def ensure_initialized(self, default_version: str = "v1.0.0") -> None:
        """Ensure at least one version exists and is active."""
        if not self.list_versions():
            self.create_version(default_version)

        if not self.current_link.exists():
            versions = self.list_versions()
            if versions:
                self.switch_to(versions[0])

    # ── Version Metadata ──────────────────────────────────────────────

    def save_metadata(self, version: str, metadata: dict) -> None:
        """
        Save metadata alongside an index version.

        Metadata typically includes:
            embedding_model, chunk_size, overlap, rerank_threshold,
            rrf_k, entity_limit, created_at, chunk_count, description
        """
        version_dir = self._root / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Index version {version} does not exist.")

        meta_path = version_dir / self.METADATA_FILE
        metadata.setdefault("created_at", time.time())
        metadata.setdefault("version", version)

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata for version %s", version)

    def get_metadata(self, version: str) -> dict:
        """
        Load metadata for a version. Returns empty dict if no metadata file.
        """
        version_dir = self._root / version
        meta_path = version_dir / self.METADATA_FILE

        if not meta_path.exists():
            return {}

        with open(meta_path, "r") as f:
            return json.load(f)

    def list_versions_with_metadata(self) -> list[dict]:
        """List all versions with their metadata."""
        result = []
        for v in self.list_versions():
            meta = self.get_metadata(v)
            result.append({
                "version": v,
                "is_active": v == self.current_version,
                "metadata": meta,
            })
        return result
