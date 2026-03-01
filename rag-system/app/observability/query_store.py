"""
Query Store - Persistent query history with full retrieval context.

Stores every query with:
- Query text, timestamp, query_id
- Retrieved chunks (with scores)
- Final answer
- Per-stage latency (retrieval, rerank, generation, total)
- Debug info (if enabled)

Persists to JSONL file (one JSON object per line) for efficient append.
Thread-safe via a lock. Supports replay by query_id.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    query_id: str
    query: str
    timestamp: float = field(default_factory=time.time)
    # Retrieval
    retrieved_chunks: list[dict] = field(default_factory=list)
    chunk_count: int = 0
    # Answer
    answer: str = ""
    citation_count: int = 0
    # Latency (seconds)
    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    # Debug
    debug_enabled: bool = False
    debug_info: Optional[dict] = None
    # Metadata
    modality_filter: Optional[str] = None
    department_filter: Optional[str] = None
    token_count: int = 0
    error: Optional[str] = None
    # Recall validation (added by POST /queries/{id}/validate)
    recall_validation: Optional[dict] = None
    # ── Research Lab extensions ──────────────────────────────────────
    # Stage-wise survival log (populated when debug=True)
    survival_log: Optional[list[dict]] = None
    # Ground truth chunk IDs for evaluation
    ground_truth_chunk_ids: Optional[list[str]] = None
    # Automated failure diagnosis
    diagnosis: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class QueryStore:
    """Append-only query history store backed by a JSONL file."""

    MAX_RECORDS_IN_MEMORY = 500  # Keep last 500 in RAM for fast access

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "query_history.jsonl"
        self._lock = threading.Lock()
        self._records: list[QueryRecord] = []
        self._index: dict[str, int] = {}  # query_id -> index in _records
        self._load()

    def _load(self) -> None:
        """Load existing history from disk."""
        if not self._file.exists():
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = QueryRecord(**{
                            k: v for k, v in data.items()
                            if k in QueryRecord.__dataclass_fields__
                        })
                        self._records.append(record)
                        self._index[record.query_id] = len(self._records) - 1
                    except Exception:
                        continue

            # Keep only last N in memory
            if len(self._records) > self.MAX_RECORDS_IN_MEMORY:
                self._records = self._records[-self.MAX_RECORDS_IN_MEMORY:]
                self._index = {
                    r.query_id: i for i, r in enumerate(self._records)
                }

            logger.info("Query history loaded: %d records.", len(self._records))
        except Exception as e:
            logger.warning("Failed to load query history: %s", e)

    def record(self, query_record: QueryRecord) -> None:
        """Append a query record."""
        with self._lock:
            self._records.append(query_record)
            self._index[query_record.query_id] = len(self._records) - 1

            # Persist to disk
            try:
                with open(self._file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(query_record.to_dict()) + "\n")
            except Exception as e:
                logger.warning("Failed to persist query record: %s", e)

            # Trim memory if needed
            if len(self._records) > self.MAX_RECORDS_IN_MEMORY:
                self._records = self._records[-self.MAX_RECORDS_IN_MEMORY:]
                self._index = {
                    r.query_id: i for i, r in enumerate(self._records)
                }

    def get_all(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Return paginated query history (newest first)."""
        with self._lock:
            total = len(self._records)
            records = list(reversed(self._records))
            page = records[offset: offset + limit]
            return [r.to_dict() for r in page], total

    def get_by_id(self, query_id: str) -> Optional[dict]:
        """Get a specific query record by ID."""
        with self._lock:
            idx = self._index.get(query_id)
            if idx is not None and idx < len(self._records):
                return self._records[idx].to_dict()
            return None

    def annotate(self, query_id: str, annotations: dict) -> bool:
        """
        Merge additional annotations into an existing query record.

        Used by recall validation to persist recall@k metrics alongside
        the original query record.

        Args:
            query_id: The query to annotate.
            annotations: Dict of fields to merge (e.g. recall_validation).

        Returns:
            True if record found and updated, False otherwise.
        """
        with self._lock:
            idx = self._index.get(query_id)
            if idx is None or idx >= len(self._records):
                return False

            record = self._records[idx]
            for key, value in annotations.items():
                if hasattr(record, key):
                    setattr(record, key, value)

            # Re-persist the entire file (JSONL doesn't support in-place update)
            self._persist_all()
            return True

    def _persist_all(self) -> None:
        """Rewrite the full JSONL file from memory (called after annotation)."""
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                for record in self._records:
                    f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning("Failed to rewrite query history: %s", e)

    def delete_by_id(self, query_id: str) -> bool:
        """Delete a single query record by ID."""
        with self._lock:
            idx = self._index.get(query_id)
            if idx is None or idx >= len(self._records):
                return False
            self._records.pop(idx)
            # Rebuild index after removal
            self._index = {
                r.query_id: i for i, r in enumerate(self._records)
            }
            self._persist_all()
            return True

    def delete_all(self) -> int:
        """Delete all query records. Returns count deleted."""
        with self._lock:
            count = len(self._records)
            self._records.clear()
            self._index.clear()
            self._persist_all()
            return count

    def get_summary(self) -> dict:
        """Return aggregate statistics."""
        with self._lock:
            if not self._records:
                return {
                    "total_queries": 0,
                    "avg_latency": 0,
                    "avg_retrieval_latency": 0,
                    "avg_generation_latency": 0,
                    "avg_chunks_per_query": 0,
                    "error_count": 0,
                }

            total = len(self._records)
            return {
                "total_queries": total,
                "avg_latency": round(
                    sum(r.total_latency for r in self._records) / total, 3
                ),
                "avg_retrieval_latency": round(
                    sum(r.retrieval_latency for r in self._records) / total, 3
                ),
                "avg_generation_latency": round(
                    sum(r.generation_latency for r in self._records) / total, 3
                ),
                "avg_chunks_per_query": round(
                    sum(r.chunk_count for r in self._records) / total, 1
                ),
                "error_count": sum(1 for r in self._records if r.error),
            }
