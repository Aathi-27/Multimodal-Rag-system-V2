"""
Task Queue - Multiprocessing queue manager for ingestion workers.

Manages separate worker pools per modality:
- Document pool (Docling)
- OCR pool (PaddleOCR)
- Audio pool (faster-whisper)

Uses multiprocessing.Process + Queue (NOT ProcessPoolExecutor).
Each modality has different CPU/GPU needs → separate pools.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Modality(str, Enum):
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class IngestionTask:
    """A single file ingestion task."""
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    file_path: str = ""
    original_filename: str = ""
    modality: Modality = Modality.DOCUMENT
    metadata: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None


class TaskQueue:
    """
    Manages ingestion task queues with separate worker processes per modality.

    Architecture:
    - One input queue per modality
    - One shared result queue
    - Workers are multiprocessing.Process instances
    - Full failure isolation per worker
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._task_registry: dict[str, IngestionTask] = {}

        # Per-modality input queues
        self._queues: dict[Modality, mp.Queue] = {
            Modality.DOCUMENT: mp.Queue(),
            Modality.IMAGE: mp.Queue(),
            Modality.AUDIO: mp.Queue(),
        }

        # Shared result queue
        self._result_queue: mp.Queue = mp.Queue()

        # Active worker processes
        self._workers: dict[Modality, list[mp.Process]] = {
            Modality.DOCUMENT: [],
            Modality.IMAGE: [],
            Modality.AUDIO: [],
        }

    def submit(self, task: IngestionTask) -> str:
        """Submit a task to the appropriate modality queue."""
        self._task_registry[task.task_id] = task
        self._queues[task.modality].put(task)
        logger.info(
            "Task submitted: %s [%s] → %s",
            task.task_id[:8],
            task.modality.value,
            task.original_filename,
        )
        return task.task_id

    def get_status(self, task_id: str) -> Optional[IngestionTask]:
        """Get the current status of a task."""
        return self._task_registry.get(task_id)

    def start_workers(
        self,
        document_worker_fn: Callable,
        ocr_worker_fn: Callable,
        audio_worker_fn: Callable,
    ) -> None:
        """Start worker processes for each modality."""
        worker_counts = {
            Modality.DOCUMENT: self._config.get("document_workers", 2),
            Modality.IMAGE: self._config.get("ocr_workers", 1),
            Modality.AUDIO: self._config.get("audio_workers", 1),
        }

        worker_fns = {
            Modality.DOCUMENT: document_worker_fn,
            Modality.IMAGE: ocr_worker_fn,
            Modality.AUDIO: audio_worker_fn,
        }

        for modality, count in worker_counts.items():
            for i in range(count):
                p = mp.Process(
                    target=worker_fns[modality],
                    args=(self._queues[modality], self._result_queue),
                    name=f"{modality.value}-worker-{i}",
                    daemon=True,
                )
                p.start()
                self._workers[modality].append(p)
                logger.info("Started %s worker %d (pid=%d)", modality.value, i, p.pid)

    def collect_results(self) -> list[dict]:
        """Collect all available results from the result queue (non-blocking)."""
        results = []
        while not self._result_queue.empty():
            try:
                result = self._result_queue.get_nowait()
                task_id = result.get("task_id")
                if task_id and task_id in self._task_registry:
                    task = self._task_registry[task_id]
                    if result.get("error"):
                        task.status = TaskStatus.FAILED
                        task.error = result["error"]
                    else:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                results.append(result)
            except Exception:
                break
        return results

    def shutdown(self) -> None:
        """Gracefully shut down all worker processes."""
        logger.info("Shutting down task queue...")

        # Send poison pills
        for modality, queue in self._queues.items():
            for _ in self._workers.get(modality, []):
                queue.put(None)  # Poison pill

        # Wait for workers to finish
        for modality, workers in self._workers.items():
            for w in workers:
                w.join(timeout=10)
                if w.is_alive():
                    logger.warning("Force killing %s (pid=%d)", w.name, w.pid)
                    w.kill()

        logger.info("Task queue shut down.")

    @property
    def active_worker_count(self) -> int:
        return sum(
            1 for workers in self._workers.values()
            for w in workers if w.is_alive()
        )
