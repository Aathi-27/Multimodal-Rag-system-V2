"""
Audio Worker - faster-whisper transcription with speaker diarization.

Pipeline:
  Audio → faster-whisper → Raw transcript segments
  → pyannote.audio → Speaker labels
  → Merge → Speaker-tagged text
  → Custom chunking (by token count, NOT by Whisper segments)

Supports: MP3, WAV, M4A
Preserves speaker metadata and timestamps.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def audio_worker_process(
    input_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process for audio transcription via faster-whisper.

    Runs in a separate process for full failure isolation.
    """
    from faster_whisper import WhisperModel

    # Load model (CTranslate2 backend)
    model = WhisperModel("small", device="cpu", compute_type="int8")

    while True:
        task = input_queue.get()
        if task is None:  # Poison pill
            break

        task_id = task.task_id
        file_path = task.file_path

        try:
            logger.info("[%s] Processing audio: %s", task_id[:8], task.original_filename)

            result = transcribe_audio(model, file_path, task.original_filename)
            result["task_id"] = task_id
            result_queue.put(result)

            logger.info(
                "[%s] Transcription complete: %d segments, %.1f seconds",
                task_id[:8],
                result.get("segment_count", 0),
                result.get("duration", 0),
            )

        except Exception as e:
            logger.error("[%s] Audio processing failed: %s", task_id[:8], e)
            result_queue.put({
                "task_id": task_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


def transcribe_audio(
    model,
    file_path: str,
    original_filename: str,
) -> dict:
    """
    Transcribe audio using faster-whisper.

    Args:
        model: WhisperModel instance.
        file_path: Path to the audio file.
        original_filename: Original filename for metadata.

    Returns:
        Dict with transcript, segments, and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Transcribe
    segments_gen, info = model.transcribe(
        str(path),
        beam_size=5,
        language="en",
        vad_filter=True,        # Voice Activity Detection to skip silence
    )

    segments = []
    full_text_parts = []

    for segment in segments_gen:
        seg_data = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "speaker": None,  # Populated by diarization step
        }
        segments.append(seg_data)
        full_text_parts.append(segment.text.strip())

    full_text = " ".join(full_text_parts)

    if not full_text.strip():
        logger.warning("No speech detected in %s", original_filename)

    return {
        "transcript": full_text,
        "segments": segments,
        "segment_count": len(segments),
        "duration": round(info.duration, 2),
        "language": info.language,
        "source": original_filename,
        "modality": "audio",
        "file_path": file_path,
    }


def add_speaker_diarization(
    audio_path: str,
    segments: list[dict],
    hf_token: Optional[str] = None,
) -> list[dict]:
    """
    Add speaker labels to transcript segments via pyannote.audio.

    Note: pyannote requires a HuggingFace token for model access.
    This step is optional and runs post-transcription.

    Args:
        audio_path: Path to the audio file.
        segments: List of transcript segments from Whisper.
        hf_token: HuggingFace API token for pyannote model access.

    Returns:
        Segments with speaker labels added.
    """
    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

        diarization = pipeline(audio_path)

        # Assign speakers to segments based on timestamp overlap
        for segment in segments:
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_mid = (seg_start + seg_end) / 2

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= seg_mid <= turn.end:
                    segment["speaker"] = speaker
                    break

            if segment["speaker"] is None:
                segment["speaker"] = "Unknown"

        return segments

    except Exception as e:
        logger.warning("Speaker diarization failed: %s. Using default labels.", e)
        for segment in segments:
            segment["speaker"] = "Speaker"
        return segments
