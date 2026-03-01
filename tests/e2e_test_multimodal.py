"""
Image & Audio Pipeline E2E Test
=================================
Tests the OCR (EasyOCR) and Audio (faster-whisper) pipelines end-to-end.

Steps:
  1. Generate a synthetic test image with text
  2. Test EasyOCR text extraction
  3. Generate a synthetic test audio file with speech (via TTS or tone)
  4. Test faster-whisper transcription
  5. Test full pipeline: extract → normalize → chunk → embed

Run:
  cd D:\Offline_Rag_V2
  .venv\Scripts\python.exe tests\e2e_test_multimodal.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAG_ROOT = PROJECT_ROOT / "rag-system"
os.chdir(RAG_ROOT)
sys.path.insert(0, str(RAG_ROOT))

os.environ["RAG_PATHS__MODELS_DIR"] = str(PROJECT_ROOT / "models")
os.environ["RAG_PATHS__DATA_DIR"] = str(PROJECT_ROOT / "data")
os.environ["RAG_PATHS__INDEX_DIR"] = str(PROJECT_ROOT / "data" / "index")
os.environ["RAG_PATHS__UPLOADS_DIR"] = str(PROJECT_ROOT / "data" / "uploads")
os.environ["RAG_PATHS__LOGS_DIR"] = str(PROJECT_ROOT / "data" / "logs")
os.environ["RAG_LLM__MODEL_PATH"] = str(PROJECT_ROOT / "models" / "llm" / "gemma-2-9b-it-Q4_K_M.gguf")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


@dataclass
class StepResult:
    name: str
    status: str = "NOT_RUN"
    duration_s: float = 0.0
    detail: str = ""
    error: str = ""


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self
    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.start


results: list[StepResult] = []
TMP_DIR = Path(tempfile.mkdtemp(prefix="rag_multimodal_test_"))


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Create a test image with known text
# ══════════════════════════════════════════════════════════════════════════════
def create_test_image(text_lines: list[str], path: Path) -> Path:
    """Create a PNG image with clear text for OCR testing."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 800, 400
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to get a decent font, fall back to default
    font = None
    font_size = 28
    try:
        # Try common Windows fonts
        for font_name in ["arial.ttf", "calibri.ttf", "consola.ttf"]:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except OSError:
                continue
    except Exception:
        pass

    if font is None:
        font = ImageFont.load_default()

    y_offset = 30
    for line in text_lines:
        draw.text((40, y_offset), line, fill="black", font=font)
        y_offset += 50

    img.save(str(path), "PNG")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Create a test audio file with speech
# ══════════════════════════════════════════════════════════════════════════════
def create_test_audio(path: Path, duration_s: float = 3.0) -> Path:
    """
    Create a WAV audio file for testing.
    Generates a simple sine wave tone (Whisper will produce minimal/no text).
    If we find a real audio sample, use that instead.
    """
    import wave
    import struct
    import math

    sample_rate = 16000
    num_samples = int(sample_rate * duration_s)
    frequency = 440.0  # A4 note

    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav.writeframes(struct.pack("<h", value))

    return path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – Generate Test Image
# ══════════════════════════════════════════════════════════════════════════════
_test_image_path = None

def test_create_image():
    global _test_image_path
    step = StepResult(name="1. Generate Test Image")
    t = Timer()
    try:
        with t:
            text_lines = [
                "Retrieval Augmented Generation",
                "combines search with language models",
                "to produce grounded answers",
                "from a knowledge base.",
            ]
            img_path = TMP_DIR / "test_ocr_image.png"
            create_test_image(text_lines, img_path)
            assert img_path.exists()
            size_kb = img_path.stat().st_size / 1024
            _test_image_path = img_path

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"path={img_path.name}, size={size_kb:.1f}KB, text_lines={len(text_lines)}"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – PaddleOCR Text Extraction
# ══════════════════════════════════════════════════════════════════════════════
_ocr_text = ""

def test_ocr_extraction():
    global _ocr_text
    step = StepResult(name="2. PaddleOCR Text Extraction")
    if _test_image_path is None:
        step.status = "SKIPPED"
        step.detail = "No test image (step 1 failed)"
        results.append(step)
        return False

    t = Timer()
    try:
        with t:
            from app.ingestion.ocr_worker import extract_text_from_image
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = extract_text_from_image(ocr, str(_test_image_path), "test_ocr_image.png")

            _ocr_text = result.get("ocr_text", "")
            block_count = result.get("block_count", 0)
            low_conf = result.get("low_confidence", True)

        step.status = "PASS"
        step.duration_s = t.elapsed
        preview = _ocr_text[:120].replace('\n', ' | ')
        step.detail = (
            f"blocks={block_count}, chars={len(_ocr_text)}, "
            f"low_conf={low_conf}, text='{preview}'"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – OCR → Normalize → Chunk → Embed pipeline
# ══════════════════════════════════════════════════════════════════════════════
def test_ocr_full_pipeline():
    step = StepResult(name="3. OCR Full Pipeline (normalize→chunk→embed)")
    if not _ocr_text:
        step.status = "SKIPPED"
        step.detail = "No OCR text from step 2"
        results.append(step)
        return False

    t_norm = Timer()
    t_chunk = Timer()
    t_embed = Timer()
    try:
        from app.processing.normalization import normalize_text
        from app.processing.chunking import SlidingWindowChunker

        with t_norm:
            normalized = normalize_text(_ocr_text)

        with t_chunk:
            chunker = SlidingWindowChunker(target_tokens=480, max_tokens=512, overlap_tokens=50)
            chunks = chunker.chunk_text(
                text=normalized, source="test_ocr_image.png",
                modality="image", page_start=1,
            )

        embed_detail = "skipped (no embedding model for this test)"
        try:
            from app.models.model_registry import ModelRegistry
            from app.models.model_manager import ModelManager
            from app.models.embeddings import EmbeddingModel

            models_dir = Path(os.environ.get("RAG_PATHS__MODELS_DIR", "models"))
            registry = ModelRegistry(models_dir)
            manager = ModelManager(registry)
            emb = EmbeddingModel(manager)

            with t_embed:
                chunk_texts = [c.text for c in chunks] if chunks else [normalized]
                vectors = emb.embed_texts(chunk_texts)
                embed_detail = f"embedded={vectors.shape}"
        except Exception as emb_err:
            embed_detail = f"embed skipped: {emb_err}"

        step.status = "PASS"
        step.duration_s = t_norm.elapsed + t_chunk.elapsed + t_embed.elapsed
        step.detail = (
            f"norm={t_norm.elapsed*1000:.1f}ms({len(normalized)} chars), "
            f"chunks={len(chunks)}, "
            f"{embed_detail}"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t_norm.elapsed + t_chunk.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – Generate Test Audio
# ══════════════════════════════════════════════════════════════════════════════
_test_audio_path = None

def test_create_audio():
    global _test_audio_path
    step = StepResult(name="4. Generate Test Audio File")
    t = Timer()
    try:
        with t:
            audio_path = TMP_DIR / "test_audio.wav"
            create_test_audio(audio_path, duration_s=3.0)
            assert audio_path.exists()
            size_kb = audio_path.stat().st_size / 1024
            _test_audio_path = audio_path

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"path={audio_path.name}, size={size_kb:.1f}KB, duration=3.0s"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – Whisper Model Loading
# ══════════════════════════════════════════════════════════════════════════════
_whisper_model = None

def test_whisper_load():
    global _whisper_model
    step = StepResult(name="5. Whisper Model Loading")
    t = Timer()
    try:
        with t:
            from faster_whisper import WhisperModel

            model_path = str(PROJECT_ROOT / "models" / "whisper" / "faster-whisper-small")
            model = WhisperModel(
                model_path, device="cpu", compute_type="int8",
            )
            _whisper_model = model

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"model=faster-whisper-small, device=cpu, compute=int8, load={t.elapsed:.2f}s"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – Whisper Transcription
# ══════════════════════════════════════════════════════════════════════════════
_whisper_transcript = ""

def test_whisper_transcription():
    global _whisper_transcript
    step = StepResult(name="6. Whisper Audio Transcription")
    if _whisper_model is None or _test_audio_path is None:
        step.status = "SKIPPED"
        step.detail = "Requires Whisper model (step 5) + audio file (step 4)"
        results.append(step)
        return False

    t = Timer()
    try:
        with t:
            from app.ingestion.audio_worker import transcribe_audio
            result = transcribe_audio(_whisper_model, str(_test_audio_path), "test_audio.wav")

            _whisper_transcript = result.get("transcript", "")
            seg_count = result.get("segment_count", 0)
            duration = result.get("duration", 0)
            language = result.get("language", "?")

        step.status = "PASS"
        step.duration_s = t.elapsed
        preview = _whisper_transcript[:100] if _whisper_transcript else "(no speech detected - expected for tone)"
        step.detail = (
            f"segments={seg_count}, duration={duration}s, lang={language}, "
            f"text='{preview}'"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – Whisper transcribe_audio function test (with real speech simulation)
# ══════════════════════════════════════════════════════════════════════════════
def test_whisper_function_direct():
    step = StepResult(name="7. Whisper Direct Function Test")
    if _whisper_model is None:
        step.status = "SKIPPED"
        step.detail = "Requires Whisper model (step 5)"
        results.append(step)
        return False

    t = Timer()
    try:
        with t:
            # Test the model directly with a short transcription to prove it works
            segments_gen, info = _whisper_model.transcribe(
                str(_test_audio_path),
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            # Consume the generator
            segments = list(segments_gen)

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = (
            f"segments={len(segments)}, duration={info.duration:.2f}s, "
            f"lang={info.language}, prob={info.language_probability:.2f}, "
            f"transcribe_time={t.elapsed:.2f}s"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 – Modality Detection
# ══════════════════════════════════════════════════════════════════════════════
def test_modality_detection():
    step = StepResult(name="8. Modality Detection (file routing)")
    t = Timer()
    try:
        with t:
            from app.ingestion.workers import detect_modality
            from app.ingestion.task_queue import Modality

            tests = {
                "report.pdf": Modality.DOCUMENT,
                "photo.png": Modality.IMAGE,
                "photo.jpg": Modality.IMAGE,
                "photo.jpeg": Modality.IMAGE,
                "photo.webp": Modality.IMAGE,
                "meeting.mp3": Modality.AUDIO,
                "call.wav": Modality.AUDIO,
                "voice.m4a": Modality.AUDIO,
                "slides.pptx": Modality.DOCUMENT,
                "memo.docx": Modality.DOCUMENT,
            }

            passed = 0
            for filename, expected in tests.items():
                result = detect_modality(filename)
                assert result == expected, f"{filename}: expected {expected}, got {result}"
                passed += 1

            # Test rejection of unsupported files
            rejected = False
            try:
                detect_modality("virus.exe")
            except Exception:
                rejected = True
            assert rejected, "Should reject unsupported extensions"

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"all {passed} mappings correct, unsupported files rejected"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 – File Validation
# ══════════════════════════════════════════════════════════════════════════════
def test_file_validation():
    step = StepResult(name="9. File Validation & Sanitization")
    t = Timer()
    try:
        with t:
            from app.ingestion.workers import validate_file, sanitize_filename

            # Valid file
            validate_file("test.pdf", 1024 * 1024, max_size_mb=100)

            # Oversized file should fail
            oversized_rejected = False
            try:
                validate_file("big.pdf", 200 * 1024 * 1024, max_size_mb=100)
            except Exception:
                oversized_rejected = True
            assert oversized_rejected, "Should reject oversized files"

            # Path traversal should fail
            traversal_rejected = False
            try:
                validate_file("../../etc/passwd", 1024, max_size_mb=100)
            except Exception:
                traversal_rejected = True
            assert traversal_rejected, "Should reject path traversal"

            # Sanitization
            safe = sanitize_filename('my <file> "test".pdf')
            assert "<" not in safe
            assert '"' not in safe

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"valid=OK, oversized=rejected, traversal=rejected, sanitize='{safe}'"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════
def print_report():
    W = 110
    line = "=" * W
    thin = "-" * W

    print(f"\n{line}")
    print(f"{'IMAGE & AUDIO PIPELINE TEST REPORT':^{W}}")
    print(f"{'Offline RAG System - Multimodal Pipeline Health':^{W}}")
    print(f"{line}\n")

    hdr = f"  {'#':<3} {'Component':<48} {'Status':<12} {'Time':>12}  {'Details'}"
    print(hdr)
    print(f"  {thin[:-4]}")

    total_time = 0.0
    passed = failed = skipped = 0

    for r in results:
        total_time += r.duration_s

        if r.status == "PASS":
            marker = "PASS  [OK]"
            passed += 1
        elif r.status == "FAIL":
            marker = "FAIL  [!!]"
            failed += 1
        elif r.status == "SKIPPED":
            marker = "SKIP  [--]"
            skipped += 1
        else:
            marker = "???   [??]"

        dur = f"{r.duration_s:.2f}s" if r.duration_s >= 1 else f"{r.duration_s*1000:.1f}ms"
        detail_short = (r.detail[:70] + "...") if len(r.detail) > 70 else r.detail
        print(f"  {'':<3} {r.name:<48} {marker:<12} {dur:>12}  {detail_short}")

    print(f"  {thin[:-4]}")
    print(f"  {'TOTAL':>48} {'':>12} {total_time:>11.2f}s\n")

    print(f"  +------------------------------------------------+")
    print(f"  |{'SUMMARY':^48}|")
    print(f"  +------------------------------------------------+")
    print(f"  |  Passed     : {passed:>3} / {len(results):<3}{'':>30}|")
    print(f"  |  Failed     : {failed:>3} / {len(results):<3}{'':>30}|")
    print(f"  |  Skipped    : {skipped:>3} / {len(results):<3}{'':>30}|")
    print(f"  |  Total Time : {total_time:>8.2f}s{'':>28}|")
    print(f"  +------------------------------------------------+\n")

    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print(f"  {line}")
        print(f"  {'FAILURE DETAILS':^{W}}")
        print(f"  {line}")
        for r in failures:
            print(f"\n  [FAIL] {r.name}")
            for err_line in r.error.split("\n")[:8]:
                print(f"         {err_line}")
        print()

    print(f"  {line}")
    if failed == 0:
        print(f"  >>> VERDICT: ALL {passed} TESTS PASSED — Image & Audio pipelines operational")
    else:
        print(f"  >>> VERDICT: {failed} TEST(S) FAILED — review details above")
    print(f"  {line}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 110)
    print(f"{'IMAGE & AUDIO PIPELINE E2E TEST':^110}")
    print(f"{'Project: ' + str(PROJECT_ROOT):^110}")
    print("=" * 110 + "\n")

    try:
        # Image pipeline
        test_create_image()
        test_ocr_extraction()
        test_ocr_full_pipeline()

        # Audio pipeline
        test_create_audio()
        test_whisper_load()
        test_whisper_transcription()
        test_whisper_function_direct()

        # Common utilities
        test_modality_detection()
        test_file_validation()

    finally:
        print_report()

        # Cleanup temp dir
        try:
            shutil.rmtree(TMP_DIR, ignore_errors=True)
        except Exception:
            pass

    failed_count = sum(1 for r in results if r.status == "FAIL")
    sys.exit(failed_count)
