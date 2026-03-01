"""
API Routes - FastAPI endpoints for upload, query, and health.

Endpoints:
  POST /upload   - Upload and index a document, image, or audio file (202)
  POST /query    - Submit a question, receive streamed response with citations
  GET  /health   - System health check
  GET  /status/{task_id} - Check upload processing status
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TaskStatusResponse,
    UploadResponse,
    DeleteDocumentResponse,
    DocumentChunksResponse,
    DocumentListResponse,
    DocumentSummary,
    ChunkDetail,
    IndexHealthResponse,
    AnalyticsResponse,
    DocumentAnalytics,
    ReindexResponse,
    QueryHistoryItem,
    QueryHistoryResponse,
    QueryDetailResponse,
    QuerySummaryResponse,
    VersionInfo,
    VersionListResponse,
    VersionSwitchResponse,
    LatencyBreakdown,
    MetricsResponse,
    ResourceStatus,
    RetrievalSettingsRequest,
    RetrievalSettingsResponse,
    RecallValidationRequest,
    RecallMetrics,
    # Research Lab models
    SurvivalResponse,
    DiagnosisResponse,
    BatchDiagnosisResponse,
    GroundTruthRequest,
    CompareRequest,
    BatchEvalRequest,
    CorpusCoverageResponse,
    EmbeddingQualityResponse,
    VersionMetadataRequest,
    VersionDetailResponse,
    # Phase 4 models
    ImageQueryResponse,
    AudioQueryResponse,
)
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Track startup time for uptime calculation
_start_time = time.time()

# In-memory task tracker (simple dict; replace with Redis/DB in production)
_task_status: dict[str, dict] = {}


# ── Upload Endpoint ───────────────────────────────────────────────────────────


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    department: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    """
    Upload and index a document, image, or audio file.

    Returns 202 Accepted with an upload_id for status tracking.
    Ingestion runs as a background task.
    """
    from app.ingestion.workers import detect_modality, sanitize_filename, validate_file

    settings = get_settings()

    # Validate file
    filename = file.filename or "unknown"
    file_bytes = await file.read()
    file_size = len(file_bytes)

    try:
        validate_file(filename, file_size, settings.server.max_upload_size_mb)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Detect modality
    try:
        modality = detect_modality(filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save file to uploads directory
    safe_name = sanitize_filename(filename)
    upload_id = uuid.uuid4().hex
    upload_dir = Path(settings.paths.uploads_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / f"{upload_id}_{safe_name}"
    dest.write_bytes(file_bytes)

    logger.info(
        "File uploaded: %s → %s [%s] (%d bytes)",
        filename, dest.name, modality.value, file_size,
    )

    # Register file in the file registry for citation navigation
    from app.api.dependencies import get_file_registry
    file_registry = get_file_registry()
    file_registry.register(
        file_id=upload_id,
        file_path=str(dest),
        file_name=filename,
        modality=modality.value,
    )

    # Track task status
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    _task_status[upload_id] = {
        "status": "processing",
        "filename": filename,
        "modality": modality.value,
    }

    # Run ingestion as a background task
    background_tasks.add_task(
        _ingest_file,
        upload_id=upload_id,
        file_path=str(dest),
        original_filename=filename,
        modality=modality.value,
        department=department,
        tags=tag_list,
    )

    return UploadResponse(
        upload_id=upload_id,
        status="processing",
        estimated_time="30s",
        filename=filename,
        modality=modality.value,
    )


def _ingest_file(
    upload_id: str,
    file_path: str,
    original_filename: str,
    modality: str,
    department: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Synchronous ingestion pipeline (runs as FastAPI BackgroundTask).

    Steps:
    1. Parse document → Markdown (via Docling)
    2. Normalize text
    3. Chunk with SlidingWindowChunker
    4. Generate embeddings (BGE-small-en-v1.5)
    5. Upsert vectors to Qdrant
    6. Rebuild BM25 index
    """
    from app.api.dependencies import get_bm25_store, get_embedding_model, get_vector_store
    from app.ingestion.document_worker import convert_document
    from app.processing.chunking import SlidingWindowChunker
    from app.processing.normalization import normalize_text

    settings = get_settings()

    try:
        logger.info("[%s] Ingestion started: %s", upload_id[:8], original_filename)

        # ── Step 1: Parse based on modality ───────────────────────────────
        page_count = 0

        if modality == "document":
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = convert_document(converter, file_path, original_filename)
            raw_text = result["markdown"]
            page_count = result.get("page_count", 0)

        elif modality == "audio":
            from faster_whisper import WhisperModel
            from app.ingestion.audio_worker import transcribe_audio

            whisper_model_dir = str(Path(settings.paths.models_dir) / "whisper" / "faster-whisper-small")
            logger.info("[%s] Loading Whisper model from %s", upload_id[:8], whisper_model_dir)
            whisper_model = WhisperModel(whisper_model_dir, device="cpu", compute_type="int8")
            result = transcribe_audio(whisper_model, file_path, original_filename)
            raw_text = result.get("transcript", "")
            if not raw_text.strip():
                logger.warning("[%s] No speech detected in audio file.", upload_id[:8])
                _task_status[upload_id] = {
                    "status": "completed",
                    "filename": original_filename,
                    "modality": modality,
                    "chunks": 0,
                    "note": "No speech detected.",
                }
                return

        elif modality == "image":
            import easyocr
            from app.ingestion.ocr_worker import extract_text_from_image

            logger.info("[%s] Running OCR on image", upload_id[:8])
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            result = extract_text_from_image(reader, file_path, original_filename)
            raw_text = result.get("ocr_text", "")
            if not raw_text.strip():
                logger.warning("[%s] No text detected in image.", upload_id[:8])
                _task_status[upload_id] = {
                    "status": "completed",
                    "filename": original_filename,
                    "modality": modality,
                    "chunks": 0,
                    "note": "No text detected in image.",
                }
                return

        else:
            logger.warning("[%s] Unknown modality '%s'.", upload_id[:8], modality)
            _task_status[upload_id] = {
                "status": "failed",
                "error": f"Unknown modality '{modality}'.",
                "filename": original_filename,
                "modality": modality,
            }
            return

        logger.info("[%s] Parsed: %d chars, %d pages", upload_id[:8], len(raw_text), page_count)

        # ── Step 2: Normalize text ────────────────────────────────────────
        normalized = normalize_text(raw_text)
        logger.info("[%s] Normalized: %d chars", upload_id[:8], len(normalized))

        # ── Step 3: Chunk ─────────────────────────────────────────────────
        chunker = SlidingWindowChunker(
            target_tokens=settings.chunking.target_tokens,
            max_tokens=settings.chunking.max_tokens,
            overlap_tokens=settings.chunking.overlap_tokens,
        )
        chunks = chunker.chunk_text(
            text=normalized,
            source=original_filename,
            modality=modality,
            page_start=1,
        )

        if not chunks:
            logger.warning("[%s] No chunks produced from document.", upload_id[:8])
            _task_status[upload_id] = {
                "status": "completed",
                "filename": original_filename,
                "modality": modality,
                "chunks": 0,
            }
            return

        logger.info("[%s] Chunked: %d chunks", upload_id[:8], len(chunks))

        # ── Step 4: Generate embeddings ───────────────────────────────────
        embedding_model = get_embedding_model()
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_model.embed_texts(chunk_texts, show_progress=True)

        logger.info("[%s] Embedded: %s", upload_id[:8], embeddings.shape)

        # ── Step 5: Upsert to Qdrant ──────────────────────────────────────
        vs = get_vector_store()
        chunk_ids = [c.chunk_id for c in chunks]
        payloads = [
            {
                "text": c.text,
                "source": c.source,
                "modality": c.modality,
                "page_start": c.page_start,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "department": department or "",
                "tags": tags or [],
                "upload_id": upload_id,
                # Phase 2: Cross-modal linking fields
                "upload_session_id": upload_id,
                "source_bundle_id": "",          # Set via bundle API (future)
                "related_chunk_ids": [],          # Populated by ChunkLinker
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
            }
            for c in chunks
        ]

        vs.upsert_chunks(chunk_ids, embeddings, payloads)
        logger.info("[%s] Upserted %d chunks to Qdrant.", upload_id[:8], len(chunk_ids))

        # ── Step 6: Rebuild BM25 index ────────────────────────────────────
        # Fetch all existing payloads from Qdrant to build global BM25
        bm25_store = get_bm25_store()
        all_points = []
        offset = None
        while True:
            result_batch, next_offset = vs.client.scroll(
                collection_name=vs.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(result_batch)
            if next_offset is None:
                break
            offset = next_offset

        all_ids = [str(p.id) for p in all_points]
        all_texts = [p.payload.get("text", "") for p in all_points]
        all_meta = [p.payload for p in all_points]

        bm25_store.build(all_ids, all_texts, all_meta)
        bm25_store.save()
        logger.info("[%s] BM25 index rebuilt: %d chunks.", upload_id[:8], len(all_ids))

        # ── Step 7: Cross-modal chunk linking ─────────────────────────────
        try:
            from app.api.dependencies import get_chunk_linker
            linker = get_chunk_linker()
            link_result = linker.link_session(upload_id)
            logger.info(
                "[%s] Linking result: %s — %d links created.",
                upload_id[:8],
                link_result.get("status", "unknown"),
                link_result.get("total_links_created", 0),
            )
        except Exception as link_err:
            logger.warning(
                "[%s] Linking failed (non-fatal): %s", upload_id[:8], link_err,
            )

        # ── Step 8: CLIP visual embedding (images only) ──────────────────
        clip_status = "skipped"
        if modality == "image":
            try:
                from app.api.dependencies import get_clip_encoder, get_image_visual_store
                clip_enc = get_clip_encoder()
                if clip_enc.is_available:
                    clip_embedding = clip_enc.encode_image(file_path)
                    if clip_embedding is not None:
                        iv_store = get_image_visual_store()
                        iv_store.ensure_collection()
                        iv_store.upsert_image(
                            image_id=upload_id,
                            embedding=clip_embedding,
                            payload={
                                "source": original_filename,
                                "modality": "image",
                                "upload_id": upload_id,
                                "department": department or "",
                                "tags": tags or [],
                                "file_path": file_path,
                            },
                        )
                        clip_status = "completed"
                        logger.info("[%s] CLIP visual embedding stored.", upload_id[:8])
                    else:
                        clip_status = "encoding_failed"
                        logger.warning("[%s] CLIP encoding returned None.", upload_id[:8])
                else:
                    clip_status = "model_unavailable"
                    logger.info("[%s] CLIP model not available, skipping visual embedding.", upload_id[:8])
            except Exception as clip_err:
                clip_status = "error"
                logger.warning(
                    "[%s] CLIP visual embedding failed (non-fatal): %s", upload_id[:8], clip_err,
                )

        # Mark task as completed
        _task_status[upload_id] = {
            "status": "completed",
            "filename": original_filename,
            "modality": modality,
            "chunks": len(chunks),
            "clip_visual": clip_status,
        }
        logger.info(
            "[%s] Ingestion complete: %s → %d chunks indexed.",
            upload_id[:8], original_filename, len(chunks),
        )

    except Exception as e:
        logger.error("[%s] Ingestion failed: %s\n%s", upload_id[:8], e, traceback.format_exc())
        _task_status[upload_id] = {
            "status": "failed",
            "error": str(e),
            "filename": original_filename,
            "modality": modality,
        }


# ── Query Endpoint ────────────────────────────────────────────────────────────


@router.post("/query")
async def query(request: QueryRequest):
    """
    Submit a question and receive a response with citations.

    For SSE streaming, use Accept: text/event-stream header.
    If debug=true, retrieval debug info is streamed before generation.
    """
    from app.api.dependencies import (
        get_analytics_tracker,
        get_cost_tracker,
        get_embedding_model,
        get_hybrid_retriever,
        get_llm_engine,
        get_metrics,
        get_query_cache,
        get_query_store,
    )
    from app.generation.prompt_templates import ChunkContext, build_prompt
    from app.processing.normalization import normalize_query
    from app.retrieval.confidence import compute_confidence
    from app.retrieval.hallucination import detect_hallucination

    query_text = normalize_query(request.query)
    query_id = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()

    logger.info("[%s] Query: %s (debug=%s)", query_id, query_text[:100], request.debug)

    m = get_metrics()
    m.queries_total.inc()

    # ── Cache lookup (skip retrieval + LLM if hit) ───────────────────
    cache = get_query_cache()
    cached = cache.get(query_text) if not request.debug else None
    if cached:
        import asyncio

        async def cached_stream():
            yield 'data: {"type": "status", "content": "generating"}\n\n'
            for word in cached.answer.split(" "):
                escaped = json.dumps(word + " ")
                yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"
                await asyncio.sleep(0.005)
            for cit in cached.citations:
                yield f"data: {json.dumps({**cit, 'type': 'citation'})}\n\n"
            # Send confidence + cost metadata
            meta = {"type": "meta", "confidence": cached.confidence, "cost": cached.cost, "cached": True}
            yield f"data: {json.dumps(meta)}\n\n"
            yield 'data: {"type": "done"}\n\n'

        m.query_latency.observe(time.perf_counter() - t_start)
        logger.info("[%s] Cache HIT — skipping retrieval + LLM", query_id)
        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"X-Query-ID": query_id},
        )

    # Generate query embedding
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query_text)

    # Hybrid retrieval (with latency tracking)
    retriever = get_hybrid_retriever()
    modality_filter = request.filters.modality if request.filters else None
    department_filter = request.filters.department if request.filters else None

    # Apply runtime overrides to retrieval parameters
    from app.config.runtime_settings import get_overrides
    runtime_overrides = get_overrides()

    t_retrieval_start = time.perf_counter()
    retrieval_output = retriever.retrieve(
        query=query_text,
        query_embedding=query_embedding,
        modality_filter=modality_filter,
        department_filter=department_filter,
        debug=request.debug,
        overrides=runtime_overrides if runtime_overrides else None,
    )
    t_retrieval_end = time.perf_counter()
    retrieval_latency = t_retrieval_end - t_retrieval_start
    m.retrieval_latency.observe(retrieval_latency)

    # Unpack debug info if present
    debug_info = None
    if request.debug and isinstance(retrieval_output, tuple):
        results, debug_info = retrieval_output
    else:
        results = retrieval_output

    # Record analytics
    try:
        tracker = get_analytics_tracker()
        tracker.record_retrieval(results)
    except Exception:
        pass  # Analytics should never break queries

    if not results:
        # Record empty-result query
        _record_query_history(
            query_id=query_id, query_text=query_text, results=[], answer="(no results)",
            retrieval_latency=retrieval_latency, rerank_latency=0, generation_latency=0,
            t_start=t_start, debug_enabled=request.debug, debug_info=debug_info,
        )
        return QueryResponse(
            answer="I could not find relevant information in the available documents.",
            citations=[],
            query_id=query_id,
        )

    # Build context chunks  (cap at 5 for comprehensive answers)
    chunks = []
    chunk_upload_ids: list[str] = []  # Track upload_id for citation file_id
    for r in results[:5]:
        meta = r.get("metadata", {})
        if not meta:
            meta = r.get("payload", {})
        chunks.append(ChunkContext(
            chunk_id=r["chunk_id"],
            text=meta.get("text", ""),
            source=meta.get("source", "unknown"),
            modality=meta.get("modality", "document"),
            page_start=meta.get("page_start"),
            speaker=meta.get("speaker"),
            timestamp_start=meta.get("timestamp_start"),
            reranker_score=r.get("reranker_score", 0.0),
        ))
        chunk_upload_ids.append(meta.get("upload_id", ""))

    # Build prompt and generate
    llm = get_llm_engine()

    # If LLM is not loaded, return a non-streaming response with retrieved context
    if not llm.is_loaded:
        try:
            llm.load()
        except Exception as e:
            # Fallback: return just the retrieved text without LLM generation
            logger.warning("[%s] LLM not available: %s. Returning raw context.", query_id, e)
            context_summary = "\n\n".join(
                f"[{c.source}, Page {c.page_start}]: {c.text[:300]}..."
                if len(c.text) > 300 else f"[{c.source}, Page {c.page_start}]: {c.text}"
                for c in chunks[:5]
            )
            citations = [
                {"source": c.source, "page": c.page_start, "modality": c.modality}
                for c in chunks[:5]
            ]
            return QueryResponse(
                answer=f"(LLM unavailable — showing retrieved context)\n\n{context_summary}",
                citations=citations,
                query_id=query_id,
            )

    # --- Fast-path: if max_tokens == 0 skip LLM, stream context as SSE --------
    if request.max_tokens is not None and request.max_tokens == 0:
        import asyncio

        async def context_stream():
            yield 'data: {"type": "status", "content": "context-only mode (max_tokens=0)"}\n\n'
            summary = "Based on the retrieved documents:\n\n"
            for c in chunks:
                page = c.page_start or "N/A"
                summary += f"[{c.source}, Page {page}]: {c.text[:200]}...\n\n"
            for word in summary.split(" "):
                escaped = json.dumps(word + " ")
                yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"
                await asyncio.sleep(0.01)
            for idx, chunk in enumerate(chunks):
                citation_data = json.dumps({
                    "type": "citation",
                    "source": chunk.source,
                    "page": chunk.page_start,
                    "speaker": chunk.speaker,
                    "modality": chunk.modality,
                    "file_id": chunk_upload_ids[idx] if idx < len(chunk_upload_ids) else None,
                    "timestamp_start": chunk.timestamp_start,
                })
                yield f"data: {citation_data}\n\n"
            yield 'data: {"type": "done"}\n\n'

        return StreamingResponse(
            context_stream(),
            media_type="text/event-stream",
            headers={"X-Query-ID": query_id},
        )

    # Context token budget = n_ctx - max_generation_tokens - safety margin
    # This ensures the built prompt always leaves room for the answer
    _llm_cfg = get_settings().llm
    _max_gen = request.max_tokens or _llm_cfg.max_new_tokens
    _context_budget = _llm_cfg.context_window - _max_gen - 64  # 64 tok safety

    prompt = build_prompt(
        query=query_text,
        chunks=chunks,
        max_context_tokens=_context_budget,
        token_counter=llm.count_tokens,
    )

    # SSE streaming response — run sync LLM in threadpool
    import asyncio
    import queue
    import threading

    async def event_generator():
        # ── Send debug info first (if requested) ─────────────────────
        if debug_info:
            def _debug_result(item: dict, stage: str) -> dict:
                """Compact a retrieval result for debug output with origin attribution."""
                meta = item.get("metadata", item.get("payload", {}))
                return {
                    "chunk_id": str(item.get("chunk_id", ""))[:12],
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page_start"),
                    "score": round(item.get("score", item.get("rrf_score", item.get("reranker_score", 0))), 4),
                    "rank": item.get("rrf_rank", item.get("rank")),
                    "reranker_score": round(item.get("reranker_score", 0), 4) if "reranker_score" in item else None,
                    "stage": stage,
                    "origin": item.get("origin"),
                    "vector_rank": item.get("vector_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                }

            debug_payload = {
                "type": "debug",
                "retrieval_mode": "text",
                "vector_results": [_debug_result(r, "vector") for r in debug_info.get("vector_results", [])],
                "bm25_results": [_debug_result(r, "bm25") for r in debug_info.get("bm25_results", [])],
                "rrf_fused": [_debug_result(r, "rrf") for r in debug_info.get("rrf_fused", [])],
                "reranked": [_debug_result(r, "reranked") for r in debug_info.get("reranked", [])],
                "dropped": [_debug_result(r, "dropped") for r in debug_info.get("dropped", [])],
                "effective_params": debug_info.get("effective_params"),
                "entity_info": debug_info.get("entity_info"),
                "image_branch_info": debug_info.get("image_branch_info"),
            }
            yield f"data: {json.dumps(debug_payload)}\n\n"

        # Immediately send a heartbeat so the client knows data is flowing
        yield 'data: {"type": "status", "content": "generating"}\n\n'

        # Run the sync LLM generator in a background thread,
        # pushing tokens into a queue for the async generator.
        token_queue: queue.Queue = queue.Queue()
        sentinel = object()

        def _generate():
            try:
                for token in llm.generate_stream(prompt, max_tokens=request.max_tokens):
                    token_queue.put(token)
            except Exception as exc:
                token_queue.put(exc)
            finally:
                token_queue.put(sentinel)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        answer_tokens: list[str] = []
        t_gen_start = time.perf_counter()

        while True:
            # poll the queue without blocking the event loop
            while True:
                try:
                    item = token_queue.get_nowait()
                    break
                except queue.Empty:
                    await asyncio.sleep(0.05)

            if item is sentinel:
                break
            if isinstance(item, Exception):
                yield f"data: {{\"type\": \"error\", \"content\": \"{item}\"}}\n\n"
                break

            answer_tokens.append(item)
            # Properly escape the token for JSON
            escaped = json.dumps(item)  # gives a quoted JSON string
            yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"

        t_gen_end = time.perf_counter()
        generation_latency = t_gen_end - t_gen_start
        m.generation_latency.observe(generation_latency)
        total_latency = time.perf_counter() - t_start
        m.query_latency.observe(total_latency)

        # ── Detect truncation (model hit token limit without natural stop) ──
        was_truncated = getattr(llm, '_last_finish_reason', None) == "length"
        if was_truncated:
            truncation_note = "\n\n---\n*[Response reached the generation limit and may be incomplete. Try a more specific question for a complete answer.]*"
            answer_tokens.append(truncation_note)
            escaped = json.dumps(truncation_note)
            yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"
            logger.warning("[%s] Answer truncated (finish_reason=length)", query_id)

        # ── Append summary fallback if model didn't produce one ──────
        current_answer = "".join(answer_tokens)
        if not was_truncated and "Summary:" not in current_answer and len(current_answer) > 200:
            # Extract the last meaningful sentence as a summary
            sentences = [s.strip() for s in current_answer.replace("\n", " ").split(". ") if len(s.strip()) > 20]
            if sentences:
                last_sentence = sentences[-1].rstrip(".")
                summary_block = f"\n\n**Summary:** {last_sentence}."
                answer_tokens.append(summary_block)
                escaped = json.dumps(summary_block)
                yield f"data: {{\"type\": \"token\", \"content\": {escaped}}}\n\n"

        final_answer = "".join(answer_tokens)

        # ── Confidence scoring ────────────────────────────────────────
        try:
            confidence_result = compute_confidence(results, chunks_used=len(chunks))
            confidence_dict = confidence_result.to_dict()
        except Exception as exc:
            logger.warning("[%s] Confidence scoring failed: %s", query_id, exc)
            confidence_dict = {"score": 0.0, "level": "low", "signals": {}, "source_count": 0, "grounding": "unavailable"}

        # ── Hallucination detection ───────────────────────────────────
        try:
            chunk_texts = [c.text for c in chunks]
            hallucination = detect_hallucination(final_answer, chunk_texts)
            hallucination_dict = hallucination.to_dict()
        except Exception as exc:
            logger.warning("[%s] Hallucination detection failed: %s", query_id, exc)
            hallucination_dict = {"grounded_ratio": 0.0, "risk_level": "unknown", "total_sentences": 0, "grounded_sentences": 0, "ungrounded_claims": []}

        # ── Cost tracking ─────────────────────────────────────────────
        try:
            prompt_tokens = llm.count_tokens(prompt)
            completion_tokens = len(answer_tokens)
            cost_tracker = get_cost_tracker()
            query_cost = cost_tracker.compute_query_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                retrieval_time_s=retrieval_latency,
                generation_time_s=generation_latency,
                total_time_s=total_latency,
            )
            cost_dict = query_cost.to_dict()
        except Exception as exc:
            logger.warning("[%s] Cost tracking failed: %s", query_id, exc)
            cost_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}

        # Send citations
        citations_list = []
        for idx, chunk in enumerate(chunks):
            cit = {
                "source": chunk.source,
                "page": chunk.page_start,
                "speaker": chunk.speaker,
                "modality": chunk.modality,
                "file_id": chunk_upload_ids[idx] if idx < len(chunk_upload_ids) else None,
                "timestamp_start": chunk.timestamp_start,
            }
            citations_list.append(cit)
            citation_data = json.dumps({**cit, "type": "citation"})
            yield f"data: {citation_data}\n\n"

        # ── Send meta event (confidence + cost + hallucination) ────────
        meta_event = json.dumps({
            "type": "meta",
            "confidence": confidence_dict,
            "cost": cost_dict,
            "hallucination": hallucination_dict,
            "cached": False,
        })
        yield f"data: {meta_event}\n\n"

        # Record query to history store
        _record_query_history(
            query_id=query_id, query_text=query_text, results=results,
            answer=final_answer, retrieval_latency=retrieval_latency,
            rerank_latency=0, generation_latency=generation_latency,
            t_start=t_start, debug_enabled=request.debug, debug_info=debug_info,
        )

        # ── Store in cache for future hits ────────────────────────────
        try:
            cache.put(query_text, results[:5], final_answer, citations_list, confidence_dict, cost_dict)
        except Exception as exc:
            logger.warning("[%s] Cache store failed: %s", query_id, exc)

        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Query-ID": query_id},
    )


# ── Health Check Endpoint ─────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check.

    Returns status of all subsystems:
    - Qdrant connection
    - BM25 index
    - LLM readiness
    - Corpus size
    - Uptime
    """
    from app.api.dependencies import (
        get_bm25_store, get_llm_engine, get_vector_store, get_model_manager,
    )

    # Qdrant
    try:
        vs = get_vector_store()
        qdrant_health = vs.health_check()
        qdrant_status = qdrant_health.get("status", "error")
        corpus_size = qdrant_health.get("chunk_count", 0)
    except Exception:
        qdrant_status = "error"
        corpus_size = 0

    # BM25
    try:
        bm25 = get_bm25_store()
        bm25_status = "loaded" if bm25.is_loaded else "not_loaded"
    except Exception:
        bm25_status = "error"

    # LLM
    try:
        llm = get_llm_engine()
        llm_status = "ready" if llm.is_loaded else "not_loaded"
    except Exception:
        llm_status = "error"

    # ML models (embeddings, reranker, clip, whisper) via ModelManager
    embeddings_status = "not_loaded"
    reranker_status = "not_loaded"
    clip_status = "not_loaded"
    whisper_status = "not_loaded"
    try:
        mm = get_model_manager()
        model_statuses = mm.health_status()
        embeddings_status = model_statuses.get("embedding", "not_loaded")
        reranker_status = model_statuses.get("reranker", "not_loaded")
        clip_status = model_statuses.get("clip_visual", "not_loaded")
        whisper_status = model_statuses.get("whisper", "not_loaded")
    except Exception:
        pass

    # Uptime
    uptime_seconds = int(time.time() - _start_time)
    hours = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60
    uptime_str = f"{hours}h{minutes}m"

    return HealthResponse(
        status="healthy" if qdrant_status == "connected" else "degraded",
        qdrant=qdrant_status,
        bm25=bm25_status,
        llm=llm_status,
        embeddings=embeddings_status,
        reranker=reranker_status,
        clip=clip_status,
        whisper=whisper_status,
        corpus_size=corpus_size,
        uptime=uptime_str,
    )


# ── Status Endpoint ───────────────────────────────────────────────────────────


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Check the processing status of an upload."""
    task = _task_status.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return TaskStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        filename=task.get("filename"),
        modality=task.get("modality"),
        error=task.get("error"),
    )


# ── File Serving Endpoint (Citation Navigation) ──────────────────────────────


@router.get("/files/{file_id}")
async def serve_file(file_id: str):
    """
    Serve an uploaded file for citation navigation.

    Supports:
    - PDF: byte-range streaming via Range header
    - Audio: streaming with Range support (seek)
    - Image: direct serve
    - Other documents: download

    Security: only files tracked in the registry can be served.
    Path traversal is blocked by ID-based lookup (no user-supplied paths).
    """
    from fastapi import Request
    from fastapi.responses import FileResponse
    from starlette.responses import Response

    from app.api.dependencies import get_file_registry

    registry = get_file_registry()
    entry = registry.get(file_id)

    if not entry:
        raise HTTPException(status_code=404, detail="File not found in registry.")

    file_path = Path(entry["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File no longer exists on disk.")

    mime_type = entry.get("file_type", "application/octet-stream")
    file_name = entry.get("file_name", file_path.name)

    return FileResponse(
        path=str(file_path),
        media_type=mime_type,
        filename=file_name,
        headers={
            "Accept-Ranges": "bytes",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/files/{file_id}/info")
async def file_info(file_id: str):
    """Return metadata about a registered file (without serving content)."""
    from app.api.dependencies import get_file_registry

    registry = get_file_registry()
    entry = registry.get(file_id)

    if not entry:
        raise HTTPException(status_code=404, detail="File not found in registry.")

    return {
        "file_id": entry["file_id"],
        "file_name": entry["file_name"],
        "file_type": entry["file_type"],
        "modality": entry["modality"],
        "file_size": entry["file_size"],
        "upload_time": entry["upload_time"],
    }


@router.get("/files")
async def list_files():
    """List all registered files."""
    from app.api.dependencies import get_file_registry

    registry = get_file_registry()
    entries = registry.list_all()

    return {
        "files": [
            {
                "file_id": e["file_id"],
                "file_name": e["file_name"],
                "file_type": e["file_type"],
                "modality": e["modality"],
                "file_size": e["file_size"],
                "upload_time": e["upload_time"],
            }
            for e in entries
        ],
        "total": len(entries),
    }


# ── Knowledge Base Endpoints ──────────────────────────────────────────────────


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all unique source documents in the vector database.

    Returns aggregated metadata (chunk count, token count, modality, etc.)
    for every document that has been ingested.
    """
    from app.api.dependencies import get_vector_store

    vs = get_vector_store()
    docs_raw = vs.list_documents()

    documents = [
        DocumentSummary(
            source=d["source"],
            modality=d.get("modality", "document"),
            department=d.get("department", ""),
            tags=d.get("tags", []),
            chunk_count=d.get("chunk_count", 0),
            total_tokens=d.get("total_tokens", 0),
            upload_id=d.get("upload_id", ""),
        )
        for d in docs_raw
    ]

    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/documents/{source:path}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(source: str):
    """
    Get all chunks for a specific source document.

    Returns chunk text, index, page number, and token count — useful for
    inspecting how a document was split.
    """
    from app.api.dependencies import get_vector_store

    vs = get_vector_store()
    chunks_raw = vs.get_chunks_by_source(source)

    if not chunks_raw:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for source '{source}'.",
        )

    chunks = [
        ChunkDetail(
            chunk_id=c["chunk_id"],
            text=c["text"],
            chunk_index=c.get("chunk_index", 0),
            page_start=c.get("page_start"),
            token_count=c.get("token_count", 0),
        )
        for c in chunks_raw
    ]

    return DocumentChunksResponse(
        source=source,
        chunks=chunks,
        total_chunks=len(chunks),
    )


@router.delete("/documents/{source:path}", response_model=DeleteDocumentResponse)
async def delete_document(source: str, background_tasks: BackgroundTasks):
    """
    Delete all chunks for a source document from Qdrant and rebuild BM25.

    This is irreversible — the original file must be re-uploaded to restore.
    """
    from app.api.dependencies import get_analytics_tracker, get_bm25_store, get_vector_store

    vs = get_vector_store()

    # Delete from Qdrant
    deleted_count = vs.delete_by_source(source)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for source '{source}'.",
        )

    # Clean up analytics for this document
    try:
        tracker = get_analytics_tracker()
        tracker.delete_source(source)
    except Exception:
        pass

    # Rebuild BM25 in the background so the response is fast
    background_tasks.add_task(_rebuild_bm25_index)

    logger.info("Deleted document '%s': %d chunks removed.", source, deleted_count)
    return DeleteDocumentResponse(
        source=source,
        deleted_chunks=deleted_count,
        status="deleted",
    )


def _rebuild_bm25_index() -> None:
    """Rebuild the global BM25 index from current Qdrant contents."""
    from app.api.dependencies import get_bm25_store, get_vector_store

    vs = get_vector_store()
    bm25_store = get_bm25_store()

    all_points = []
    offset = None
    while True:
        result_batch, next_offset = vs.client.scroll(
            collection_name=vs.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(result_batch)
        if next_offset is None:
            break
        offset = next_offset

    all_ids = [str(p.id) for p in all_points]
    all_texts = [p.payload.get("text", "") for p in all_points]
    all_meta = [p.payload for p in all_points]

    bm25_store.build(all_ids, all_texts, all_meta)
    bm25_store.save()
    logger.info("BM25 index rebuilt after document deletion: %d chunks.", len(all_ids))


# ── Index Health Endpoint ─────────────────────────────────────────────────────


@router.get("/index/health", response_model=IndexHealthResponse)
async def index_health():
    """
    Detailed index health metrics for the System page.

    Returns:
        Total chunks, avg tokens, largest document, BM25 vocab size, etc.
    """
    from app.api.dependencies import get_bm25_store, get_vector_store

    vs = get_vector_store()
    settings = get_settings()

    # Get all documents from Qdrant with their stats
    docs_raw = vs.list_documents()

    total_chunks = sum(d["chunk_count"] for d in docs_raw)
    total_tokens = sum(d["total_tokens"] for d in docs_raw)
    avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0.0

    largest_doc = ""
    largest_doc_chunks = 0
    for d in docs_raw:
        if d["chunk_count"] > largest_doc_chunks:
            largest_doc = d["source"]
            largest_doc_chunks = d["chunk_count"]

    # BM25 stats
    bm25 = get_bm25_store()
    bm25_chunk_count = bm25.chunk_count
    bm25_vocab_size = 0
    if bm25.is_loaded and bm25._bm25 is not None:
        try:
            bm25_vocab_size = len(bm25._bm25.idf)
        except Exception:
            pass

    return IndexHealthResponse(
        total_chunks=total_chunks,
        total_documents=len(docs_raw),
        avg_tokens_per_chunk=round(avg_tokens, 1),
        largest_document=largest_doc,
        largest_document_chunks=largest_doc_chunks,
        embedding_dimension=settings.qdrant.vector_size,
        qdrant_collection=settings.qdrant.collection_name,
        bm25_chunk_count=bm25_chunk_count,
        bm25_vocab_size=bm25_vocab_size,
        total_tokens=total_tokens,
    )


# ── Analytics Endpoint ────────────────────────────────────────────────────────


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """
    Per-document retrieval analytics.

    Returns retrieval count, last queried time, avg reranker score,
    and avg rank position for each source document.
    """
    from app.api.dependencies import get_analytics_tracker

    tracker = get_analytics_tracker()
    all_stats = tracker.get_stats()

    documents = [
        DocumentAnalytics(
            source=src,
            retrieval_count=stats.get("retrieval_count", 0),
            last_queried=stats.get("last_queried", 0),
            avg_reranker_score=round(stats.get("avg_reranker_score", 0), 4),
            avg_rank_position=round(stats.get("avg_rank_position", 0), 1),
        )
        for src, stats in all_stats.items()
    ]

    return AnalyticsResponse(documents=documents)


# ── Re-Index Endpoint ─────────────────────────────────────────────────────────


@router.post("/documents/{source:path}/reindex", response_model=ReindexResponse)
async def reindex_document(source: str, background_tasks: BackgroundTasks):
    """
    Re-index a document: delete existing chunks, re-parse, re-chunk, re-embed.

    The original uploaded file must still exist in the uploads directory.
    """
    from app.api.dependencies import get_vector_store

    settings = get_settings()
    vs = get_vector_store()

    # Find the upload file by matching source in payloads
    chunks = vs.get_chunks_by_source(source)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source}'.")

    # Find the upload file on disk
    uploads_dir = Path(settings.paths.uploads_dir)
    matching_files = list(uploads_dir.glob(f"*_{source}"))
    if not matching_files:
        # Try exact match
        matching_files = [f for f in uploads_dir.iterdir() if f.name.endswith(source)]

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail=f"Upload file for '{source}' not found on disk. Re-upload required.",
        )

    upload_file = matching_files[0]

    # Delete existing chunks
    deleted = vs.delete_by_source(source)
    logger.info("Re-index '%s': deleted %d existing chunks.", source, deleted)

    # Re-run ingestion in background
    upload_id = uuid.uuid4().hex
    _task_status[upload_id] = {
        "status": "processing",
        "filename": source,
        "modality": "document",
    }

    background_tasks.add_task(
        _ingest_file,
        upload_id=upload_id,
        file_path=str(upload_file),
        original_filename=source,
        modality="document",
        department=None,
        tags=[],
    )

    return ReindexResponse(
        source=source,
        status="reindexing",
        message=f"Deleted {deleted} chunks. Re-ingestion started (task: {upload_id[:8]}).",
    )


# ── Helper: Record Query History ──────────────────────────────────────────────


def _record_query_history(
    query_id: str,
    query_text: str,
    results: list,
    answer: str,
    retrieval_latency: float,
    rerank_latency: float,
    generation_latency: float,
    t_start: float,
    debug_enabled: bool = False,
    debug_info: dict | None = None,
) -> None:
    """Record a query to the persistent query history store."""
    try:
        from app.api.dependencies import get_query_store
        from app.observability.query_store import QueryRecord

        store = get_query_store()

        chunks_summary = []
        for r in results[:10]:
            meta = r.get("metadata", r.get("payload", {}))
            chunks_summary.append({
                "chunk_id": str(r.get("chunk_id", ""))[:12],
                "source": meta.get("source", "unknown"),
                "score": round(r.get("reranker_score", r.get("rrf_score", 0)), 4),
            })

        record = QueryRecord(
            query_id=query_id,
            query=query_text,
            retrieved_chunks=chunks_summary,
            chunk_count=len(results),
            answer=answer[:2000],  # Truncate very long answers
            retrieval_latency=round(retrieval_latency, 4),
            rerank_latency=round(rerank_latency, 4),
            generation_latency=round(generation_latency, 4),
            total_latency=round(time.perf_counter() - t_start, 4),
            debug_enabled=debug_enabled,
            debug_info=debug_info,
            # Persist survival log when available (debug mode)
            survival_log=debug_info.get("survival_log") if debug_info else None,
        )
        store.record(record)
    except Exception as e:
        logger.warning("Failed to record query history: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# ── Query History Endpoints ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/queries", response_model=QueryHistoryResponse)
async def list_queries(limit: int = 50, offset: int = 0):
    """
    Paginated list of past queries (newest first).
    """
    from app.api.dependencies import get_query_store

    store = get_query_store()
    records, total = store.get_all(limit=limit, offset=offset)

    items = [
        QueryHistoryItem(
            query_id=r["query_id"],
            query=r["query"],
            timestamp=r.get("timestamp", 0),
            chunk_count=r.get("chunk_count", 0),
            answer=r.get("answer", "")[:300],  # Short preview
            retrieval_latency=r.get("retrieval_latency", 0),
            rerank_latency=r.get("rerank_latency", 0),
            generation_latency=r.get("generation_latency", 0),
            total_latency=r.get("total_latency", 0),
            debug_enabled=r.get("debug_enabled", False),
            error=r.get("error"),
            token_count=r.get("token_count", 0),
        )
        for r in records
    ]

    return QueryHistoryResponse(queries=items, total=total)


@router.get("/queries/summary", response_model=QuerySummaryResponse)
async def query_summary():
    """Aggregate query performance statistics."""
    from app.api.dependencies import get_query_store

    store = get_query_store()
    summary = store.get_summary()
    return QuerySummaryResponse(**summary)


@router.delete("/queries")
async def delete_all_queries():
    """Delete all query history records."""
    from app.api.dependencies import get_query_store

    store = get_query_store()
    count = store.delete_all()
    return {"deleted": count, "message": f"Deleted {count} query records."}


@router.delete("/queries/{query_id}")
async def delete_query(query_id: str):
    """Delete a specific query record."""
    from app.api.dependencies import get_query_store

    store = get_query_store()
    success = store.delete_by_id(query_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")
    return {"deleted": True, "query_id": query_id}


@router.get("/queries/{query_id}", response_model=QueryDetailResponse)
async def get_query_detail(query_id: str):
    """Full detail for a specific past query including retrieved chunks."""
    from app.api.dependencies import get_query_store

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    return QueryDetailResponse(**{
        k: v for k, v in record.items()
        if k in QueryDetailResponse.__fields__
    })


@router.post("/queries/{query_id}/replay")
async def replay_query(query_id: str):
    """
    Replay a past query: re-run the same query text through the pipeline.
    Returns the query_id of the new run (clients should call POST /query).
    """
    from app.api.dependencies import get_query_store

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    return {"original_query": record["query"], "instruction": "POST /query with this text"}


# ══════════════════════════════════════════════════════════════════════════════
# ── Index Version Control Endpoints ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def _get_dir_size_mb(path: Path) -> float:
    """Calculate directory size in MB."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass
    return round(total / (1024 * 1024), 2)


@router.get("/versions", response_model=VersionListResponse)
async def list_versions():
    """List all index versions with their sizes."""
    from app.api.dependencies import get_index_manager

    mgr = get_index_manager()
    current = mgr.current_version
    versions = []

    for v in mgr.list_versions():
        version_dir = mgr._root / v
        size_mb = _get_dir_size_mb(version_dir)
        versions.append(VersionInfo(
            version=v,
            is_active=(v == current),
            size_mb=size_mb,
        ))

    return VersionListResponse(versions=versions, current_version=current)


@router.post("/versions/{version}/switch", response_model=VersionSwitchResponse)
async def switch_version(version: str):
    """Switch the active index to a different version."""
    from app.api.dependencies import get_index_manager

    mgr = get_index_manager()
    try:
        mgr.switch_to(version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return VersionSwitchResponse(
        version=version,
        status="switched",
        message=f"Active index switched to {version}. Restart services to apply.",
    )


@router.delete("/versions/{version}")
async def delete_version(version: str):
    """Delete an index version (cannot delete the active version)."""
    from app.api.dependencies import get_index_manager

    mgr = get_index_manager()
    try:
        mgr.delete_version(version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    return {"version": version, "status": "deleted"}


# ══════════════════════════════════════════════════════════════════════════════
# ── Metrics Endpoint ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics_snapshot():
    """Per-stage latency metrics and counters for the Metrics Dashboard."""
    from app.api.dependencies import get_metrics

    m = get_metrics()

    def _breakdown(h) -> LatencyBreakdown:
        return LatencyBreakdown(
            avg=round(h.avg, 4),
            p50=round(h.p50(), 4),
            p95=round(h.p95(), 4),
            count=h.count,
        )

    return MetricsResponse(
        uploads_total=int(m.uploads_total.value),
        upload_errors=int(m.upload_errors.value),
        queries_total=int(m.queries_total.value),
        query_latency=_breakdown(m.query_latency),
        retrieval_latency=_breakdown(m.retrieval_latency),
        rerank_latency=_breakdown(m.rerank_latency),
        generation_latency=_breakdown(m.generation_latency),
        corpus_size=int(m.corpus_size.value),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Cache Stats + Cost Summary Endpoint ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/cache/stats")
async def get_cache_stats():
    """Return query cache hit/miss stats and cost summary."""
    from app.api.dependencies import get_query_cache, get_cost_tracker

    cache = get_query_cache()
    cost_tracker = get_cost_tracker()
    return {
        "cache": cache.stats(),
        "cost": cost_tracker.summary(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Business Impact Endpoint ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/business-impact")
async def get_business_impact():
    """
    Business impact dashboard: ROI, time saved, cost efficiency, hallucination rate.
    Produces the Before→After delta table judges want to see.
    """
    from app.api.dependencies import get_cost_tracker, get_metrics
    from app.observability.business_impact import estimate_business_impact

    m = get_metrics()
    cost_tracker = get_cost_tracker()

    cost_summary = cost_tracker.summary()
    avg_response_time = m.query_latency.avg if m.query_latency.count > 0 else 0

    impact = estimate_business_impact(
        avg_response_time_s=avg_response_time,
        compute_cost_per_1k_usd=cost_summary.get("cost_per_1k_queries_usd", 0),
    )

    return {
        "impact": impact.to_dict(),
        "system_metrics": {
            "total_queries": int(m.queries_total.value),
            "avg_latency_s": round(avg_response_time, 2),
            "p50_latency_s": round(m.query_latency.p50(), 2),
            "p95_latency_s": round(m.query_latency.p95(), 2),
            "cache_hit_rate": cost_summary.get("cache_hit_rate", 0),
        },
        "cost": cost_summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Resource Monitor Endpoint ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/resources", response_model=ResourceStatus)
async def get_resources():
    """Live system resource usage (CPU, RAM, GPU, Disk)."""
    import psutil

    # CPU + RAM
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("D:\\")

    result = ResourceStatus(
        cpu_percent=cpu_percent,
        ram_used_mb=round(ram.used / (1024 ** 2), 1),
        ram_total_mb=round(ram.total / (1024 ** 2), 1),
        ram_percent=ram.percent,
        disk_used_gb=round(disk.used / (1024 ** 3), 1),
        disk_total_gb=round(disk.total / (1024 ** 3), 1),
        disk_percent=round(disk.percent, 1),
    )

    # GPU (nvidia-smi via subprocess — reliable on Windows)
    try:
        import subprocess
        nvidia_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        parts = [p.strip() for p in nvidia_output.split(",")]
        if len(parts) >= 4:
            result.gpu_name = parts[0]
            result.gpu_memory_used_mb = float(parts[1])
            result.gpu_memory_total_mb = float(parts[2])
            result.gpu_utilization = float(parts[3])
            if result.gpu_memory_total_mb > 0:
                result.gpu_memory_percent = round(
                    result.gpu_memory_used_mb / result.gpu_memory_total_mb * 100, 1
                )
    except Exception:
        pass  # No GPU or nvidia-smi not available

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ── Runtime Retrieval Settings Endpoints ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/settings/retrieval", response_model=RetrievalSettingsResponse)
async def get_retrieval_settings():
    """
    Get current effective retrieval parameters (base config + runtime overrides).
    """
    from app.config.runtime_settings import get_effective_retrieval_params, get_overrides

    effective = get_effective_retrieval_params()
    overrides = get_overrides()

    return RetrievalSettingsResponse(
        vector_top_k=effective["vector_top_k"],
        bm25_top_k=effective["bm25_top_k"],
        rrf_k=effective["rrf_k"],
        rerank_count=effective["rerank_count"],
        rerank_threshold=effective["rerank_threshold"],
        rerank_min_results=effective["rerank_min_results"],
        overrides=overrides,
    )


@router.patch("/settings/retrieval", response_model=RetrievalSettingsResponse)
async def update_retrieval_settings(body: RetrievalSettingsRequest):
    """
    Patch runtime retrieval parameters.  Only non-null fields are applied.

    Changes take effect on the next query — no restart needed.
    """
    from app.config.runtime_settings import (
        get_effective_retrieval_params,
        get_overrides,
        set_overrides,
    )

    # Build patch dict from non-null fields
    patch = {k: v for k, v in body.dict().items() if v is not None}

    if patch:
        errors = set_overrides(patch)
        if errors:
            raise HTTPException(status_code=422, detail={"validation_errors": errors})

    effective = get_effective_retrieval_params()
    overrides = get_overrides()

    return RetrievalSettingsResponse(
        vector_top_k=effective["vector_top_k"],
        bm25_top_k=effective["bm25_top_k"],
        rrf_k=effective["rrf_k"],
        rerank_count=effective["rerank_count"],
        rerank_threshold=effective["rerank_threshold"],
        rerank_min_results=effective["rerank_min_results"],
        overrides=overrides,
    )


@router.delete("/settings/retrieval")
async def reset_retrieval_settings():
    """Reset all runtime overrides to config.yaml defaults."""
    from app.config.runtime_settings import reset_overrides

    reset_overrides()
    return {"status": "reset", "message": "All runtime overrides cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# ── Recall Validation Endpoints ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/queries/{query_id}/validate", response_model=RecallMetrics)
async def validate_query_recall(query_id: str, body: RecallValidationRequest):
    """
    Annotate retrieved chunks as relevant / irrelevant for a past query.

    Computes recall@5, recall@10, and MRR from the annotations and stores
    them alongside the query record.
    """
    from app.api.dependencies import get_query_store

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    # Build annotation lookup
    annotation_map = {a.chunk_id: a.relevant for a in body.annotations}
    retrieved_ids = [c.get("chunk_id", "") for c in record.get("retrieved_chunks", [])]

    # Compute recall@k
    relevant_set = {cid for cid, rel in annotation_map.items() if rel}
    annotated_count = len(annotation_map)
    relevant_count = len(relevant_set)

    def recall_at(k: int) -> float | None:
        if not relevant_set:
            return None
        top_k_ids = set(retrieved_ids[:k])
        return round(len(relevant_set & top_k_ids) / len(relevant_set), 4)

    # MRR: reciprocal of the rank of the first relevant chunk
    mrr: float | None = None
    if relevant_set:
        for rank_idx, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant_set:
                mrr = round(1.0 / rank_idx, 4)
                break
        if mrr is None:
            mrr = 0.0

    # Persist annotations into the query record
    store.annotate(query_id, {
        "recall_validation": {
            "annotations": [a.dict() for a in body.annotations],
            "recall_at_5": recall_at(5),
            "recall_at_10": recall_at(10),
            "mrr": mrr,
            "relevant_count": relevant_count,
            "annotated_count": annotated_count,
        }
    })

    return RecallMetrics(
        query_id=query_id,
        total_retrieved=len(retrieved_ids),
        annotated_count=annotated_count,
        relevant_count=relevant_count,
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        mrr=mrr,
        annotations=body.annotations,
    )


@router.get("/queries/{query_id}/recall", response_model=RecallMetrics)
async def get_query_recall(query_id: str):
    """
    Get recall validation results for a past query (if annotations exist).
    """
    from app.api.dependencies import get_query_store

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    rv = record.get("recall_validation")
    if rv is None:
        raise HTTPException(status_code=404, detail="No recall validation found for this query.")

    retrieved_ids = [c.get("chunk_id", "") for c in record.get("retrieved_chunks", [])]

    return RecallMetrics(
        query_id=query_id,
        total_retrieved=len(retrieved_ids),
        annotated_count=rv.get("annotated_count", 0),
        relevant_count=rv.get("relevant_count", 0),
        recall_at_5=rv.get("recall_at_5"),
        recall_at_10=rv.get("recall_at_10"),
        mrr=rv.get("mrr"),
        annotations=[],  # Don't re-send all annotations on GET
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Survival Tracking ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/queries/{query_id}/survival", response_model=SurvivalResponse)
async def get_query_survival(query_id: str):
    """
    Get stage-wise survival log for a past query.

    If the query was run with debug=True, the survival log is already stored.
    Otherwise, returns an empty log with instructions.
    """
    from app.api.dependencies import get_query_store
    from app.observability.survival_tracker import compute_survival_summary

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    survival_log = record.get("survival_log", [])
    summary = None
    if survival_log:
        summary = compute_survival_summary(survival_log)

    return SurvivalResponse(
        query_id=query_id,
        query=record.get("query", ""),
        survival_log=survival_log,
        summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Failure Diagnosis ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/queries/{query_id}/diagnosis", response_model=DiagnosisResponse)
async def diagnose_query(query_id: str):
    """
    Run automated failure diagnosis on a past query.

    Analyzes survival log, recall validation, and debug info to classify
    the root cause of poor retrieval.
    """
    from app.api.dependencies import get_failure_diagnoser, get_query_store

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    diagnoser = get_failure_diagnoser()
    diagnosis = diagnoser.diagnose(record)

    # Persist diagnosis in the query record
    store.annotate(query_id, {"diagnosis": diagnosis.to_dict()})

    return DiagnosisResponse(
        query_id=query_id,
        root_cause=diagnosis.root_cause,
        confidence=diagnosis.confidence,
        evidence=diagnosis.evidence,
        recommendations=diagnosis.recommendations,
        secondary_causes=diagnosis.secondary_causes,
    )


@router.post("/diagnosis/batch", response_model=BatchDiagnosisResponse)
async def batch_diagnose(limit: int = 50):
    """
    Run failure diagnosis across recent queries.

    Returns aggregate cause distribution and common recommendations.
    """
    from app.api.dependencies import get_failure_diagnoser, get_query_store

    store = get_query_store()
    records, _ = store.get_all(limit=limit, offset=0)

    diagnoser = get_failure_diagnoser()
    result = diagnoser.diagnose_batch(records)

    return BatchDiagnosisResponse(**result)


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Ground Truth Tagging ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/queries/{query_id}/ground-truth")
async def set_ground_truth(query_id: str, body: GroundTruthRequest):
    """
    Tag a query with ground truth chunk IDs for evaluation purposes.

    These IDs represent the chunks that SHOULD have been retrieved.
    """
    from app.api.dependencies import get_query_store

    store = get_query_store()
    record = store.get_by_id(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")

    store.annotate(query_id, {
        "ground_truth_chunk_ids": body.ground_truth_chunk_ids,
    })

    return {
        "query_id": query_id,
        "ground_truth_count": len(body.ground_truth_chunk_ids),
        "status": "tagged",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Experiment Comparison ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/experiments/compare")
async def compare_experiments(body: CompareRequest):
    """
    Run side-by-side comparison: same query, two different parameter sets.

    Returns overlap analysis, unique chunks, rank differences, score diffs.
    """
    from app.api.dependencies import get_experiment_engine

    engine = get_experiment_engine()

    params_a = {k: v for k, v in body.params_a.dict().items() if v is not None}
    params_b = {k: v for k, v in body.params_b.dict().items() if v is not None}

    result = engine.compare(
        query=body.query,
        params_a=params_a,
        params_b=params_b,
        label_a=body.label_a,
        label_b=body.label_b,
    )

    return result


@router.post("/experiments/batch-evaluate")
async def batch_evaluate(body: BatchEvalRequest):
    """
    Run batch evaluation on a set of test queries.

    Computes aggregate metrics: recall@5, recall@10, MRR, latency stats.
    Queries with ground_truth_chunk_ids get recall/MRR computation.
    """
    from app.api.dependencies import get_experiment_engine

    engine = get_experiment_engine()

    test_queries = [q.dict() for q in body.queries]
    overrides = {k: v for k, v in body.overrides.dict().items() if v is not None} if body.overrides else None

    result = engine.batch_evaluate(
        test_queries=test_queries,
        overrides=overrides,
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Corpus Coverage ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/corpus/coverage", response_model=CorpusCoverageResponse)
async def get_corpus_coverage():
    """
    Analyze how thoroughly the corpus is utilized by queries.

    Returns per-chunk retrieval frequency, never-retrieved chunks,
    hotspot/coldspot analysis, and per-source breakdown.
    """
    from app.api.dependencies import get_corpus_coverage_analyzer

    analyzer = get_corpus_coverage_analyzer()
    result = analyzer.analyze()

    return CorpusCoverageResponse(**result)


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Embedding Quality ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/embeddings/quality", response_model=EmbeddingQualityResponse)
async def check_embedding_quality(sample_size: int = 200):
    """
    Run embedding quality diagnostics.

    Analyzes vector norms, cosine similarity distribution, and per-source stats.
    """
    from app.api.dependencies import get_embedding_quality_checker

    checker = get_embedding_quality_checker()
    result = checker.check(sample_size=sample_size)

    return EmbeddingQualityResponse(**result)


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Version Metadata ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/versions/{version}/metadata")
async def save_version_metadata(version: str, body: VersionMetadataRequest):
    """
    Save metadata alongside an index version (embedding model, chunk size, etc.).
    """
    from app.api.dependencies import get_index_manager

    mgr = get_index_manager()
    metadata = {k: v for k, v in body.dict().items() if v is not None}

    try:
        mgr.save_metadata(version, metadata)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    return {"version": version, "metadata": metadata, "status": "saved"}


@router.get("/versions/{version}/metadata", response_model=VersionDetailResponse)
async def get_version_metadata(version: str):
    """
    Get metadata for a specific index version.
    """
    from app.api.dependencies import get_index_manager

    mgr = get_index_manager()
    version_dir = mgr._root / version
    if not version_dir.exists():
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    metadata = mgr.get_metadata(version)

    # Compute size
    size_mb = 0.0
    try:
        for f in version_dir.rglob("*"):
            if f.is_file():
                size_mb += f.stat().st_size
        size_mb = round(size_mb / (1024 * 1024), 2)
    except Exception:
        pass

    return VersionDetailResponse(
        version=version,
        is_active=(version == mgr.current_version),
        size_mb=size_mb,
        metadata=metadata,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: Cross-Modal Linking ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/linking/distribution/{upload_id}")
async def linking_distribution(upload_id: str):
    """
    Analyze cosine similarity distribution for a single upload session.

    Returns percentile statistics for threshold calibration.
    Use this BEFORE committing to a similarity_threshold value.
    """
    from app.api.dependencies import get_chunk_linker

    linker = get_chunk_linker()
    result = linker.analyze_session_distribution(upload_id)

    if result.get("status") == "insufficient_chunks":
        raise HTTPException(
            status_code=400,
            detail=f"Session {upload_id[:8]} has fewer than 2 chunks.",
        )
    return result


@router.get("/linking/distribution")
async def linking_global_distribution(sample_limit: int = 500):
    """
    Analyze cosine similarity distribution across the entire corpus (sampled).

    Returns percentile statistics for global threshold calibration.
    """
    from app.api.dependencies import get_chunk_linker

    linker = get_chunk_linker()
    result = linker.analyze_global_distribution(sample_limit=sample_limit)

    if result.get("status") == "insufficient_chunks":
        raise HTTPException(status_code=400, detail="Corpus has fewer than 2 chunks.")
    return result


@router.post("/linking/run/{upload_id}")
async def run_linking(upload_id: str):
    """
    Manually trigger cross-modal linking for a specific upload session.

    Useful for re-linking after threshold adjustment.
    """
    from app.api.dependencies import get_chunk_linker

    linker = get_chunk_linker()
    result = linker.link_session(upload_id)
    return result


@router.get("/linking/status/{upload_id}")
async def linking_status(upload_id: str):
    """
    Check linking status for a specific upload session.

    Returns chunk count, how many have links, and link statistics.
    """
    from app.api.dependencies import get_vector_store

    vs = get_vector_store()
    chunks = vs.get_chunks_by_upload_id(upload_id)

    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks for upload {upload_id[:8]}.")

    total = len(chunks)
    with_links = 0
    total_links = 0
    for c in chunks:
        related = c["payload"].get("related_chunk_ids", [])
        if related:
            with_links += 1
            total_links += len(related)

    return {
        "upload_id": upload_id,
        "total_chunks": total,
        "chunks_with_links": with_links,
        "total_links": total_links,
        "avg_links_per_chunk": round(total_links / total, 2) if total > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab: CLIP Visual Search ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/clip/health")
async def clip_health():
    """
    Check CLIP subsystem health: model availability + collection status.
    """
    from app.api.dependencies import get_clip_encoder, get_image_visual_store

    clip_enc = get_clip_encoder()
    iv_store = get_image_visual_store()

    return {
        "clip_enabled": get_settings().clip.enabled,
        "model_available": clip_enc.is_available,
        "collection": iv_store.health_check(),
        "device": get_settings().clip.device,
    }


@router.post("/clip/search")
async def clip_visual_search(query: str, top_k: int = 5):
    """
    Direct CLIP visual search for testing. Returns image results ranked by
    CLIP similarity.  Does NOT go through the full hybrid retrieval pipeline.
    """
    from app.api.dependencies import get_clip_encoder, get_image_visual_store

    clip_enc = get_clip_encoder()
    if not clip_enc.is_available:
        raise HTTPException(status_code=503, detail="CLIP model not available.")

    iv_store = get_image_visual_store()

    clip_query_emb = clip_enc.encode_text(query)
    if clip_query_emb is None:
        raise HTTPException(status_code=500, detail="CLIP text encoding failed.")

    results = iv_store.search(query_vector=clip_query_emb, top_k=top_k)

    return {
        "query": query,
        "results": [
            {
                "image_id": r["chunk_id"],
                "score": round(r["score"], 4),
                "source": r["payload"].get("source", "unknown"),
                "upload_id": r["payload"].get("upload_id", ""),
            }
            for r in results
        ],
        "count": len(results),
    }


@router.get("/clip/intent")
async def clip_intent_check(query: str):
    """
    Test visual intent detection for a query.

    Returns whether the image branch would fire for this query.
    """
    from app.retrieval.visual_intent_detector import detect_visual_intent

    return detect_visual_intent(query)


# ══════════════════════════════════════════════════════════════════════════════
# ── Phase 4: Image-as-Query ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/query/image", response_model=ImageQueryResponse)
async def query_image(
    file: UploadFile = File(...),
    text_prompt: Optional[str] = Form(None),
    top_k: int = Form(10),
    department: Optional[str] = Form(None),
    debug: bool = Form(False),
):
    """
    Image-as-Query: upload an image and retrieve related content.

    Pipeline (Strategy A):
      Image → CLIP image encoder → search image_visual_embeddings
      → linked expansion to text chunks → rerank → response

    CLIP vectors NEVER search the BGE text collection directly.
    """
    from app.api.dependencies import get_image_query_retriever, get_metrics

    settings = get_settings()
    if not settings.clip.enabled or not settings.clip.enabled_for_query:
        raise HTTPException(
            status_code=503,
            detail="Image query is disabled. Set clip.enabled_for_query=true in config.",
        )

    query_id = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()

    m = get_metrics()
    m.queries_total.inc()

    # Save the uploaded image to a temporary location
    upload_dir = Path(settings.paths.uploads_dir) / "query_images"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "query_image.png"
    file_bytes = await file.read()
    temp_path = upload_dir / f"{query_id}_{filename}"
    temp_path.write_bytes(file_bytes)

    logger.info(
        "[%s] Image query: %s (%d bytes, prompt=%s)",
        query_id, filename, len(file_bytes),
        text_prompt[:50] if text_prompt else "none",
    )

    try:
        retriever = get_image_query_retriever()
        output = retriever.retrieve(
            image_path=str(temp_path),
            text_prompt=text_prompt,
            top_k=top_k,
            department_filter=department,
            debug=debug,
        )

        # Retriever now always returns (results, info_dict)
        if isinstance(output, tuple):
            results, extra_info = output
        else:
            results = output
            extra_info = {}

        debug_info = extra_info if debug else {}
        ocr_fallback = extra_info.get("ocr_fallback", False)
        ocr_text = extra_info.get("ocr_text") or ""
        timings = extra_info.get("timings", {})

        total_latency = time.perf_counter() - t_start
        m.query_latency.observe(total_latency)

        # Extract modality contribution
        modality_contribution = {
            "image_hits": sum(1 for r in results if r.get("origin") in ("image_query", "text_assisted_image")),
            "linked_hits": sum(1 for r in results if r.get("origin") == "image_linked"),
            "ocr_fallback_hits": sum(1 for r in results if str(r.get("origin", "")).startswith("ocr_fallback")),
            "text_hits": 0,
        }

        latency_split = {
            "image_embedding_ms": timings.get("image_embedding_ms", 0),
            "image_search_ms": timings.get("image_search_ms", 0),
            "linked_expansion_ms": timings.get("linked_expansion_ms", 0),
            "ocr_ms": timings.get("ocr_ms", 0),
            "ocr_retrieval_ms": timings.get("ocr_retrieval_ms", 0),
            "rerank_ms": timings.get("rerank_ms", 0),
            "total_ms": round(total_latency * 1000, 2),
        }

        # ─── LLM generation when OCR fallback produced results ───────
        answer = ""
        citation_list: list[dict] = []
        gen_latency = 0.0

        if ocr_fallback and results:
            from app.api.dependencies import get_llm_engine
            from app.generation.prompt_templates import ChunkContext, build_prompt

            llm = get_llm_engine()
            if not llm.is_loaded:
                try:
                    llm.load()
                except Exception as le:
                    logger.warning("LLM load failed, skipping generation: %s", le)
                    llm = None

            if llm and llm.is_loaded:
                user_question = text_prompt or "Explain what is shown in this image."
                question_with_context = (
                    f"{user_question}\n\n"
                    f"[Text extracted from the uploaded image via OCR]:\n{ocr_text[:2000]}"
                )

                chunks_for_prompt = []
                for r in results[:8]:
                    meta = r.get("metadata", {})
                    chunks_for_prompt.append(ChunkContext(
                        chunk_id=str(r.get("chunk_id", "")),
                        text=meta.get("text", ""),
                        source=meta.get("source", "unknown"),
                        modality=meta.get("modality", "document"),
                        page_start=meta.get("page_start"),
                        speaker=meta.get("speaker"),
                        timestamp_start=meta.get("timestamp_start"),
                        reranker_score=r.get("reranker_score", r.get("rrf_score", 0)),
                    ))

                prompt = build_prompt(question_with_context, chunks_for_prompt)

                t_gen = time.perf_counter()
                try:
                    answer = llm.generate(prompt, max_tokens=512)
                except Exception as ge:
                    logger.warning("LLM generation failed: %s", ge)
                    answer = ""
                gen_latency = round((time.perf_counter() - t_gen) * 1000, 2)
                latency_split["generation_ms"] = gen_latency

                # Build citations from results
                for r in results[:8]:
                    meta = r.get("metadata", {})
                    citation_list.append({
                        "source": meta.get("source", "unknown"),
                        "page": meta.get("page_start"),
                        "modality": meta.get("modality", "document"),
                    })

        elif ocr_fallback and not results and ocr_text.strip():
            # OCR extracted text but no knowledge base matches — answer from OCR alone
            from app.api.dependencies import get_llm_engine
            llm = get_llm_engine()
            if not llm.is_loaded:
                try:
                    llm.load()
                except Exception:
                    llm = None

            if llm and llm.is_loaded:
                user_question = text_prompt or "Explain what is shown in this image."
                direct_prompt = (
                    "<|im_start|>system\n"
                    "You are a helpful assistant. Answer the user's question based on the text extracted from an image via OCR. "
                    "Be accurate and cite what you see in the extracted text.<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"The following text was extracted from an uploaded image:\n\n{ocr_text[:3000]}\n\n"
                    f"QUESTION: {user_question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                t_gen = time.perf_counter()
                try:
                    answer = llm.generate(direct_prompt, max_tokens=512)
                except Exception as ge:
                    logger.warning("LLM generation failed: %s", ge)
                    answer = ""
                gen_latency = round((time.perf_counter() - t_gen) * 1000, 2)
                latency_split["generation_ms"] = gen_latency

        latency_split["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        # Record to query store for observability
        try:
            from app.api.dependencies import get_query_store
            store = get_query_store()
            store.store_query(
                query_id=query_id,
                query_text=f"[IMAGE_QUERY] {text_prompt or filename}",
                results=results[:5],
                answer=answer or "(image query — no text generation)",
                retrieval_latency=total_latency,
                rerank_latency=0,
                generation_latency=gen_latency / 1000,
                debug_info={
                    "retrieval_mode": "image",
                    "ocr_fallback": ocr_fallback,
                    "modality_contribution": modality_contribution,
                    "latency_split": latency_split,
                },
            )
        except Exception:
            pass

        return ImageQueryResponse(
            query_id=query_id,
            retrieval_mode="image",
            results=[
                {
                    "chunk_id": str(r.get("chunk_id", "")),
                    "source": r.get("metadata", {}).get("source", "unknown"),
                    "origin": r.get("origin", "unknown"),
                    "score": round(r.get("rrf_score", r.get("reranker_score", 0)), 4),
                    "clip_score": r.get("clip_score"),
                    "modality": r.get("metadata", {}).get("modality", "unknown"),
                    "text_preview": r.get("metadata", {}).get("text", "")[:200],
                }
                for r in results
            ],
            result_count=len(results),
            modality_contribution=modality_contribution,
            latency_split=latency_split,
            ocr_text=ocr_text if ocr_fallback else None,
            answer=answer if answer else None,
            citations=citation_list,
            debug_info=extra_info if debug else None,
        )

    except Exception as e:
        logger.error("[%s] Image query failed: %s\n%s", query_id, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ── Phase 4.4: Audio-as-Query ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/query/audio", response_model=AudioQueryResponse)
async def query_audio(
    file: UploadFile = File(...),
    max_tokens: int = Form(512),
    department: Optional[str] = Form(None),
):
    """
    Audio-as-Query: upload an audio file, transcribe via Whisper,
    then run the standard text query pipeline.

    Flow:  Audio → Whisper transcription → text query path (existing)
    No waveform embeddings. No additional vector space.
    """
    from app.api.dependencies import (
        get_embedding_model,
        get_hybrid_retriever,
        get_llm_engine,
        get_metrics,
        get_query_store,
    )
    from app.generation.prompt_templates import ChunkContext, build_prompt
    from app.processing.normalization import normalize_query

    settings = get_settings()
    query_id = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()

    m = get_metrics()
    m.queries_total.inc()

    # Save audio to temp location
    upload_dir = Path(settings.paths.uploads_dir) / "query_audio"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "query_audio.wav"
    file_bytes = await file.read()
    temp_path = upload_dir / f"{query_id}_{filename}"
    temp_path.write_bytes(file_bytes)

    logger.info("[%s] Audio query: %s (%d bytes)", query_id, filename, len(file_bytes))

    try:
        # ── Step 1: Whisper transcription ─────────────────────────────
        t_transcribe = time.perf_counter()
        from faster_whisper import WhisperModel
        from app.ingestion.audio_worker import transcribe_audio

        whisper_model_dir = str(Path(settings.paths.models_dir) / "whisper" / "faster-whisper-small")
        whisper_model = WhisperModel(whisper_model_dir, device="cpu", compute_type="int8")
        result = transcribe_audio(whisper_model, str(temp_path), filename)
        transcript = result.get("transcript", "")
        transcription_latency = (time.perf_counter() - t_transcribe) * 1000

        if not transcript.strip():
            return AudioQueryResponse(
                query_id=query_id,
                retrieval_mode="audio",
                transcript="",
                answer="No speech detected in audio file.",
                transcription_latency_ms=round(transcription_latency, 2),
                total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        logger.info(
            "[%s] Transcribed: %d chars in %.0fms",
            query_id, len(transcript), transcription_latency,
        )

        # ── Step 2: Standard text query pipeline ─────────────────────
        query_text = normalize_query(transcript)
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(query_text)

        retriever = get_hybrid_retriever()
        modality_filter = None
        if department:
            department_filter = department
        else:
            department_filter = None

        from app.config.runtime_settings import get_overrides
        runtime_overrides = get_overrides()

        retrieval_output = retriever.retrieve(
            query=query_text,
            query_embedding=query_embedding,
            department_filter=department_filter,
            debug=False,
            overrides=runtime_overrides if runtime_overrides else None,
        )
        results = retrieval_output

        if not results:
            return AudioQueryResponse(
                query_id=query_id,
                retrieval_mode="audio",
                transcript=transcript,
                answer="No relevant documents found for the transcribed audio.",
                transcription_latency_ms=round(transcription_latency, 2),
                total_latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # ── Step 3: LLM generation ───────────────────────────────────
        chunks = []
        for r in results[:5]:
            meta = r.get("metadata", r.get("payload", {}))
            chunks.append(ChunkContext(
                chunk_id=r["chunk_id"],
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                modality=meta.get("modality", "document"),
                page_start=meta.get("page_start"),
                speaker=meta.get("speaker"),
                timestamp_start=meta.get("timestamp_start"),
                reranker_score=r.get("reranker_score", 0.0),
            ))

        llm = get_llm_engine()
        answer = "(LLM unavailable)"
        if llm.is_loaded or True:
            try:
                if not llm.is_loaded:
                    llm.load()
                prompt = build_prompt(
                    query=query_text,
                    chunks=chunks,
                    token_counter=llm.count_tokens,
                )
                answer = ""
                for token in llm.generate_stream(prompt, max_tokens=max_tokens):
                    answer += token
            except Exception as e:
                logger.warning("[%s] LLM generation failed: %s", query_id, e)
                answer = f"(LLM error: {e})"

        total_latency = (time.perf_counter() - t_start) * 1000

        citations = [
            {
                "source": c.source,
                "page": c.page_start,
                "modality": c.modality,
                "speaker": c.speaker,
            }
            for c in chunks
        ]

        # Record to query store
        try:
            store = get_query_store()
            store.store_query(
                query_id=query_id,
                query_text=f"[AUDIO_QUERY] {transcript[:200]}",
                results=results[:5],
                answer=answer[:500],
                retrieval_latency=(time.perf_counter() - t_start),
                rerank_latency=0,
                generation_latency=0,
                debug_info={"retrieval_mode": "audio", "transcript": transcript[:500]},
            )
        except Exception:
            pass

        return AudioQueryResponse(
            query_id=query_id,
            retrieval_mode="audio",
            transcript=transcript,
            answer=answer,
            citations=citations,
            transcription_latency_ms=round(transcription_latency, 2),
            total_latency_ms=round(total_latency, 2),
        )

    except Exception as e:
        logger.error("[%s] Audio query failed: %s\n%s", query_id, e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
