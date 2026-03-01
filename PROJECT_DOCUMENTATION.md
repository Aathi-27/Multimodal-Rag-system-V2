<!-- # Offline RAG V2 — Complete Project Documentation

> **A fully offline, GPU-accelerated, multimodal Retrieval-Augmented Generation system**
> Built for air-gapped environments with zero cloud dependency.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack & Justifications](#2-technology-stack--justifications)
3. [ML Models — What, Why & Specifications](#3-ml-models--what-why--specifications)
4. [Python Libraries — What & Why](#4-python-libraries--what--why)
5. [Frontend Libraries — What & Why](#5-frontend-libraries--what--why)
6. [System Architecture](#6-system-architecture)
7. [Directory Structure](#7-directory-structure)
8. [Configuration System](#8-configuration-system)
9. [Ingestion Pipeline](#9-ingestion-pipeline)
10. [Processing Pipeline](#10-processing-pipeline)
11. [Retrieval Pipeline](#11-retrieval-pipeline)
12. [Generation Pipeline](#12-generation-pipeline)
13. [API Layer — All 34 Endpoints](#13-api-layer--all-34-endpoints)
14. [Frontend Application](#14-frontend-application)
15. [Observability & Monitoring](#15-observability--monitoring)
16. [Research Lab](#16-research-lab)
17. [Experimentation Engine](#17-experimentation-engine)
18. [Index Versioning System](#18-index-versioning-system)
19. [Dependency Injection Graph](#19-dependency-injection-graph)
20. [Error Handling & Fault Tolerance](#20-error-handling--fault-tolerance)
21. [Storage Layout](#21-storage-layout)
22. [Data Flow — End to End](#22-data-flow--end-to-end)
23. [Deployment & Startup](#23-deployment--startup)
24. [Hardware Requirements](#24-hardware-requirements)

---

## 1. Project Overview

**Offline RAG V2** is an enterprise-grade, fully offline Retrieval-Augmented Generation system designed for organizations that operate in **air-gapped, privacy-critical, or bandwidth-constrained** environments. Every component — from text extraction to embedding generation to LLM inference — runs locally on commodity hardware with GPU acceleration.

### Core Capabilities

| Capability | Description |
|---|---|
| **Multimodal Ingestion** | Documents (PDF, DOCX, PPTX), images (PNG, JPG, WEBP), and audio (MP3, WAV, M4A, OGG, FLAC) |
| **Hybrid Retrieval** | Semantic vector search + keyword BM25 search fused via Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | Neural reranker refines the top candidates for precision |
| **Entity-Aware Retrieval** | Regex-extracted entities inject additional relevant chunks into the pipeline |
| **Streaming LLM Generation** | Token-by-token SSE streaming with citation attribution |
| **GPU Acceleration** | CUDA-powered embedding, reranking, and LLM inference on NVIDIA GTX 1650 |
| **Research Lab** | Survival tracking, failure diagnosis, corpus coverage, embedding quality monitoring, A/B experiments |
| **Knowledge Base CRUD** | Full document management with per-document analytics and re-indexing |
| **Zero Cloud Dependency** | All models stored locally, no API keys, no internet required at runtime |

### Design Principles

1. **Fully Offline**: All ML models are pre-downloaded and stored in `models/`. No network calls at runtime.
2. **Process Isolation**: Each ingestion modality runs in its own `multiprocessing.Process` — a crash in OCR cannot kill the audio worker.
3. **Lazy Model Loading**: Models are loaded on-demand and cached via `@lru_cache` singletons. Cold-start loads only what's needed.
4. **Observable by Default**: Structured JSON logging, per-stage tracing, Prometheus-style metrics, and query history are built in from day one.
5. **Research-Grade Diagnostics**: Every query can produce a full survival log showing exactly which chunks lived or died at each pipeline stage and why.

---

## 2. Technology Stack & Justifications

### Backend

| Technology | Version | Role | Why This Choice |
|---|---|---|---|
| **Python** | 3.11 | Primary language | Best ML ecosystem (PyTorch, Transformers, llama-cpp-python). 3.11 for 25% speed improvement over 3.10. |
| **FastAPI** | ≥0.104 | Web framework | Async-native, automatic OpenAPI docs, Pydantic validation, SSE streaming support. Fastest Python framework for REST APIs. |
| **Uvicorn** | ≥0.24 | ASGI server | Production-grade ASGI server. Runs FastAPI with native async I/O. Low overhead. |
| **Pydantic** | v2 | Data validation | Type-safe request/response models. Automatic JSON serialization. 5–50× faster than v1 via Rust core. |
| **Qdrant** | localhost:6333 | Vector database | High-performance vector similarity search with HNSW indexing. Runs as standalone binary — no Docker required. Supports filtering, payload storage, scroll API. |
| **PyYAML** | ≥6.0 | Config parser | Loads the central `config.yaml` file. Human-readable configuration format. |
| **Python-dotenv** | ≥1.0 | Env management | Loads `.env` files for environment variable overrides (RAG_SECTION__KEY format). |

### Frontend

| Technology | Version | Role | Why This Choice |
|---|---|---|---|
| **React** | 18.3.1 | UI framework | Component-based architecture, hooks for state management, massive ecosystem. |
| **TypeScript** | 5.6.3 | Language | Static typing catches bugs at compile time. IntelliSense support in VS Code. |
| **Vite** | 6.0.3 | Build tool | Instant HMR, ES module-native dev server, 10–100× faster than Webpack. |
| **TailwindCSS** | 3.4.16 | Styling | Utility-first CSS framework. Zero runtime overhead. Design system via CSS custom properties. |
| **React Router** | 6.28.0 | Routing | Declarative routing for single-page application navigation between pages. |

### Infrastructure

| Technology | Role | Why |
|---|---|---|
| **Qdrant** | Vector database | Local binary, no Docker needed. HNSW index, cosine similarity, payload filtering. |
| **CUDA 12.8** | GPU compute | Accelerates PyTorch operations (embedding, reranking) and llama.cpp LLM inference. |
| **Multiprocessing** | Worker isolation | `multiprocessing.Process` + `Queue` — full crash isolation per modality worker. |

---

## 3. ML Models — What, Why & Specifications

### 3.1 BGE-small-en-v1.5 — Embedding Model

| Property | Value |
|---|---|
| **Full Name** | BAAI/bge-small-en-v1.5 |
| **Role** | Converts text chunks and queries into 384-dimensional dense vectors for semantic similarity search |
| **Dimensions** | 384 |
| **Max Tokens** | 512 |
| **Model Format** | SafeTensors (loaded via `sentence-transformers`) |
| **Normalization** | L2-normalized output vectors (cosine similarity = dot product) |
| **Device** | Auto-detects CUDA if available, falls back to CPU |
| **Batch Size** | 64 chunks per batch during ingestion |
| **Local Path** | `models/embeddings/bge-small-en-v1.5/` |

**Why BGE-small?** It ranks among the top models on the MTEB leaderboard for its size class. At only 33M parameters, it loads in <1 second and embeds a 64-chunk batch in ~200ms on GPU. The 384-dim vectors keep Qdrant storage compact while maintaining high retrieval quality. The `Represent this sentence for searching relevant passages:` query prefix is used for asymmetric search.

**How it's used:**
- `embed_texts(texts, batch_size=64)` — Batch-encodes document chunks during ingestion
- `embed_query(query)` — Encodes a user query with the special query prefix
- `embed_single(text)` — Encodes a single text (used by analytics)

---

### 3.2 BGE-reranker-base — Cross-Encoder Reranker

| Property | Value |
|---|---|
| **Full Name** | BAAI/bge-reranker-base |
| **Role** | Rescores the top RRF-fused candidates using cross-attention between the query and each chunk |
| **Architecture** | CrossEncoder (query and document encoded jointly, not separately) |
| **Loaded Via** | `sentence_transformers.CrossEncoder` |
| **Device** | CPU (can use CUDA) |
| **Candidates Scored** | Top 15 from RRF fusion |
| **Score Normalization** | Sigmoid: `1 / (1 + exp(-raw_score))` → [0, 1] range |
| **Threshold** | 0.15 (chunks below this are dropped) |
| **Minimum Results** | 5 (even if all scores are below threshold, keep at least 5) |
| **Target Latency** | <50ms for 15 candidates |
| **Local Path** | `models/reranker/bge-reranker-base/` |

**Why a cross-encoder?** Bi-encoder models (like BGE-small) encode the query and document independently — fast but less accurate. A cross-encoder jointly attends to both the query and the chunk text, capturing fine-grained semantic relationships. This dramatically improves precision at the cost of speed, which is why we only apply it to the top 15 candidates (not all 50).

**How it's used:**
- The `Reranker.rerank()` method builds `(query, chunk_text)` pairs and calls `model.predict(pairs)`.
- Raw logit scores are sigmoid-normalized to [0, 1].
- Results are sorted by score descending, and those below 0.15 are dropped (unless doing so would leave fewer than 5 results).

---

### 3.3 Qwen2.5-3B-Instruct-Q4_K_M — Language Model

| Property | Value |
|---|---|
| **Full Name** | Qwen2.5-3B-Instruct (4-bit quantized, K_M variant) |
| **Role** | Generates natural language answers from retrieved context using ChatML prompt format |
| **Parameters** | 3 billion |
| **Quantization** | Q4_K_M (GGUF format) — 4-bit with medium dequantization quality |
| **Context Window** | 32,768 tokens |
| **Max New Tokens** | 512 |
| **Temperature** | 0.7 |
| **GPU Layers** | 36 (all transformer layers offloaded to GTX 1650 CUDA) |
| **Stop Tokens** | `<\|im_end\|>`, `<\|endoftext\|>` |
| **Inference Engine** | llama-cpp-python (Python bindings for llama.cpp) |
| **Local Path** | `models/llm/qwen2.5-3b-instruct-q4_k_m.gguf` |

**Why Qwen2.5-3B?** At 3B parameters with Q4_K_M quantization, it fits entirely in the GTX 1650's 4GB VRAM with room to spare. Despite its small size, Qwen2.5 demonstrates strong instruction-following and citation capabilities. The GGUF format via llama.cpp provides optimized inference with Flash Attention, KV-cache, and batch processing built in.

**How it's used:**
- `LLMEngine.load()` initializes the `llama_cpp.Llama` object with n_ctx=32768, n_gpu_layers=36
- `LLMEngine.generate(prompt, max_tokens)` runs single-shot generation
- `LLMEngine.generate_stream(prompt)` yields tokens one at a time for SSE streaming
- Streaming logs TTFT (time to first token) and tokens/second

---

### 3.4 Gemma-2-9B-IT-Q4_K_M — Alternative LLM (Available)

| Property | Value |
|---|---|
| **Full Name** | Google Gemma-2-9B-IT (4-bit quantized) |
| **Role** | Larger alternative LLM for higher-quality generation |
| **Parameters** | 9 billion |
| **Local Path** | `models/llm/gemma-2-9b-it-Q4_K_M.gguf` |
| **Status** | Downloaded, available but not the default model |

**Why available?** Provides a quality upgrade path. When running on hardware with ≥8GB VRAM, this model produces significantly better answers. The system can swap models by changing `llm.model_path` in `config.yaml`.

---

### 3.5 Faster-Whisper-small — Speech-to-Text

| Property | Value |
|---|---|
| **Full Name** | Systran/faster-whisper-small |
| **Role** | Transcribes audio files (MP3, WAV, M4A, OGG, FLAC) to text |
| **Architecture** | CTranslate2-optimized Whisper (OpenAI Whisper re-implementation) |
| **Compute Type** | int8 (8-bit quantized for CPU efficiency) |
| **Device** | CPU |
| **Beam Size** | 5 |
| **Language** | English (forced) |
| **VAD Filter** | Enabled (Voice Activity Detection skips silence) |
| **Output** | Transcribed segments with timestamps and optional speaker labels |
| **Local Path** | `models/whisper/faster-whisper-small/` |

**Why Faster-Whisper instead of OpenAI Whisper?** CTranslate2 provides a 4× speed improvement over the original OpenAI Whisper implementation while maintaining identical accuracy. The int8 quantization further reduces memory usage by 50%. VAD filtering skips silent segments, improving throughput on files with pauses.

**How it's used:**
- `audio_worker.py` loads the WhisperModel in a separate `multiprocessing.Process`
- `model.transcribe(audio_path, beam_size=5, language="en", vad_filter=True)` returns segments
- Each segment includes start time, end time, and text — these become chunk timestamps

---

### 3.6 EasyOCR — Image Text Extraction

| Property | Value |
|---|---|
| **Full Name** | EasyOCR (JaidedAI) |
| **Role** | Extracts text from images (PNG, JPG, WEBP) with bounding box locations |
| **Languages** | English |
| **Device** | CPU (gpu=False) |
| **Confidence Threshold** | 0.7 (text below this is discarded) |
| **Output** | List of (bounding_box, text, confidence) tuples |
| **Local Path** | `models/ocr/` (auto-downloads models on first use) |

**Why EasyOCR?** It supports 80+ languages out of the box, requires no Tesseract installation, and handles varied fonts, handwriting, and image qualities robustly. The confidence threshold of 0.7 filters out noisy detections while retaining legitimate text.

**How it's used:**
- `ocr_worker.py` creates an `easyocr.Reader(['en'], gpu=False)` in a worker process
- `reader.readtext(image_path)` returns detections with bounding boxes
- Results above the confidence threshold are concatenated into a single text block

---

### 3.7 PaddleOCR — Secondary OCR (Available)

| Property | Value |
|---|---|
| **Full Name** | PaddlePaddle OCR |
| **Role** | Alternative/fallback OCR engine |
| **Local Path** | `models/ocr/paddleocr/` |
| **Status** | Downloaded, available as alternative |

---

### 3.8 Pyannote-audio — Speaker Diarization (Optional)

| Property | Value |
|---|---|
| **Role** | Identifies and labels different speakers in audio transcripts |
| **Status** | Optional dependency, installed but not active by default |
| **Used When** | Multi-speaker audio needs per-speaker chunk attribution |

---

## 4. Python Libraries — What & Why

### Core Web Framework

| Library | Version Constraint | Purpose |
|---|---|---|
| `fastapi` | ≥0.104 | Async REST API framework with automatic OpenAPI documentation, Pydantic v2 integration, dependency injection, and SSE streaming support |
| `uvicorn[standard]` | ≥0.24 | Production ASGI server for FastAPI. The `[standard]` extra includes uvloop (faster event loop) and httptools (faster HTTP parsing) |
| `pydantic` | (bundled with FastAPI) | 70+ Pydantic models define every request/response schema with automatic validation, serialization, and OpenAPI spec generation |
| `python-multipart` | ≥0.0.6 | Enables `UploadFile` handling in FastAPI. Required for multipart/form-data file uploads |

### Document Processing

| Library | Purpose |
|---|---|
| `docling` | IBM's document converter. Extracts structured Markdown from PDF, DOCX, and PPTX files, preserving headers, tables, and page boundaries. Runs in a subprocess via `multiprocessing.Process` for crash isolation |
| `ftfy` | "Fixes Text For You" — Repairs mojibake, broken Unicode, HTML entities, and other encoding corruption. Part of the text normalization pipeline |
| `pillow` | Python Imaging Library fork. Opens and validates image files before passing to EasyOCR. Handles format conversion and image metadata extraction |

### ML & Inference

| Library | Purpose |
|---|---|
| `sentence-transformers` | Loads both the BGE-small-en-v1.5 embedding model (via `SentenceTransformer`) and the BGE-reranker-base cross-encoder (via `CrossEncoder`). Provides batch encoding with automatic GPU offloading and L2 normalization |
| `llama-cpp-python` | Python bindings for the C++ llama.cpp inference engine. Runs GGUF-format LLMs with GPU offloading, KV-cache, Flash Attention, and token streaming. The `LLMEngine` class wraps this for load/unload/generate/stream operations |
| `torch` | PyTorch — the foundational ML framework. Required by sentence-transformers for tensor operations, CUDA device management, and model loading |
| `numpy` | Numerical array operations. Used for embedding normalization, cosine similarity computation, vector manipulation, sigmoid scoring in the reranker, and statistical analysis in the embedding quality checker |
| `faster-whisper` | CTranslate2-optimized Whisper for speech-to-text. 4× faster than original OpenAI Whisper with identical accuracy. Supports int8 quantization, VAD filtering, and beam search |
| `easyocr` | Deep-learning OCR engine. Extracts text with bounding boxes and confidence scores from images. Supports 80+ languages with no external dependencies like Tesseract |
| `pyannote.audio` | Optional speaker diarization. Identifies distinct speakers in audio files for per-speaker chunk attribution |

### Search & Retrieval

| Library | Purpose |
|---|---|
| `qdrant-client` | Official Python client for the Qdrant vector database. Handles collection creation (HNSW config), point upsert with payloads, similarity search with filtering, scroll pagination, and health checks |
| `rank-bm25` | BM25Okapi implementation for keyword-based retrieval. Tokenized index is built in-memory and serialized to pickle files. Provides the sparse retrieval branch that complements vector search |

### Configuration & Utilities

| Library | Purpose |
|---|---|
| `pyyaml` | Parses the central `config.yaml` file into a Python dictionary. Used by the `Settings` loader to initialize all dataclass configurations |
| `python-dotenv` | Loads `.env` files for environment variable overrides. Enables the `RAG_SECTION__KEY=value` override mechanism without modifying config.yaml |
| `huggingface_hub` | Used by the model download script (`scripts/download_models.py`) to fetch models from the HuggingFace Hub during initial setup |

### Testing

| Library | Purpose |
|---|---|
| `pytest` | Test framework for unit and integration tests |
| `pytest-asyncio` | Enables `async def test_*` functions for testing FastAPI's async endpoints |
| `httpx` | Async HTTP client used as the test transport for FastAPI's `TestClient` |

---

## 5. Frontend Libraries — What & Why

### Production Dependencies

| Library | Version | Purpose |
|---|---|---|
| `react` | 18.3.1 | Component-based UI framework. Uses hooks (useState, useEffect, useRef, useCallback) for state management. React 18's concurrent features enable smooth SSE streaming UI updates |
| `react-dom` | 18.3.1 | DOM renderer for React. Provides `createRoot` for React 18's concurrent rendering mode |
| `react-router-dom` | 6.28.0 | Client-side routing. Provides `BrowserRouter`, `Routes`, `Route`, `Link`, `useNavigate` for SPA navigation between ChatPage, KnowledgeBase, QueryHistory, SystemStatus, FailureDiagnosis, and ExperimentLab pages |
| `axios` | 1.7.9 | HTTP client for REST API calls. Configured with baseURL pointing to the FastAPI backend. Handles file uploads with progress tracking, JSON request/response, and error interceptors |
| `react-markdown` | 9.0.1 | Renders Markdown-formatted LLM responses in the chat UI. Supports code blocks, headers, lists, tables, and inline formatting |
| `remark-gfm` | (peer dep) | GitHub Flavored Markdown plugin for react-markdown. Enables tables, strikethrough, task lists, and autolinks in LLM responses |
| `rehype-sanitize` | (peer dep) | Sanitizes rendered HTML to prevent XSS attacks from LLM-generated content |
| `lucide-react` | 0.564.0 | Icon library. Provides consistent, tree-shakeable SVG icons (Upload, Search, Settings, Brain, FileText, Clock, Activity, AlertTriangle, etc.) used throughout the UI |

### Development Dependencies

| Library | Version | Purpose |
|---|---|---|
| `typescript` | 5.6.3 | Static type checking for all `.tsx` and `.ts` files. Catches type errors at compile time |
| `vite` | 6.0.3 | Build tool and dev server. Provides instant HMR, ES module dev serving, and optimized production bundling. Configured with proxy rules to forward `/upload`, `/query`, `/health`, etc. to the FastAPI backend |
| `tailwindcss` | 3.4.16 | Utility-first CSS framework. Configured with custom CSS properties (`--bg-primary`, `--text-primary`, etc.) for a design system with Inter font, slate palette, and depth layers |
| `autoprefixer` | (peer dep) | PostCSS plugin that adds vendor prefixes for cross-browser CSS compatibility |
| `postcss` | (peer dep) | CSS transformation pipeline used by Tailwind and Autoprefixer |
| `@vitejs/plugin-react` | (dev dep) | Vite plugin that enables React Fast Refresh (HMR) and JSX transformation |
| `@types/react` | (dev dep) | TypeScript type definitions for React |
| `@types/react-dom` | (dev dep) | TypeScript type definitions for React DOM |

---

## 6. System Architecture

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND  (React 18 + TypeScript + Vite)       │
│  ChatPage │ KnowledgeBase │ QueryHistory │ SystemStatus │ ResearchLab  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP / SSE (Vite Proxy → localhost:8000)
┌──────────────────────────────▼──────────────────────────────────────────┐
│                         API LAYER  (FastAPI + Pydantic)                 │
│  routes.py (1703 lines, 34 endpoints) │ models.py (472 lines, 70+ schemas)│
│  dependencies.py (DI container)      │ CORS Middleware                 │
└──────┬────────────┬───────────┬─────────┬──────────┬───────────────────┘
       │            │           │         │          │
┌──────▼────┐ ┌─────▼──────┐ ┌─▼──────┐ ┌▼───────┐ ┌▼──────────────┐
│ Ingestion │ │ Processing │ │Retrieval│ │Generatn│ │ Observability  │
│           │ │            │ │        │ │        │ │                │
│ Workers   │ │ Chunking   │ │Vector  │ │ LLM    │ │ Logging        │
│ TaskQueue │ │ Normalize  │ │BM25    │ │ Prompt │ │ Metrics        │
│ Document  │ │ Validators │ │Hybrid  │ │Templates│ │ Tracing        │
│ OCR       │ │ Entity     │ │Reranker│ │ Stream │ │ QueryStore     │
│ Audio     │ │ Extraction │ │        │ │        │ │ Survival       │
└───────────┘ └────────────┘ └────────┘ └────────┘ │ Diagnosis      │
                                                    │ Coverage       │
                                                    │ EmbedQuality   │
                                                    └────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                   │
│  Qdrant (vectors + payloads) │ BM25 pickle │ JSONL query history       │
│  Uploaded files              │ Versioned index dirs │ Rotating logs    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Directory | Files | Responsibility |
|---|---|---|---|
| **API** | `app/api/` | `routes.py`, `models.py`, `dependencies.py` | HTTP endpoints, Pydantic schemas, dependency injection |
| **Config** | `app/config/` | `settings.py` | YAML loading, 11 dataclasses, env overrides |
| **Ingestion** | `app/ingestion/` | `document_worker.py`, `ocr_worker.py`, `audio_worker.py`, `task_queue.py`, `workers.py` | File reception, modality routing, worker pool management |
| **Processing** | `app/processing/` | `chunking.py`, `normalization.py`, `validators.py`, `entity_extractor.py` | Sliding-window chunking, Unicode normalization, token validation, entity extraction |
| **Retrieval** | `app/retrieval/` | `vector_store.py`, `bm25_store.py`, `hybrid_retriever.py`, `reranker.py` | Qdrant wrapper, BM25 index, RRF fusion, cross-encoder reranking |
| **Generation** | `app/generation/` | `llm_engine.py`, `prompt_templates.py` | LLM loading/inference/streaming, prompt construction with dual system prompts |
| **Models** | `app/models/` | `embeddings.py`, `model_manager.py`, `model_registry.py` | Embedding model wrapper, lazy model lifecycle, registry of all 7 model types |
| **Observability** | `app/observability/` | `logging_config.py`, `metrics.py`, `tracing.py`, `query_store.py`, `survival_tracker.py`, `failure_diagnosis.py`, `corpus_coverage.py`, `embedding_quality.py` | Structured logging, counters/histograms/gauges, distributed tracing, query persistence, Research Lab analysis engines |
| **Experimentation** | `app/experimentation/` | `experiment_engine.py` | Side-by-side comparison, batch evaluation, parameter sweep |
| **Versioning** | `app/versioning/` | `index_manager.py`, `checksum.py` | Symlink-based index version switching, BM25↔Qdrant checksum validation |
| **Utils** | `app/utils/` | `circuit_breaker.py`, `exceptions.py` | Circuit breaker pattern, 15+ custom exception classes |

---

## 7. Directory Structure

```
D:/Offline_Rag_V2/
├── models/                           # All ML models (pre-downloaded)
│   ├── embeddings/
│   │   └── bge-small-en-v1.5/       # 384-dim embedding model (~66MB)
│   ├── reranker/
│   │   └── bge-reranker-base/       # Cross-encoder reranker (~440MB)
│   ├── llm/
│   │   ├── qwen2.5-3b-instruct-q4_k_m.gguf    # Primary LLM (~2GB)
│   │   └── gemma-2-9b-it-Q4_K_M.gguf          # Alternative LLM (~5GB)
│   ├── whisper/
│   │   └── faster-whisper-small/    # Speech-to-text (~461MB)
│   └── ocr/
│       └── paddleocr/               # OCR models
│
├── rag-system/                       # Main application
│   ├── config.yaml                   # Central configuration (86 lines)
│   ├── requirements.txt              # Python dependencies (20 packages)
│   ├── app/
│   │   ├── main.py                   # FastAPI app, lifespan, CORS, routes
│   │   ├── api/                      # 34 endpoints, 70+ Pydantic models, DI
│   │   ├── config/                   # Settings loader (11 dataclasses)
│   │   ├── ingestion/                # Worker processes (doc, OCR, audio)
│   │   ├── processing/               # Chunking, normalization, validation
│   │   ├── retrieval/                # Vector, BM25, hybrid, reranker
│   │   ├── generation/               # LLM engine, prompt templates
│   │   ├── models/                   # Embedding model, model manager/registry
│   │   ├── observability/            # Logging, metrics, tracing, Research Lab
│   │   ├── experimentation/          # Experiment engine
│   │   ├── versioning/               # Index version manager, checksums
│   │   └── utils/                    # Circuit breaker, custom exceptions
│   └── data/
│       ├── uploads/                  # Uploaded files stored as {uuid}_{filename}
│       ├── index/
│       │   ├── current -> v1.0.0     # Symlink/junction to active version
│       │   └── v1.0.0/
│       │       ├── qdrant/           # Qdrant snapshot data
│       │       └── bm25/            # BM25 pickle files
│       └── logs/                     # Rotating JSON log files
│
├── frontend/                         # React SPA
│   ├── src/
│   │   ├── App.tsx                   # Root component with routing
│   │   ├── pages/                    # 6 page components
│   │   ├── components/               # Shared UI components (Skeleton, etc.)
│   │   └── styles/                   # CSS with design system variables
│   ├── package.json
│   ├── vite.config.ts                # Dev server proxy configuration
│   └── tailwind.config.js
│
├── scripts/                          # Utility scripts
│   ├── download_models.py            # HuggingFace model downloader
│   ├── health_check.py               # System health verification
│   └── init_index.py                 # Index initialization
│
├── tests/                            # Test suite
│   ├── test_chunking.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   └── test_retrieval.py
│
└── docker/                           # Docker support
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 8. Configuration System

### Central Config: `config.yaml`

All system parameters are defined in a single YAML file at `rag-system/config.yaml`. The configuration is loaded once at startup and cached as a singleton via `@lru_cache`.

### Configuration Dataclasses

The `Settings` class in `app/config/settings.py` contains 11 nested dataclasses:

| Dataclass | Key Parameters | Purpose |
|---|---|---|
| `ChunkingConfig` | `target_tokens=480`, `max_tokens=512`, `overlap_tokens=50`, `method=sliding_window` | Controls how documents are split into chunks |
| `RetrievalConfig` | `vector_top_k=50`, `bm25_top_k=50`, `rrf_k=60`, `rerank_count=15`, `rerank_threshold=0.15`, `rerank_min_results=5` | Retrieval pipeline parameters |
| `MemoryConfig` | `entity_per_limit=10`, `entity_global_limit=20` | Entity injection limits |
| `PageEmbeddingsConfig` | `activation_threshold=50`, `fallback_similarity=0.25`, `fallback_min_pages=2` | Page-level embedding for large documents |
| `LLMConfig` | `model_path`, `context_window=32768`, `max_new_tokens=512`, `temperature=0.7`, `gpu_layers=36`, `stop_tokens` | LLM inference parameters |
| `SLMConfig` | `enabled=false`, `context_window=4096`, `max_new_tokens=256`, `temperature=0.3`, `gpu_layers=0` | Optional small language model |
| `QdrantConfig` | `host=localhost`, `port=6333`, `collection_name=rag_chunks`, `vector_size=384`, `distance=Cosine`, `hnsw_m=16`, `hnsw_ef_construct=100`, `hnsw_ef_search=64` | Qdrant vector database settings |
| `ServerConfig` | `host=0.0.0.0`, `port=8000`, `workers=1`, `max_upload_size_mb=100` | Uvicorn server settings |
| `PathsConfig` | `data_dir`, `uploads_dir`, `index_dir`, `logs_dir`, `models_dir` | All file system paths |
| `WorkerConfig` | `document_workers=2`, `ocr_workers=1`, `audio_workers=1`, `task_timeout=300` | Worker pool sizes |
| `Settings` (root) | All above + `log_level=INFO`, `log_json=true` | Top-level container |

### Environment Variable Overrides

Any configuration value can be overridden via environment variables using the format:
```
RAG_SECTION__KEY=value
```

Examples:
- `RAG_LLM__GPU_LAYERS=0` → Disable GPU offloading
- `RAG_RETRIEVAL__RERANK_THRESHOLD=0.2` → Raise reranker threshold
- `RAG_SERVER__PORT=9000` → Change server port

The override mechanism preserves types (int, float, bool, str) automatically.

---

## 9. Ingestion Pipeline

### File Upload Flow

```
User uploads file via POST /upload
        │
        ▼
┌─ Validate file ─────────────────────────────┐
│  • Extension check (11 supported types)      │
│  • Size check (≤100MB)                       │
│  • Filename sanitization (no path traversal) │
└─────────────┬───────────────────────────────┘
              │
        ▼ Save to disk
   data/uploads/{uuid}_{original_filename}
              │
        ▼ Detect modality
   .pdf/.docx/.pptx → Document
   .png/.jpg/.webp   → Image
   .mp3/.wav/.m4a    → Audio
              │
        ▼ Background task (_ingest_file)
```

### Modality-Specific Workers

Each modality has a dedicated worker that runs in a separate `multiprocessing.Process`:

**Document Worker** (`document_worker.py`):
- Uses IBM Docling's `DocumentConverter`
- Converts PDF/DOCX/PPTX to structured Markdown
- Preserves page boundaries, headers, and tables
- Runs in subprocess for crash isolation

**OCR Worker** (`ocr_worker.py`):
- Uses EasyOCR with `Reader(['en'], gpu=False)`
- Extracts text with bounding boxes and confidence scores
- Filters detections below 0.7 confidence threshold
- Concatenates results into a single text block

**Audio Worker** (`audio_worker.py`):
- Uses Faster-Whisper with `WhisperModel("small", cpu, int8)`
- Transcribes with `beam_size=5`, `language="en"`, `vad_filter=True`
- Produces timestamped segments for chunk attribution
- Optional pyannote speaker diarization

### Task Queue Architecture

```
┌────────────────┐     ┌───────────────────┐
│   Document Q   │────▶│ Document Worker 1 │
│   (mp.Queue)   │────▶│ Document Worker 2 │
└────────────────┘     └───────────────────┘
                                │
┌────────────────┐     ┌───────▼───────────┐
│    Image Q     │────▶│   OCR Worker 1    │
│   (mp.Queue)   │     └───────────────────┘
└────────────────┘              │
                                │ All results → Shared Result Queue
┌────────────────┐     ┌───────▼───────────┐
│    Audio Q     │────▶│  Audio Worker 1   │
│   (mp.Queue)   │     └───────────────────┘
└────────────────┘
```

### Worker Isolation

Each worker is a `multiprocessing.Process` (NOT `ProcessPoolExecutor`). This provides:
- **Full memory isolation**: A crash in EasyOCR cannot corrupt the Whisper worker
- **Independent failure domains**: Each modality can fail independently
- **Circuit breaker protection**: The `CircuitBreaker` class tracks failures per worker and temporarily stops dispatching if a worker is repeatedly crashing

---

## 10. Processing Pipeline

### Text Normalization (`normalization.py`)

Every piece of extracted text passes through a normalization pipeline before chunking:

```
Raw text
  │
  ▼ Unicode NFC normalization
  │ (canonical decomposition + composition)
  │
  ▼ Quote standardization
  │ (QUOTE_MAP: curly quotes → straight quotes)
  │
  ▼ Tab → space conversion
  │
  ▼ Collapse multiple spaces → single space
  │
  ▼ Collapse multiple newlines → double newline (paragraph boundary)
  │
  ▼ Strip leading/trailing whitespace
  │
  = Clean normalized text
```

The same pipeline is applied to queries via `normalize_query()` to ensure query-document consistency.

### Sliding-Window Chunking (`chunking.py`)

The `SlidingWindowChunker` implements a carefully designed chunking algorithm:

| Parameter | Value | Purpose |
|---|---|---|
| `target_tokens` | 480 | Ideal chunk size — leaves room for metadata within the 512-token embedding limit |
| `max_tokens` | 512 | Hard upper limit — BGE-small-en-v1.5 truncates beyond this |
| `overlap_tokens` | 50 | Context overlap between consecutive chunks to avoid information loss at boundaries |

**Algorithm Steps:**

1. **Sentence Splitting**: Text is split using a regex pattern that detects sentence boundaries (`.!?` followed by space/newline). This respects the natural structure of text.

2. **Markdown Header Preservation**: If a chunk begins with a Markdown header (`# Section`), it is carried forward as context prefix for subsequent chunks from that section.

3. **Window Accumulation**: Sentences are accumulated into a window until reaching `target_tokens` (480).

4. **Overlap Computation**: The last `overlap_tokens` (50) worth of sentences from the previous chunk are prepended to the next chunk's start, ensuring no information is lost at chunk boundaries.

5. **Force-Split for Oversized Sentences**: If a single sentence exceeds 512 tokens (rare, but possible with tables or code), it is forcibly split at the token level.

**Chunk Dataclass:**
```python
@dataclass
class Chunk:
    chunk_id: str           # UUID
    text: str               # Chunk content
    token_count: int        # Validated ≤ 512
    source: str             # Original filename
    modality: str           # "document", "image", "audio"
    page_start: int | None  # Starting page number
    chunk_index: int        # Position within document
    speaker: str | None     # Audio speaker label
    timestamps: dict | None # Audio start/end times
```

### Token Validation (`validators.py`)

Before embedding, every chunk is validated:
- **Hard token limit**: No chunk may exceed 512 tokens. Violations raise `TokenLimitExceededError`.
- **Required metadata**: Each modality has required fields:
  - Documents: `source`, `modality`, `page_start`
  - Images: `source`, `modality`
  - Audio: `source`, `modality`, `speaker`

### Entity Extraction (`entity_extractor.py`)

Extracts named entities from queries using four regex patterns:

| Pattern | Matches | Example |
|---|---|---|
| `_ACRONYM_RE` | 2–6 uppercase letter sequences | `NASA`, `HTTP`, `GDPR` |
| `_QUOTED_RE` | Quoted phrases (2–60 chars) | `"machine learning"`, `"React hooks"` |
| `_CAPITALIZED_PHRASE_RE` | Consecutive capitalized words | `New York`, `Sliding Window` |
| `_HYPHENATED_RE` | Hyphenated compound terms | `cross-encoder`, `state-of-the-art` |

The `find_entity_chunks()` function then performs BM25 lookups for each extracted entity, injecting additional relevant chunks into the retrieval pipeline (up to 10 per entity, 20 globally).

---

## 11. Retrieval Pipeline

The retrieval pipeline is a **7-stage process** orchestrated by `HybridRetriever.retrieve()`:

### Stage 1: Parallel Search

Two search branches run simultaneously:

**Vector Search** (Qdrant):
- Query embedding (384-dim) is compared against all stored chunk embeddings via cosine similarity
- HNSW index with `m=16`, `ef_construct=100`, `ef_search=64` provides approximate nearest neighbor search
- Returns top 50 results with scores and full payloads
- Supports optional modality and department filtering

**BM25 Keyword Search** (rank-bm25):
- Query is normalized and whitespace-tokenized
- BM25Okapi scores all indexed chunks against query tokens
- Returns top 50 results sorted by BM25 score
- Provides complementary keyword-matching that catches exact term matches the vector search may miss

### Stage 2: Reciprocal Rank Fusion (RRF)

RRF combines the two ranked lists into a single fused ranking:

$$\text{RRF}(c) = \sum_{s \in \{vector, bm25\}} \frac{1}{k + rank_s(c)}$$

Where $k = 60$ (constant), and $rank_s(c)$ is the chunk's rank in search source $s$.

- Chunks appearing in both lists get boosted (summed scores)
- Each chunk is tagged with origin: `"vector"`, `"bm25"`, or `"both"`
- The fused list preserves per-source ranks for debugging

### Stage 2b: Entity Injection

After RRF fusion, extracted entities trigger additional BM25 lookups:
- Up to 10 chunks per entity, 20 globally
- Entity-injected chunks are appended with `rrf_score=0.0` and `origin="entity"`
- Deduplication prevents adding chunks already in the fused list

### Stage 3: Cross-Encoder Reranking

The top 15 RRF candidates are scored by the BGE-reranker-base cross-encoder:
- `(query, chunk_text)` pairs are jointly encoded
- Raw logit scores are sigmoid-normalized to [0, 1]
- Results sorted by reranker score descending
- Chunks below threshold (0.15) are dropped
- Minimum 5 results safeguard ensures non-empty results

### Stage 4: Origin Propagation

Each reranked result retains its full lineage:
- `origin`: "vector", "bm25", "both", or "entity"
- `vector_rank`: Original rank in vector search (or null)
- `bm25_rank`: Original rank in BM25 search (or null)
- `reranker_score`: Cross-encoder confidence

### Stage 5: Survival Log (Debug Mode)

When `debug=True`, a complete survival log is computed showing every chunk that appeared in any stage, whether it survived to the final results, and if not, exactly where and why it was dropped:

| Drop Stage | Reason |
|---|---|
| `rrf_fusion` | Low RRF score, below fusion cutoff |
| `rerank_selection` | RRF rank too low (not in top 15) |
| `reranker` | Reranker score below threshold (0.15) |
| `rerank_selection` | Entity-injected but not selected for reranking |

### Stage 6: Result Packaging

Final results are returned with full metadata:
```python
{
    "chunk_id": "abc123...",
    "reranker_score": 0.87,
    "origin": "both",
    "vector_rank": 3,
    "bm25_rank": 7,
    "metadata": {
        "text": "...",
        "source": "manual.pdf",
        "page_start": 14,
        "modality": "document",
        "token_count": 445
    }
}
```

### Stage 7: Debug Info (Optional)

In debug mode, the full pipeline state is returned alongside results:
- Top 20 vector results, BM25 results, RRF-fused candidates
- All reranked results and all dropped candidates
- Effective parameters used
- Entity extraction details
- Complete survival log

---

## 12. Generation Pipeline

### Prompt Construction (`prompt_templates.py`)

The system uses a **dual prompt strategy** that automatically selects the appropriate system prompt based on query complexity:

**Rigorous Prompt** (for analytical queries):
- Triggered when the query matches analytical keywords: `explain`, `compare`, `difference`, `analyze`, `why`, `how does`, `advantage`, `trade-off`, etc.
- Uses `SYSTEM_PROMPT_RIGOROUS` — an architect-level prompt with 7 strict rules:
  1. Answer ONLY from the provided context
  2. Cite every claim with `[Source: filename, Page N]`
  3. If information is insufficient, say "The provided context does not contain..."
  4. Use precise technical vocabulary from the sources
  5. Preserve formulas, numbers, and proper nouns exactly
  6. Structure answers with headers and bullet points
  7. Never hallucinate beyond the context

**Concise Prompt** (for factual queries):
- Used for straightforward questions
- Uses `SYSTEM_PROMPT_CONCISE` with 6 rules:
  1. Provide factual, sourced answers
  2. Mandatory citation for every claim
  3. If not in context, say so
  4. Keep answers focused and brief
  5. Use original terminology
  6. No speculation

**Context Formatting:**
- Retrieved chunks are grouped by source document
- Within each source group, chunks are sorted by reranker score (descending)
- Each chunk is tagged with `[Source: filename, Page N | Score: 0.87]`
- Total context is budget-capped at ~6000 tokens to leave room for the answer

**Prompt Assembly (ChatML format):**
```
<|im_start|>system
{selected_system_prompt}<|im_end|>
<|im_start|>user
Context:
[Source: manual.pdf, Page 14 | Score: 0.87]
{chunk text}
...

Question: {user query}<|im_end|>
<|im_start|>assistant
```

### LLM Engine (`llm_engine.py`)

The `LLMEngine` class wraps `llama_cpp.Llama`:

| Method | Description |
|---|---|
| `load()` | Initializes the Llama model with `n_ctx=32768`, `n_gpu_layers=36`, `verbose=False` |
| `unload()` | Releases the model from memory |
| `generate(prompt, max_tokens)` | Runs single-shot completion, returns full text |
| `generate_stream(prompt)` | Creates a streaming completion, yields tokens one at a time |
| `count_tokens(text)` | Tokenizes text and returns the count (used for context budget) |

**Streaming Flow:**
1. `create_completion(stream=True)` returns a generator
2. Each yielded chunk contains a `choices[0]["text"]` token
3. TTFT (time to first token) is measured and logged
4. Total tokens and tokens/second are logged at stream end
5. Tokens are sent to the frontend via Server-Sent Events (SSE)

---

## 13. API Layer — All 34 Endpoints

All endpoints are defined in `app/api/routes.py` (1703 lines) with a shared `APIRouter` prefix.

### Core Endpoints

| # | Method | Path | Description |
|---|---|---|---|
| 1 | `POST` | `/upload` | Upload a file for ingestion (PDF, DOCX, PPTX, PNG, JPG, WEBP, MP3, WAV, M4A). Returns 202 with task_id. Runs ingestion as BackgroundTask. |
| 2 | `POST` | `/query` | Submit a query. Returns SSE stream with retrieval debug → token-by-token answer → citations → done. |
| 3 | `GET` | `/health` | System health check: Qdrant connectivity, BM25 status, LLM load status, corpus size, uptime. |
| 4 | `GET` | `/status/{task_id}` | Check ingestion task status (pending/processing/completed/failed). |

### Knowledge Base Management

| # | Method | Path | Description |
|---|---|---|---|
| 5 | `GET` | `/documents` | List all ingested documents with aggregated metadata (chunk count, token count, modality). |
| 6 | `GET` | `/documents/{source}/chunks` | Get all chunks for a specific document, sorted by chunk_index. |
| 7 | `DELETE` | `/documents/{source}` | Delete a document and all its chunks from Qdrant and BM25. Triggers BM25 rebuild. |
| 8 | `POST` | `/documents/{source}/reindex` | Re-ingest a document: delete old chunks, re-extract, re-chunk, re-embed, re-index. |
| 9 | `GET` | `/index/health` | Detailed index health: total chunks, documents, avg tokens/chunk, largest document, BM25 vocab size. |
| 10 | `GET` | `/analytics` | Per-document retrieval analytics: query count, avg reranker score, avg rank position. |

### Query History & Replay

| # | Method | Path | Description |
|---|---|---|---|
| 11 | `GET` | `/queries` | Paginated query history (newest first, 50 per page). |
| 12 | `GET` | `/queries/summary` | Aggregate statistics: avg latency, avg chunks/query, error count. |
| 13 | `GET` | `/queries/{query_id}` | Full query record with retrieved chunks, debug info, latency breakdown. |
| 14 | `POST` | `/queries/{query_id}/replay` | Re-run a historical query through the current pipeline (compares old vs. new results). |

### Index Versioning

| # | Method | Path | Description |
|---|---|---|---|
| 15 | `GET` | `/versions` | List all index versions with active indicator. |
| 16 | `POST` | `/versions/{version}/switch` | Switch active index to specified version (updates symlink). |
| 17 | `DELETE` | `/versions/{version}` | Delete an index version (cannot delete active). |
| 18 | `POST` | `/versions/{version}/metadata` | Save metadata (embedding model, chunk size, etc.) for a version. |
| 19 | `GET` | `/versions/{version}/metadata` | Retrieve version metadata. |

### System Monitoring

| # | Method | Path | Description |
|---|---|---|---|
| 20 | `GET` | `/metrics` | Prometheus-style metrics: upload/query counts, latency histograms (avg, p95), corpus size. |
| 21 | `GET` | `/resources` | System resources: CPU %, RAM used/total, GPU name/memory/utilization, disk usage. |

### Settings Control

| # | Method | Path | Description |
|---|---|---|---|
| 22 | `GET` | `/settings/retrieval` | Get current retrieval parameters (including any runtime overrides). |
| 23 | `PATCH` | `/settings/retrieval` | Apply runtime overrides to retrieval parameters (without modifying config.yaml). |
| 24 | `DELETE` | `/settings/retrieval` | Reset all runtime overrides back to config.yaml defaults. |

### Recall Validation

| # | Method | Path | Description |
|---|---|---|---|
| 25 | `POST` | `/queries/{query_id}/validate` | Submit human relevance annotations for retrieved chunks. Computes recall@5, recall@10, MRR. |
| 26 | `GET` | `/queries/{query_id}/recall` | Retrieve computed recall metrics for a validated query. |

### Research Lab Endpoints

| # | Method | Path | Description |
|---|---|---|---|
| 27 | `GET` | `/queries/{query_id}/survival` | Full survival analysis: per-chunk stage tracking, drop reasons, survival rates. |
| 28 | `GET` | `/queries/{query_id}/diagnosis` | Automated failure diagnosis: root cause classification with confidence, evidence, recommendations. |
| 29 | `POST` | `/diagnosis/batch` | Batch diagnosis across multiple queries: cause distribution, avg confidence, common recommendations. |
| 30 | `POST` | `/queries/{query_id}/ground-truth` | Tag a query with ground truth chunk IDs for evaluation. |
| 31 | `POST` | `/experiments/compare` | Side-by-side comparison: run same query with two parameter sets, compute overlap, rank diffs, Jaccard similarity. |
| 32 | `POST` | `/experiments/batch-evaluate` | Batch evaluation: run test queries with optional ground truth, compute recall@5, recall@10, MRR, latency stats. |
| 33 | `GET` | `/corpus/coverage` | Corpus coverage analysis: hotspot/coldspot chunks, per-source coverage rates, never-retrieved chunks. |
| 34 | `GET` | `/embeddings/quality` | Embedding quality diagnostics: norm stats, cosine similarity distribution, outlier detection, health assessment. |

---

## 14. Frontend Application

### Pages

| Page | Route | Description |
|---|---|---|
| **ChatPage** | `/` | Primary interface. 2-column layout: chat messages (left) with citations/debug panel (right). SSE streaming shows tokens in real-time. Supports file upload inline. |
| **KnowledgeBase** | `/knowledge-base` | Document management. Lists all ingested documents with chunk counts, token counts, modality badges. Delete, re-index, and inspect individual chunks. |
| **QueryHistory** | `/query-history` | Historical query browser. Shows all past queries with latency breakdown, chunk counts, and error status. Click to view full retrieved context. |
| **SystemStatus** | `/system-status` | System dashboard. Shows health status, resource usage (CPU/RAM/GPU), metrics, retrieval settings with live adjustment controls. |
| **FailureDiagnosis** | `/failure-diagnosis` | Research Lab page. Analyzes queries with poor results, shows root-cause classification, confidence scores, and actionable recommendations. |
| **ExperimentLab** | `/experiment-lab` | Research Lab page. Side-by-side parameter comparison, batch evaluation, and corpus coverage analysis. |

### Design System

- **Font**: Inter (sans-serif)
- **Color Palette**: Slate-based with CSS custom properties (`--bg-primary`, `--bg-secondary`, `--text-primary`, `--accent`, etc.)
- **Depth Layers**: Card elevation via `--shadow-sm`, `--shadow-md`, `--shadow-lg`
- **Icons**: Lucide React (tree-shakeable SVG icons)
- **Micro-interactions**: Hover transitions, loading skeletons, toast notifications
- **Responsive**: Mobile-friendly with collapsible sidebar navigation

### SSE Streaming Protocol

The `/query` endpoint uses Server-Sent Events for real-time communication:

```
event: retrieval
data: {"chunks": [...], "debug_info": {...}}

event: token
data: {"token": "The"}

event: token
data: {"token": " answer"}

event: token
data: {"token": " is..."}

event: citations
data: {"citations": [{"source": "manual.pdf", "page": 14}]}

event: done
data: {"query_id": "abc123", "total_tokens": 156}
```

---

## 15. Observability & Monitoring

### Structured Logging (`logging_config.py`)

- **Format**: JSON-structured log entries with `timestamp`, `level`, `logger`, `message`, `module`, `function`, `line`
- **Extra Fields**: `trace_id`, `upload_id`, `query_id`, `request_id` for request correlation
- **Output**: Console (stdout) + rotating file (`data/logs/rag-system.log`, 10MB max, 5 backups)
- **Noise Reduction**: Third-party loggers (uvicorn, httpx, sentence_transformers, qdrant_client) set to WARNING level

### Metrics Collector (`metrics.py`)

Custom Prometheus-compatible metrics (no external Prometheus dependency):

| Metric Type | Name | Description |
|---|---|---|
| Counter | `uploads_total` | Total file uploads (with per-modality labels) |
| Counter | `upload_errors` | Upload errors by type |
| Counter | `queries_total` | Total queries processed |
| Histogram | `query_latency_seconds` | End-to-end query latency (avg, p95) |
| Histogram | `retrieval_latency_seconds` | Retrieval stage latency |
| Histogram | `rerank_latency_seconds` | Reranking latency |
| Histogram | `generation_latency_seconds` | LLM generation latency |
| Histogram | `retrieval_hits` | Number of hits per query |
| Histogram | `rrf_top_score` | Top RRF score per query |
| Gauge | `corpus_size` | Total chunks in corpus |
| Gauge | `active_workers` | Active ingestion workers |
| Gauge | `models_loaded` | Number of loaded models |

The `track_latency()` context manager automatically measures and records elapsed time.

### Distributed Tracing (`tracing.py`)

OpenTelemetry-compatible trace structure (without the OTel dependency):

- **Trace**: Represents a complete request lifecycle (upload or query)
- **Span**: Individual stage within a trace (e.g., "retrieval", "reranking", "generation")
- Each span records: `name`, `trace_id`, `span_id`, `start_time`, `end_time`, `duration_ms`, `status`, `error`, `attributes`
- Traces are created with `tracer.start_trace()` and spans with `tracer.span(trace, name)`
- Total duration and per-span timing available via `trace.to_dict()`

### Query Store (`query_store.py`)

Persistent, append-only query history backed by JSONL:

- **Storage**: `data/query_history.jsonl` (one JSON record per line)
- **In-Memory Cache**: Last 500 records for fast access
- **Thread Safety**: `threading.Lock` protects all mutations
- **Record Fields**: `query_id`, `query`, `timestamp`, `retrieved_chunks` (with scores), `answer`, `citation_count`, per-stage latency (`retrieval_latency`, `rerank_latency`, `generation_latency`, `total_latency`), `debug_info`, `survival_log`, `ground_truth_chunk_ids`, `diagnosis`
- **Operations**: `record()` (append), `get_all()` (paginated), `get_by_id()`, `annotate()` (merge extra data like recall validation)

---

## 16. Research Lab

The Research Lab is a suite of advanced diagnostic and analysis tools built on top of the observability layer.

### Survival Tracker (`survival_tracker.py`)

Tracks every chunk's journey through the retrieval pipeline:

```
Vector Search → BM25 Search → RRF Fusion → Entity Injection → Reranking → Final
```

For each unique chunk that appeared in ANY stage, the tracker records:
- Which stages it was present in (with rank and score at each)
- Whether it survived to the final results
- If dropped: exact stage and human-readable reason

**Drop Reason Classification:**

| `dropped_at` | `dropped_reason` |
|---|---|
| `rrf_fusion` | "Only in vector results, not fused" |
| `rrf_fusion` | "Only in BM25 results, not fused" |
| `rrf_fusion` | "Low RRF score, below fusion cutoff" |
| `rerank_selection` | "RRF rank too low (not in top 15)" |
| `reranker` | "Reranker score below threshold (0.15)" |
| `rerank_selection` | "Entity-injected but not selected for reranking" |

### Failure Diagnosis (`failure_diagnosis.py`)

Automated root-cause classification engine that analyzes poor-performing queries:

**Root Cause Categories:**

| Root Cause | Meaning |
|---|---|
| `corpus_gap` | The corpus doesn't contain relevant content for this query |
| `embedding_mismatch` | Query/document embedding similarity is too low |
| `rerank_threshold` | Good candidates were dropped by the reranker threshold |
| `rrf_dilution` | Too many irrelevant chunks diluted the fusion stage |
| `entity_miss` | Entity injection failed to find relevant chunks |
| `parameter_issue` | Current settings are suboptimal for this query type |
| `no_failure` | Insufficient data to diagnose |

Each diagnosis includes:
- **Confidence** (0.0–1.0) — How certain the diagnosis is
- **Evidence** — Specific data points supporting the diagnosis (e.g., "Only 2 vector matches found")
- **Recommendations** — Actionable steps (e.g., "Upload more documents related to this topic")
- **Secondary Causes** — Other potential contributing factors

**Batch Diagnosis** analyzes multiple queries at once and returns:
- Cause distribution (count per root cause)
- Average confidence
- Most common recommendations across all queries
- Individual diagnosis per query

### Corpus Coverage Analyzer (`corpus_coverage.py`)

Analyzes how thoroughly the corpus is being utilized across all queries:

| Metric | Description |
|---|---|
| `coverage_rate` | Fraction of total corpus chunks ever retrieved |
| `never_retrieved_count` | Number of chunks that have never appeared in any query's results |
| `hotspot_chunks` | Top 10 most frequently retrieved chunks (possible over-reliance) |
| `coldspot_chunks` | Least frequently retrieved chunks (>0 retrievals) |
| `per_source_coverage` | Per-document breakdown: total chunks, retrieved chunks, coverage rate |

Scans up to 500 historical queries to build the retrieval frequency map.

### Embedding Quality Checker (`embedding_quality.py`)

Monitors the quality of stored embeddings by sampling vectors from Qdrant:

**Norm Analysis:**
- Average L2 norm (should be ~1.0 for normalized BGE embeddings)
- Norm standard deviation (high variance → inconsistent normalization)
- Outlier count (norms >2σ from mean)

**Cosine Similarity Analysis:**
- Pairwise cosine similarity on a 100-vector subset
- Returns mean, std, min, max, p25, p75
- Very high mean (>0.8) → possible chunk redundancy
- Very low mean (<0.1) → poor embedding quality

**Health Assessment:**
- `"good"`: No issues detected
- `"warning"`: 1–2 minor issues
- `"critical"`: 3+ issues or zero vectors

---

## 17. Experimentation Engine

The `ExperimentEngine` (`app/experimentation/experiment_engine.py`) enables controlled retrieval experiments:

### Side-by-Side Comparison

Run the same query through the pipeline with two different parameter sets and compare:

- **Overlap**: How many chunks appear in both result sets
- **Jaccard Similarity**: $\frac{|A \cap B|}{|A \cup B|}$
- **Unique to A / Unique to B**: Chunks found by one parameter set but not the other
- **Rank Differences**: For overlapping chunks, how their ranks differ between sets
- **Score Comparison**: Reranker score differences for overlapping chunks
- **Latency Difference**: Which parameter set is faster

### Batch Evaluation

Run a test dataset of queries with optional ground truth:

- **Input**: List of queries with optional `ground_truth_chunk_ids` and `expected_source`
- **Metrics Computed**:
  - Recall@5: Fraction of ground truth in top 5 results
  - Recall@10: Fraction of ground truth in top 10 results
  - MRR (Mean Reciprocal Rank): Average of 1/rank for first relevant result
  - Per-query latency
  - Aggregate statistics across all test queries

---

## 18. Index Versioning System

### Architecture

The index versioning system (`app/versioning/index_manager.py`) manages multiple index versions using filesystem-level isolation:

```
data/index/
├── current → v1.0.0          # Symlink (Linux) or Junction (Windows)
├── v1.0.0/
│   ├── qdrant/                # Qdrant snapshot data
│   ├── bm25/                  # BM25 pickle files
│   │   ├── bm25_index.pkl     # Serialized BM25Okapi object
│   │   └── bm25_metadata.pkl  # Chunk IDs + metadata
│   └── version_metadata.json  # Embedding model, chunk size, etc.
└── v1.1.0/
    ├── qdrant/
    ├── bm25/
    └── version_metadata.json
```

### Operations

| Operation | Method | Description |
|---|---|---|
| **Create** | `create_version("v1.1.0")` | Creates directory structure with qdrant/ and bm25/ subdirs |
| **Switch** | `switch_to("v1.1.0")` | Updates the `current` symlink/junction to point to the new version |
| **Delete** | `delete_version("v1.0.0")` | Removes version directory (cannot delete active version) |
| **List** | `list_versions()` | Returns sorted list of all version strings |
| **Metadata** | `save_metadata(version, {...})` | Persists JSON metadata (embedding model, chunk size, overlap, etc.) |
| **Initialize** | `ensure_initialized()` | Creates `v1.0.0` and sets it as current if no versions exist |

### BM25↔Qdrant Consistency

The `IndexChecksum` class computes a hash of all chunk IDs in the BM25 index and validates it against Qdrant's chunk IDs at startup. If there's a mismatch (e.g., chunks were added to Qdrant but BM25 wasn't rebuilt), the system triggers an automatic BM25 rebuild.

---

## 19. Dependency Injection Graph

All service instances are managed by `app/api/dependencies.py` using `@lru_cache(maxsize=1)` singletons:

```
get_settings() ─────────────┬──► get_model_registry() ──► get_model_manager()
                            │                                    │
                            │                    ┌───────────────┼──────────────┐
                            │                    ▼               ▼              ▼
                            ├──► get_embedding_model()   get_reranker()   (other models)
                            │         │                      │
                            │         │                      │
                            ├──► get_vector_store()          │
                            │         │                      │
                            ├──► get_bm25_store()            │
                            │         │                      │
                            │    ┌────┘     ┌────────────────┘
                            │    ▼          ▼
                            ├──► get_hybrid_retriever()
                            │
                            ├──► get_llm_engine()
                            ├──► get_slm_engine()
                            ├──► get_query_store()
                            ├──► get_index_manager()
                            ├──► get_metrics()
                            ├──► get_analytics_tracker()
                            ├──► get_failure_diagnoser()
                            │
                            ├──► get_corpus_coverage_analyzer()
                            │         (depends on query_store + vector_store)
                            │
                            ├──► get_embedding_quality_checker()
                            │         (depends on vector_store + embedding_model)
                            │
                            └──► get_experiment_engine()
                                      (depends on hybrid_retriever + embedding_model)
```

All dependencies are lazily initialized — calling `get_embedding_model()` for the first time triggers the actual model load.

---

## 20. Error Handling & Fault Tolerance

### Custom Exception Hierarchy

All exceptions inherit from `RAGBaseError`, which includes a `trace_id` for request correlation:

```
RAGBaseError
├── FileValidationError
│   ├── UnsupportedFileTypeError      (.pdf, .docx, etc. not in allowed list)
│   ├── FileSizeLimitError            (>100MB)
│   ├── CorruptedFileError            (cannot parse)
│   └── PasswordProtectedError        (encrypted PDF)
├── NormalizationError                (text normalization failure)
├── ChunkingError
│   └── TokenLimitExceededError       (chunk >512 tokens)
├── EmptyDocumentError                (no extractable text)
├── VectorStoreError                  (Qdrant connection/query failure)
├── BM25IndexError                    (index load/query failure)
├── IndexChecksumError                (BM25↔Qdrant mismatch)
├── RerankerError                     (cross-encoder failure)
├── LLMNotLoadedError                 (LLM accessed before loading)
├── LLMGenerationError                (inference failure)
├── ContextTooLongError               (prompt exceeds context window)
└── GenerationTimeoutError            (LLM inference timeout)
```

### Circuit Breaker Pattern

The `CircuitBreaker` class protects against cascading failures in worker processes:

**States:**
- **CLOSED** (normal): Calls pass through. Failures increment counter.
- **OPEN** (tripped): After `failure_threshold` (5) consecutive failures, all calls are rejected with `CircuitBreakerError` for `recovery_timeout` (60 seconds).
- **HALF_OPEN** (testing): After recovery timeout, allows test calls. 2 consecutive successes → CLOSED. Any failure → OPEN again.

Used for:
- OCR worker (PaddleOCR hangs/crashes)
- Audio worker (Whisper stalls)
- Qdrant connections (network issues)

---

## 21. Storage Layout

| Location | Format | Contents |
|---|---|---|
| `data/uploads/` | Original files | `{uuid}_{original_filename}` — preserved for re-indexing |
| `data/index/current/qdrant/` | Qdrant binary | Vector embeddings (384-dim) + metadata payloads for all chunks |
| `data/index/current/bm25/bm25_index.pkl` | Python pickle | Serialized BM25Okapi object with tokenized corpus |
| `data/index/current/bm25/bm25_metadata.pkl` | Python pickle | Chunk IDs + full metadata dicts aligned with BM25 index |
| `data/query_history.jsonl` | JSONL | One JSON record per line — full query records with chunks, answers, latencies |
| `data/logs/rag-system.log` | JSON lines | Rotating log file (10MB × 5 backups) with structured JSON entries |
| `models/` | Various | Pre-downloaded ML models (SafeTensors, GGUF, CTranslate2) |

---

## 22. Data Flow — End to End

### Ingestion Flow

```
File Upload (POST /upload)
    │
    ▼
Validate (extension, size, filename)
    │
    ▼
Save to data/uploads/{uuid}_{name}
    │
    ▼
Detect modality (.pdf→Document, .png→Image, .mp3→Audio)
    │
    ▼
Background Task: _ingest_file()
    │
    ├─► Document Worker (Docling) → Markdown text + page numbers
    ├─► OCR Worker (EasyOCR) → Text with confidence scores
    └─► Audio Worker (Whisper) → Timestamped transcript
         │
         ▼
    Normalize text (NFC, quotes, whitespace)
         │
         ▼
    Sliding-window chunk (target=480, max=512, overlap=50)
         │
         ▼
    Validate chunks (token limit, required metadata)
         │
         ▼
    Embed chunks (BGE-small, batch=64, 384-dim)
         │
         ▼
    Upsert to Qdrant (vectors + payloads)
         │
         ▼
    Rebuild BM25 index (tokenize + pickle)
         │
         ▼
    Update metrics (corpus_size, upload_latency)
```

### Query Flow

```
User Query (POST /query, SSE stream)
    │
    ▼
Normalize query text
    │
    ▼
Embed query (BGE-small, with query prefix)
    │
    ▼
┌──────────────────┬──────────────────┐
│  Vector Search   │  BM25 Search     │  (parallel, top 50 each)
│  (Qdrant, k=50)  │  (rank-bm25, k=50)│
└────────┬─────────┴────────┬─────────┘
         │                  │
         ▼                  ▼
    RRF Fusion (k=60, deduplicate, tag origins)
         │
         ▼
    Entity Extraction (regex: acronyms, quoted, capitalized, hyphenated)
         │
         ▼
    Entity Injection (BM25 lookup, ≤10/entity, ≤20 global)
         │
         ▼
    Cross-Encoder Reranking (top 15, threshold=0.15, min=5)
         │
    SSE event: "retrieval" → send chunks + debug info to frontend
         │
         ▼
    Build prompt (select system prompt, format context, ChatML)
         │
         ▼
    LLM streaming generation (Qwen2.5-3B, token by token)
         │
    SSE events: "token" → stream each token to frontend
         │
         ▼
    Extract citations from answer
         │
    SSE event: "citations" → send citation list
         │
    SSE event: "done" → send query_id, total_tokens
         │
         ▼
    Persist to QueryStore (JSONL) with full context
         │
         ▼
    Update metrics (query_latency, retrieval_hits, etc.)
```

---

## 23. Deployment & Startup

### Prerequisites

1. **Python 3.11** with venv at `D:/Offline_Rag_V2/.venv/`
2. **Node.js 18+** for the frontend
3. **Qdrant** running at `localhost:6333`
4. **NVIDIA CUDA 12.8** drivers installed
5. **All models downloaded** to `models/` directory

### Backend Startup Sequence

The FastAPI application follows a strict startup lifecycle defined in `app/main.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Setup structured logging (JSON format, rotating files)
    setup_logging(log_dir=Path(settings.paths.logs_dir), level=settings.log_level)

    # 2. Ensure required directories exist
    #    (data/, uploads/, index/, logs/)

    # 3. Initialize index versioning
    #    (creates v1.0.0 + current symlink if first run)

    # 4. Connect to Qdrant
    #    (ensure collection exists with 384-dim, Cosine, HNSW config)

    # 5. Load BM25 index from disk
    #    (validate checksum against Qdrant, rebuild if mismatch)

    # 6. Pre-load embedding model
    #    (BGE-small-en-v1.5 loaded into GPU memory)

    yield  # Application runs

    # Cleanup (not currently needed — models GC'd)
```

### Starting the System

**Backend:**
```
cd D:/Offline_Rag_V2/rag-system
../.venv/Scripts/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```
cd D:/Offline_Rag_V2/frontend
npm run dev
```

**Qdrant:**
```
qdrant.exe  (runs on localhost:6333)
```

### CORS Configuration

The backend allows requests from the Vite dev server:
- `http://localhost:5173`
- `http://127.0.0.1:5173`

All methods, headers, and credentials are allowed.

---

## 24. Hardware Requirements

### Current Development Hardware

| Component | Specification |
|---|---|
| **GPU** | NVIDIA GTX 1650 (4GB VRAM, SM 7.5, Turing architecture) |
| **CUDA** | 12.8 |
| **CPU** | Multi-core (≥4 cores recommended for worker parallelism) |
| **RAM** | ≥16GB recommended |
| **Disk** | ≥20GB for models, additional space for corpus |

### GPU Memory Budget

| Component | VRAM Usage |
|---|---|
| Qwen2.5-3B Q4_K_M (36 layers) | ~2.0 GB |
| BGE-small-en-v1.5 | ~0.1 GB |
| BGE-reranker-base | ~0.4 GB |
| CUDA overhead | ~0.3 GB |
| **Total** | **~2.8 GB / 4.0 GB** |

### Model Storage

| Model | Size |
|---|---|
| BGE-small-en-v1.5 | ~66 MB |
| BGE-reranker-base | ~440 MB |
| Qwen2.5-3B-Q4_K_M | ~2.0 GB |
| Gemma-2-9B-Q4_K_M | ~5.0 GB |
| Faster-Whisper-small | ~461 MB |
| EasyOCR models | ~100 MB |
| **Total** | **~8.1 GB** |

---

## Summary of Key Numbers

| Metric | Value |
|---|---|
| Total Python dependencies | 20 packages |
| Total frontend dependencies | 8 production + 8 dev |
| Total API endpoints | 34 |
| Total Pydantic schemas | 70+ |
| Total ML models | 7 registered (5 active by default) |
| Embedding dimensions | 384 |
| Chunk target size | 480 tokens |
| Chunk hard limit | 512 tokens |
| Chunk overlap | 50 tokens |
| Vector search top-k | 50 |
| BM25 search top-k | 50 |
| RRF fusion constant | k=60 |
| Rerank candidates | 15 |
| Rerank threshold | 0.15 |
| Minimum results safeguard | 5 |
| Entity per-limit | 10 |
| Entity global limit | 20 |
| LLM context window | 32,768 tokens |
| LLM GPU layers | 36 |
| Context budget per prompt | ~6,000 tokens |
| Max upload file size | 100 MB |
| Query history in-memory | 500 records |
| Log rotation | 10 MB × 5 files |
| Backend code | ~5,000+ lines Python |
| Frontend code | ~2,044 Vite modules |
| Routes file | 1,703 lines |
| API models file | 472 lines |

---

*This documentation covers the complete Offline RAG V2 system as of its current state. Every model, library, configuration parameter, API endpoint, data flow, and architectural decision has been documented with exact values from the source code.* -->
