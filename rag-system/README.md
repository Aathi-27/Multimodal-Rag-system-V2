# Offline RAG System — Backend

> See the **[main README](../README.md)** for full project documentation, architecture, and workflow.

This directory contains the FastAPI backend for the Multimodal RAG platform.

## Quick Start

```bash
# From project root
cd rag-system
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Architecture

6-layer pipeline: **Ingestion → Processing → Indexing → Retrieval → Reranking → Generation**

| Component | Technology |
|-----------|-----------|
| Document Parsing | Docling (DocumentConverter → Markdown) |
| Chunking | Custom sliding window (480 target, 512 max, 50 overlap) |
| Embedding | BAAI/bge-small-en-v1.5 (384 dim, CUDA) |
| Vector DB | Qdrant (persistent, HNSW m=16, ef=100) |
| Keyword Search | rank-bm25 (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | BAAI/bge-reranker-base (cross-encoder, CPU) |
| LLM | Qwen2.5-1.5B-Instruct Q4_K_M via llama-cpp-python |
| Visual Search | CLIP ViT-B/32 (512-dim, separate Qdrant collection) |
| Speech-to-Text | faster-whisper small (CTranslate2, int8) |
| OCR | EasyOCR (English, CPU) |

## Quick Start

### 1. Download Models

```bash
python scripts/download_models.py --all
```

Models are stored in `models/` (external volume, not baked into Docker):
- `models/llm/gemma-2-9b-it-Q4_K_M.gguf`
- `models/embeddings/bge-small-en-v1.5/`
- `models/reranker/bge-reranker-base/`
- `models/whisper/faster-whisper-small/`
- `models/ocr/paddleocr/`

### 2. Initialize Index

```bash
python scripts/init_index.py --version v1.0.0
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 -v $(pwd)/data/index/current/qdrant:/qdrant/storage qdrant/qdrant:v1.7.0
```

### 4. Run Application

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Docker Compose (Full Stack)

```bash
cd docker
docker compose up -d
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload document, image, or audio file |
| `POST` | `/query` | Query with SSE streaming + citations |
| `GET` | `/health` | System health check |

## Health Check

```bash
python scripts/health_check.py
```

## Configuration

Edit `config.yaml` or use environment variable overrides:

```bash
RAG_LLM__GPU_LAYERS=0        # CPU-only mode
RAG_QDRANT__HOST=qdrant       # Docker service name
RAG_LLM__TEMPERATURE=0.5     # Lower temperature
```

## Project Structure

```
rag-system/
├── app/
│   ├── main.py                 # FastAPI entry
│   ├── api/                    # Routes, Pydantic models, DI
│   ├── ingestion/              # Workers: document, OCR, audio
│   ├── processing/             # Normalization, chunking, validation
│   ├── retrieval/              # Qdrant, BM25, hybrid RRF, reranker
│   ├── generation/             # llama.cpp engine, prompt templates
│   ├── models/                 # Model registry, manager, embeddings
│   ├── versioning/             # Index versioning, checksum validation
│   ├── observability/          # Metrics, tracing, structured logging
│   ├── config/                 # Settings loader (YAML + env)
│   └── utils/                  # Circuit breaker, exceptions
├── data/                       # Runtime data (uploads, indexes, logs)
├── models/                     # ML models (mounted volume)
├── scripts/                    # Download, init, health check
├── tests/                      # Unit & integration tests
├── docker/                     # Dockerfile, docker-compose.yml
├── config.yaml                 # Main configuration
└── requirements.txt            # Python dependencies
```

## Requirements

- **GPU**: Optional NVIDIA CUDA (8GB VRAM ideal)
- **CPU**: Supported (set `gpu_layers: 0`)
- **RAM**: 16GB minimum recommended
- **Disk**: SSD required
- **Python**: 3.10+

## License

Internal / Proprietary
