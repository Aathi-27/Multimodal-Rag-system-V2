# Multi-stage build for RAG System
# Models are NOT baked in — mount /models as a volume

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY rag-system/ /app/

# Create data directories
RUN mkdir -p /data/uploads /data/index /data/logs

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    RAG_PATHS__DATA_DIR=/data \
    RAG_PATHS__UPLOADS_DIR=/data/uploads \
    RAG_PATHS__INDEX_DIR=/data/index \
    RAG_PATHS__LOGS_DIR=/data/logs \
    RAG_PATHS__MODELS_DIR=/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
