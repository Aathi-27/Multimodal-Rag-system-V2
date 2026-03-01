#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# RAG System Startup Script
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo "=========================================="
echo "  RAG System - Startup"
echo "=========================================="

# 1. Run health check
echo "[1/3] Running health check..."
python scripts/health_check.py
if [ $? -ne 0 ]; then
    echo "Health check failed. Fix issues above before starting."
    exit 1
fi

# 2. Initialize index if needed
echo "[2/3] Initializing index..."
python scripts/init_index.py --version v1.0.0

# 3. Start the application
echo "[3/3] Starting FastAPI server..."
echo "Server will be available at http://0.0.0.0:8000"
echo "API docs at http://0.0.0.0:8000/docs"
echo "=========================================="

uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
