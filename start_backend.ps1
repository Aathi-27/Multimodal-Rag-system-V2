Set-Location d:\Offline_Rag_V2\rag-system
Write-Host "Starting RAG Backend..." -ForegroundColor Cyan
Write-Host ""
& d:\Offline_Rag_V2\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
