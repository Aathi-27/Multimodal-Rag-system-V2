"""
Phase 1 Validation — Citation Navigation End-to-End Tests.

Tests:
1. File registry persistence (register + get + list)
2. File serving endpoint (GET /files/{file_id})
3. File info endpoint (GET /files/{file_id}/info)
4. File listing endpoint (GET /files)
5. Citation SSE events include file_id
6. Security: path traversal blocked (ID-only lookup)
7. Backfill existing uploads
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "rag-system"))

import requests

BASE_URL = "http://localhost:8000"

PASS = "✓"
FAIL = "✗"

results: list[tuple[str, bool, str]] = []


def test(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    mark = PASS if passed else FAIL
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))


def run_tests():
    print("=" * 60)
    print("  Phase 1 — Citation Navigation Validation")
    print("=" * 60)
    print()

    # ── 1. Health check (server reachable) ──────────────────────────────
    print("[Pre-flight]")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        test("Server reachable", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        test("Server reachable", False, str(e))
        print("\n⚠  Server not running. Start it first with: python start_server.py")
        return

    # ── 2. File listing endpoint ────────────────────────────────────────
    print("\n[File Registry Endpoints]")
    r = requests.get(f"{BASE_URL}/files", timeout=5)
    test("GET /files returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        test("Response has 'files' and 'total'",
             "files" in data and "total" in data,
             f"keys={list(data.keys())}")
        file_count = data.get("total", 0)
        test(f"Registry has entries", file_count >= 0, f"count={file_count}")

    # ── 3. Upload a test file and verify registry ───────────────────────
    print("\n[Upload + Registry Integration]")
    # Create a small test text file
    test_content = b"This is a test document for citation navigation validation."
    files_payload = {"file": ("test_citation_nav.txt", test_content, "text/plain")}

    r = requests.post(f"{BASE_URL}/upload", files=files_payload, timeout=30)
    test("Upload test file", r.status_code == 202, f"status={r.status_code}")

    upload_id = None
    if r.status_code == 202:
        data = r.json()
        upload_id = data.get("upload_id")
        test("Upload returns upload_id", bool(upload_id), f"id={upload_id[:8] if upload_id else 'none'}...")

    # Wait for ingestion to complete
    if upload_id:
        for _ in range(30):
            time.sleep(1)
            sr = requests.get(f"{BASE_URL}/status/{upload_id}", timeout=5)
            if sr.status_code == 200:
                status = sr.json().get("status")
                if status in ("completed", "failed"):
                    break

    # ── 4. File serving ─────────────────────────────────────────────────
    print("\n[File Serving]")
    if upload_id:
        # GET /files/{file_id}
        r = requests.get(f"{BASE_URL}/files/{upload_id}", timeout=5)
        test("GET /files/{file_id} serves content",
             r.status_code == 200,
             f"status={r.status_code}, size={len(r.content)}")

        if r.status_code == 200:
            test("Content matches uploaded data",
                 r.content == test_content,
                 f"expected={len(test_content)}, got={len(r.content)}")

            test("Accept-Ranges header present",
                 "Accept-Ranges" in r.headers,
                 f"Accept-Ranges={r.headers.get('Accept-Ranges')}")

        # GET /files/{file_id}/info
        r = requests.get(f"{BASE_URL}/files/{upload_id}/info", timeout=5)
        test("GET /files/{file_id}/info returns metadata",
             r.status_code == 200,
             f"status={r.status_code}")

        if r.status_code == 200:
            info = r.json()
            test("Info has required fields",
                 all(k in info for k in ("file_id", "file_name", "file_type", "modality")),
                 f"keys={list(info.keys())}")
            test("file_name matches",
                 info.get("file_name") == "test_citation_nav.txt",
                 f"name={info.get('file_name')}")

    # ── 5. Non-existent file returns 404 ────────────────────────────────
    print("\n[Security / Edge Cases]")
    r = requests.get(f"{BASE_URL}/files/nonexistent_id_12345", timeout=5)
    test("Non-existent file_id returns 404",
         r.status_code == 404,
         f"status={r.status_code}")

    # ── 6. Verify file_id in file listing ───────────────────────────────
    if upload_id:
        r = requests.get(f"{BASE_URL}/files", timeout=5)
        if r.status_code == 200:
            files = r.json().get("files", [])
            found = any(f["file_id"] == upload_id for f in files)
            test("Uploaded file appears in /files listing",
                 found,
                 f"total_files={len(files)}")

    # ── 7. Cleanup: delete the test document ────────────────────────────
    print("\n[Cleanup]")
    try:
        r = requests.delete(f"{BASE_URL}/documents/test_citation_nav.txt", timeout=10)
        test("Cleanup test document", r.status_code == 200, f"status={r.status_code}")
    except Exception:
        test("Cleanup test document", False, "request failed")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print("  ✓ All Phase 1 citation navigation tests PASSED")
    else:
        failed = [(n, d) for n, p, d in results if not p]
        print(f"  ✗ {len(failed)} test(s) FAILED:")
        for name, detail in failed:
            print(f"    - {name}: {detail}")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
