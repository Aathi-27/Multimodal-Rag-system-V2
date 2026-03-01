"""Quick smoke test for all new API endpoints."""
import sys
import time
import requests

BASE = "http://localhost:8000"

def test(name, method, url, expected_status=200):
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        elif method == "POST":
            r = requests.post(url, timeout=10)
        elif method == "DELETE":
            r = requests.delete(url, timeout=10)
        ok = r.status_code == expected_status
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name}: {r.status_code} {'OK' if ok else 'FAIL'}")
        if ok and method == "GET":
            data = r.json()
            # Print a short preview
            keys = list(data.keys()) if isinstance(data, dict) else []
            print(f"    Keys: {keys[:8]}")
        return ok
    except Exception as e:
        print(f"  ✗ {name}: ERROR - {e}")
        return False

print("=" * 60)
print("Smoke Testing New API Endpoints")
print("=" * 60)

# Wait for server
print("\nWaiting for server...")
for _ in range(30):
    try:
        requests.get(f"{BASE}/health", timeout=2)
        print("  Server is up!\n")
        break
    except Exception:
        time.sleep(2)
else:
    print("  Server not responding after 60s. Exiting.")
    sys.exit(1)

results = []

print("── Existing Endpoints ──")
results.append(test("GET /health", "GET", f"{BASE}/health"))
results.append(test("GET /documents", "GET", f"{BASE}/documents"))
results.append(test("GET /index/health", "GET", f"{BASE}/index/health"))
results.append(test("GET /analytics", "GET", f"{BASE}/analytics"))

print("\n── NEW: Query History ──")
results.append(test("GET /queries", "GET", f"{BASE}/queries"))
results.append(test("GET /queries/summary", "GET", f"{BASE}/queries/summary"))
results.append(test("GET /queries/nonexist", "GET", f"{BASE}/queries/nonexist", 404))

print("\n── NEW: Metrics ──")
results.append(test("GET /metrics", "GET", f"{BASE}/metrics"))

print("\n── NEW: Resources ──")
results.append(test("GET /resources", "GET", f"{BASE}/resources"))

print("\n── NEW: Versions ──")
results.append(test("GET /versions", "GET", f"{BASE}/versions"))

print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
print(f"Results: {passed}/{total} passed")
print("=" * 60)

if passed < total:
    sys.exit(1)
