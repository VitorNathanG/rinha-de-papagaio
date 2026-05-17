"""
Walk the canonical rinha test-data.json once, POST every entry to the
backend, and dump the entries whose verdict disagrees with the ground
truth `expected_approved`.

Output written to data/mismatches.json so we can investigate them offline
(no need to keep the backend up afterwards).
"""
import http.client
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
TEST_DATA = Path("/projects/rinha-de-backend-2026/test/test-data.json")
OUT_PATH = ROOT / "data" / "mismatches.json"
HOST = "localhost"
PORT = 9999
PATH = "/fraud-score"


def main():
    payload = json.loads(TEST_DATA.read_text())
    entries = payload["entries"]
    n = len(entries)
    print(f"[mismatch] sweeping {n:,} entries against {HOST}:{PORT}{PATH}",
          flush=True)

    # Reuse one HTTP/1.1 keep-alive connection — same hot path the rinha
    # test exercises (k6 default is connection reuse).
    conn = http.client.HTTPConnection(HOST, PORT, timeout=2.0)
    headers = {"Content-Type": "application/json"}

    mismatches = []
    t0 = time.time()
    next_report = 5000
    for i, entry in enumerate(entries):
        expected = entry["expected_approved"]
        # Use compact JSON (no spaces after colons/commas) to match exactly
        # what k6's JSON.stringify produces — our backend's byte parser
        # scans for "amount":N with no whitespace.
        body_bytes = json.dumps(entry["request"], separators=(",", ":")).encode()
        try:
            conn.request("POST", PATH, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            body = json.loads(data)
        except Exception as e:
            mismatches.append({
                "index": i,
                "expected_approved": expected,
                "error": str(e),
                "request": entry["request"],
            })
            # Reconnect after an error to keep the keep-alive clean.
            conn.close()
            conn = http.client.HTTPConnection(HOST, PORT, timeout=2.0)
            continue

        if body.get("approved") != expected:
            mismatches.append({
                "index": i,
                "expected_approved": expected,
                "actual_approved": body.get("approved"),
                "actual_fraud_score": body.get("fraud_score"),
                "request": entry["request"],
            })

        if i + 1 == next_report:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[mismatch] {i+1:,}/{n:,}  "
                  f"({rate:.0f} req/s)  "
                  f"mismatches so far: {len(mismatches)}", flush=True)
            next_report += 5000

    elapsed = time.time() - t0
    print(f"[mismatch] done in {elapsed:.1f}s  "
          f"({n/elapsed:.0f} req/s)")
    print(f"[mismatch] total mismatches: {len(mismatches)}")

    fp = sum(1 for m in mismatches
             if m.get("expected_approved") is True
             and m.get("actual_approved") is False)
    fn = sum(1 for m in mismatches
             if m.get("expected_approved") is False
             and m.get("actual_approved") is True)
    errs = sum(1 for m in mismatches if "error" in m)
    print(f"[mismatch]   fp={fp}  fn={fn}  http_errors={errs}")

    OUT_PATH.write_text(json.dumps(mismatches, indent=2))
    print(f"[mismatch] wrote {OUT_PATH}")

    # Echo each mismatch summary
    for m in mismatches:
        print(json.dumps(m))


if __name__ == "__main__":
    sys.exit(main() or 0)
