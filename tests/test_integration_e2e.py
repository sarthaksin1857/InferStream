"""End-to-end integration test for the InferStream distributed stack.

This test mirrors the --simulate mode of start_all.sh, but runs as a proper
pytest test so it can be debugged, CI'd, and gives readable assertions.

What it does
------------
1. Starts the coordinator, frontend, and worker as subprocesses (same as
   start_all.sh) with their stdout/stderr piped to log files under /tmp so
   you can inspect them after a failure.
2. Waits for each service to be healthy before moving on.
3. Submits NUM_REQUESTS prompts to POST /api/generate.
4. Polls GET /api/result/{id} until all requests reach COMPLETED.
5. Measures wall-clock latency per request (submit → completed) and overall
   throughput (total tokens / total wall time).
6. Asserts that every request completed and prints a summary table.

Running
-------
    uv run pytest tests/test_integration_e2e.py -s -v

The -s flag lets print() output through so you see the step-by-step log
while the test is running. Without it pytest captures output and only shows
it on failure.

Timeout
-------
The test has a generous MAX_WAIT_SECONDS budget (default 300 s) for all
requests to complete. On a CPU-only machine or with a heavier model you may
need to increase it. Generation time is dominated by the number of tokens
requested (NUM_REQUESTS × MAX_NEW_TOKENS × slots), not request count.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Configuration — tweak these without touching the test logic
# ---------------------------------------------------------------------------

COORDINATOR_ADDR = "localhost:50051"
FRONTEND_URL     = "http://localhost:8000"

NUM_REQUESTS     = 16      # how many prompts to fire
MAX_NEW_TOKENS   = 50      # passed as "short" length — see worker LENGTH_MAPPING
POLL_INTERVAL    = 2.0     # seconds between GetResult sweeps
MODEL_LOAD_WAIT   = 60     # seconds for frontend HTTP to become available
WORKER_EXTRA_WAIT = 15    # extra seconds after frontend is up for model+gRPC
MAX_WAIT_SECONDS  = 300   # hard timeout for all requests to complete

PROMPTS: List[str] = [
    "Once upon a time in distributed systems,",
    "What is the meaning of life?",
    "How do you build a large language model?",
    "To be or not to be, that is the question.",
    "The quick brown fox jumps over the lazy dog.",
    "A long time ago in a galaxy far, far away...",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, never mind how long,",
    "It is a truth universally acknowledged,",
    "In the beginning God created the heavens and the earth.",
    "Two roads diverged in a yellow wood,",
    "I wandered lonely as a cloud",
    "Water water everywhere and all the boards did shrink;",
    "Shall I compare thee to a summer's day?",
    "Fourscore and seven years ago our fathers brought forth",
    "Ask not what your country can do for you,",
]

# ---------------------------------------------------------------------------
# Internal bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class RequestRecord:
    """Per-request timing and result storage."""
    request_id:     str
    prompt:         str
    submit_time:    float             = field(default_factory=time.monotonic)
    complete_time:  Optional[float]   = None
    generated_text: Optional[str]     = None
    status:         str               = "PENDING"

    @property
    def latency_seconds(self) -> Optional[float]:
        if self.complete_time is not None:
            return self.complete_time - self.submit_time
        return None


# ---------------------------------------------------------------------------
# HTTP helpers — use stdlib only (no requests dependency)
# ---------------------------------------------------------------------------

def _http_post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _http_ok(url: str) -> None:
    """Perform a GET and raise on non-2xx status, without parsing the body."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"HTTP {resp.status} from {url}")


def _wait_for_http(url: str, timeout: float = 30.0, interval: float = 1.0) -> None:
    """Block until the URL returns HTTP 2xx or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _http_ok(url)
            return
        except Exception:
            time.sleep(interval)
    raise RuntimeError(f"Service at {url} did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Service lifecycle helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR      = Path("/tmp/inferstream_integration_test")


def _start_service(
    args: List[str],
    log_name: str,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Launch a subprocess, redirecting stdout+stderr to a log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    log_fh   = open(log_path, "w")
    proc = subprocess.Popen(
        args,
        stdout=log_fh,
        stderr=log_fh,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, **(env or {})},
    )
    print(f"    [pid={proc.pid}] {' '.join(args[:3])} … → {log_path}")
    return proc


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_e2e_distributed_inference() -> None:
    """
    Full-stack integration test: coordinator → worker → frontend → HTTP client.

    Submits NUM_REQUESTS prompts, waits for all to complete, and asserts
    correctness + prints a timing summary table.
    """
    procs: List[subprocess.Popen] = []

    try:
        # ===================================================================
        # PHASE 1 — Start services
        # ===================================================================
        print(f"\n{'='*70}")
        print("  PHASE 1 — Starting services")
        print(f"{'='*70}")

        coord = _start_service(
            ["uv", "run", "inferstream-coordinator"],
            "coordinator.log",
        )
        procs.append(coord)

        # Give coordinator a moment to bind its port before the frontend
        # tries to connect.
        time.sleep(2)

        frontend = _start_service(
            ["uv", "run", "inferstream-frontend"],
            "frontend.log",
        )
        procs.append(frontend)

        worker = _start_service(
            ["uv", "run", "python", "src/inferstream/worker/server.py"],
            "worker.log",
        )
        procs.append(worker)

        # ===================================================================
        # PHASE 2 — Wait for frontend HTTP health
        # ===================================================================
        print(f"\n{'='*70}")
        print("  PHASE 2 — Waiting for frontend to be ready")
        print(f"{'='*70}")

        print(f"  Waiting up to {MODEL_LOAD_WAIT}s for frontend to be ready…")
        # Poll /docs — a lightweight FastAPI built-in endpoint that
        # doesn't depend on static files or gRPC being connected yet.
        _wait_for_http(
            f"{FRONTEND_URL}/docs",
            timeout=MODEL_LOAD_WAIT,
        )
        print("  ✓ Frontend is up")

        # Wait for the worker to finish loading the model and register
        # with the coordinator over gRPC before we submit requests.
        print(f"  Waiting {WORKER_EXTRA_WAIT}s for worker model load + gRPC registration…")
        time.sleep(WORKER_EXTRA_WAIT)
        print("  ✓ Worker should be ready")

        # ===================================================================
        # PHASE 3 — Submit requests
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"  PHASE 3 — Submitting {NUM_REQUESTS} requests")
        print(f"{'='*70}")

        records: List[RequestRecord] = []
        for i, prompt in enumerate(PROMPTS[:NUM_REQUESTS]):
            resp = _http_post(
                f"{FRONTEND_URL}/api/generate",
                {"prompt": prompt, "length": "short"},
            )
            req_id = resp["request_id"]
            records.append(RequestRecord(request_id=req_id, prompt=prompt))
            print(f"  [{i+1:02d}/{NUM_REQUESTS}] {req_id}  '{prompt[:45]}'")

        by_id: Dict[str, RequestRecord] = {r.request_id: r for r in records}

        # ===================================================================
        # PHASE 4 — Poll until all complete
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"  PHASE 4 — Polling (timeout={MAX_WAIT_SECONDS}s, interval={POLL_INTERVAL}s)")
        print(f"{'='*70}")

        start_poll = time.monotonic()
        pending_ids = set(by_id.keys())
        completed_count = 0

        while pending_ids:
            elapsed = time.monotonic() - start_poll
            assert elapsed < MAX_WAIT_SECONDS, (
                f"Timed out after {MAX_WAIT_SECONDS}s — "
                f"{len(pending_ids)}/{NUM_REQUESTS} requests still pending.\n"
                f"Check logs in {LOG_DIR}/"
            )

            time.sleep(POLL_INTERVAL)

            still_pending = set()
            for rid in pending_ids:
                try:
                    result = _http_get(f"{FRONTEND_URL}/api/result/{rid}")
                except Exception as exc:
                    print(f"  Warning: GetResult failed for {rid}: {exc}")
                    still_pending.add(rid)
                    continue

                status = result.get("status", "UNKNOWN")
                rec    = by_id[rid]
                rec.status = status

                if status == "COMPLETED":
                    rec.complete_time   = time.monotonic()
                    rec.generated_text  = result.get("generated_text", "")
                    completed_count += 1
                    print(
                        f"  ✔ [{completed_count:02d}/{NUM_REQUESTS}] "
                        f"{rid}  latency={rec.latency_seconds:.1f}s"
                    )
                else:
                    still_pending.add(rid)

            pending_ids = still_pending

            if pending_ids:
                status_counts: Dict[str, int] = {}
                for rid in pending_ids:
                    s = by_id[rid].status
                    status_counts[s] = status_counts.get(s, 0) + 1
                status_str = ", ".join(f"{v}×{k}" for k, v in sorted(status_counts.items()))
                print(
                    f"  ⏳ {len(pending_ids)} remaining "
                    f"({elapsed:.0f}s elapsed) — {status_str}"
                )

        total_wall = time.monotonic() - start_poll

        # ===================================================================
        # PHASE 5 — Results + metrics
        # ===================================================================
        print(f"\n{'='*70}")
        print("  PHASE 5 — Results")
        print(f"{'='*70}")

        for rec in sorted(records, key=lambda r: r.latency_seconds or 0):
            text_preview = (rec.generated_text or "")[:60].replace("\n", " ")
            print(
                f"\n  Prompt:  {rec.prompt}\n"
                f"  Output:  {text_preview}…\n"
                f"  Latency: {rec.latency_seconds:.2f}s"
            )

        # ===================================================================
        # PHASE 6 — Throughput summary
        # ===================================================================
        latencies   = [r.latency_seconds for r in records if r.latency_seconds is not None]
        total_tokens = NUM_REQUESTS * MAX_NEW_TOKENS  # approximate (short=50)
        throughput   = total_tokens / total_wall if total_wall > 0 else 0

        print(f"\n{'='*70}")
        print("  TIMING SUMMARY")
        print(f"{'='*70}")
        print(f"  Requests completed : {NUM_REQUESTS}")
        print(f"  Total wall time    : {total_wall:.1f}s")
        print(f"  Approx throughput  : {throughput:.1f} tokens/s")
        if latencies:
            print(f"  Latency  min       : {min(latencies):.1f}s")
            print(f"  Latency  max       : {max(latencies):.1f}s")
            print(f"  Latency  avg       : {sum(latencies)/len(latencies):.1f}s")
        print(f"  Log dir            : {LOG_DIR}/")
        print(f"{'='*70}\n")

        # ===================================================================
        # ASSERTIONS
        # ===================================================================
        assert len([r for r in records if r.status == "COMPLETED"]) == NUM_REQUESTS, (
            f"Not all requests completed. Statuses: "
            f"{[(r.request_id[:8], r.status) for r in records]}"
        )
        for rec in records:
            assert rec.generated_text, (
                f"Request {rec.request_id} completed but generated_text is empty"
            )
            assert rec.latency_seconds is not None and rec.latency_seconds > 0, (
                f"Request {rec.request_id} has unexpected latency {rec.latency_seconds}"
            )

    finally:
        # ===================================================================
        # TEARDOWN — always kill spawned processes
        # ===================================================================
        print("\n  Tearing down services…")
        for proc in reversed(procs):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("  ✓ All services stopped.")


if __name__ == "__main__":
    # Allow running directly: `uv run python tests/test_integration_e2e.py`
    test_e2e_distributed_inference()
