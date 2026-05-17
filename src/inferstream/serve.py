"""inferstream-serve — launch coordinator + frontend together.

Usage:
    inferstream-serve [--coordinator-port 50051] [--frontend-port 8000]

Ctrl+C tears both processes down cleanly.
"""

import argparse
import signal
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the InferStream coordinator and frontend together."
    )
    parser.add_argument(
        "--coordinator-port", type=int, default=50051,
        help="Port for the gRPC coordinator (default: 50051)"
    )
    parser.add_argument(
        "--frontend-port", type=int, default=8000,
        help="Port for the HTTP frontend (default: 8000)"
    )
    args = parser.parse_args()

    coordinator_addr = f"localhost:{args.coordinator_port}"

    coordinator_cmd = [sys.executable, "-m", "inferstream.coordinator.server"]
    frontend_cmd = [
        sys.executable, "-m", "inferstream.frontend.server",
        "--coordinator", coordinator_addr,
        "--port", str(args.frontend_port),
    ]

    procs: list[subprocess.Popen] = []

    def _shutdown(signum=None, frame=None):
        print("\n[inferstream-serve] Shutting down...", flush=True)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        # Give them a moment to exit gracefully, then force-kill.
        deadline = time.monotonic() + 5.0
        for p in procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()
        print("[inferstream-serve] All processes stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[inferstream-serve] Starting coordinator on :{args.coordinator_port}", flush=True)
    procs.append(subprocess.Popen(coordinator_cmd))

    # Small delay so the coordinator is listening before the frontend tries to
    # connect on startup.
    time.sleep(1.0)

    print(
        f"[inferstream-serve] Starting frontend on :{args.frontend_port} "
        f"(coordinator → {coordinator_addr})",
        flush=True,
    )
    procs.append(subprocess.Popen(frontend_cmd))

    print("[inferstream-serve] Running. Press Ctrl+C to stop.", flush=True)

    # Wait until one of the children dies unexpectedly, then tear everything down.
    while True:
        for p in procs:
            if p.poll() is not None:
                print(
                    f"[inferstream-serve] Process {p.pid} exited unexpectedly "
                    f"(returncode={p.returncode}). Shutting down.",
                    flush=True,
                )
                _shutdown()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
