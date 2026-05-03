# InferStream

Building a distributed LLM inference system designed to run across local laptops and machines. The goal is to demonstrate distributed compute, request batching for efficiency, scalable inference throughput, and basic LLM serving concepts.

## Architecture

The system follows a simple distributed layout with an in-memory queue, avoiding the need for complex external brokers like Kafka or Redis.

It consists of three main components:
1. **Frontend UI**: A FastAPI web server hosting a beautiful, modern chat interface for users to submit inference tasks.
2. **Coordinator Service**: The central "brain" of the system. It handles incoming client requests, manages an in-memory queue grouped by task length (`SHORT`, `MEDIUM`, `LONG`), and distributes tasks to available workers via gRPC.
3. **Worker Nodes**: Distributed compute layers that load the model into memory (e.g., DistilGPT2 via Hugging Face), poll the coordinator for work, and run the PyTorch LLM generation loops.

## Layout

- `src/inferstream/` — installable package.
  - `coordinator/`: Central service logic and queueing.
  - `frontend/`: FastAPI web server and UI static assets.
  - `worker/`: Distributed compute node logic.
  - `inference/`: PyTorch LLM execution and KV-cache management.
  - `grpc/`: Generated Python stubs from `.proto` definitions.
  - `metrics/`: Generic metric registry for tracking throughput, latency, and memory.
- `proto/` — `.proto` sources; regenerate Python stubs with `./scripts/regenerate_proto.sh`.
- `tests/` — pytest suite and standalone local inference demo.

## Running the Distributed System

To spin up the full distributed system locally (Coordinator, Frontend UI, and Worker Node), you can use the provided convenience script:

```bash
./scripts/start_all.sh
```

This will boot all three services in the correct order in the background. It will automatically shut them all down cleanly when you press `Ctrl+C`.

Once running, open `http://localhost:8000/` in your browser to interact with the UI.

## Commands & Testing

You can run the core inference engine locally without starting the distributed servers by running the standalone demo test:

```bash
uv sync --group dev             # install project + dev deps (pytest)
uv run pytest tests/test_demo.py -s  # run the local standalone inference demo
./scripts/regenerate_proto.sh   # compile protobufs to python stubs
```

## Publishing (PyPI)

Build artifacts land in `dist/` (gitignored). Upload requires a [PyPI API token](https://pypi.org/manage/account/token/):

```bash
uv build
UV_PUBLISH_TOKEN=<your-pypi-token> uv publish
```

On GitHub Actions you can use [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) instead of a long-lived token.