# InferStream

Building a distributed LLM inference system designed to run across local laptops and machines. The goal is to demonstrate distributed compute, request batching for efficiency, scalable inference throughput, and basic LLM serving concepts.

## Architecture

The system follows a simple distributed layout with an in-memory queue, avoiding the need for complex external brokers like Kafka or Redis.

It consists of three main components:
1. **Coordinator Service**: The central "brain" of the system. It handles incoming client requests, assigns `request_id`s, manages an in-memory queue, and routes requests to workers.
2. **Worker Nodes**: Distributed compute layers that load the model into memory (e.g., DistilGPT2 via Hugging Face), poll the coordinator for batches of work via gRPC, and run the actual PyTorch LLM forward passes (batching multiple requests when possible to improve throughput).
3. **gRPC Layer**: Internal communication between the Coordinator and Workers is handled strictly via gRPC for high efficiency.

## Layout

- `src/inferstream/` — installable package.
  - `coordinator/`: Central service logic and queueing.
  - `worker/`: Distributed compute node logic.
  - `inference/`: Actual PyTorch LLM execution and KV-cache management.
  - `grpc/`: Generated Python stubs from `.proto` definitions.
  - `metrics/`: Generic metric registry for tracking throughput, latency, and memory.
- `proto/` — `.proto` sources; regenerate Python stubs with `./scripts/regenerate_proto.sh` (requires `uv sync --group dev`). Generated `*_pb2.py` files under `src/` are committed.
- `tests/` — pytest suite.

## Commands

```bash
uv sync --group dev    # install project + dev deps (pytest)
uv build               # wheel + sdist in dist/
uv run inferstream     # demo CLI (downloads model on first run)
uv run pytest          # tests
./scripts/regenerate_proto.sh   # protobuf → src/inferstream/grpc/generated/
```

### Running the Services

To spin up the system locally, you need to start the Coordinator and the Frontend UI.

1. **Start the Coordinator (gRPC Backend)**:
   ```bash
   uv run inferstream-coordinator
   ```
   This will run on `[::]:50051`.

2. **Start the Frontend UI (FastAPI Server)**:
   In a new terminal window:
   ```bash
   uv run inferstream-frontend
   ```
   This will bind to `http://0.0.0.0:8000/`. You can open this in your browser to interact with the chat UI.

## Publishing (PyPI)

Build artifacts land in `dist/` (gitignored). Upload requires a [PyPI API token](https://pypi.org/manage/account/token/):

```bash
uv build
UV_PUBLISH_TOKEN=<your-pypi-token> uv publish
```

On GitHub Actions you can use [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) instead of a long-lived token.