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

You can optionally specify a Hugging Face model to load (defaults to `Qwen/Qwen2.5-3B-Instruct`):
```bash
./scripts/start_all.sh --model meta-llama/Llama-3.1-8B-Instruct
```

This will boot all three services in the correct order in the background. It will automatically shut them all down cleanly when you press `Ctrl+C`.

Once running, open `http://localhost:8000/` in your browser to interact with the UI.

### Running Across Multiple Machines (e.g. Raspberry Pi + MacBook)

You can run the lightweight Coordinator and Frontend on a low-power device like a Raspberry Pi, while running the heavy inference Worker on your MacBook.

1. **On the Raspberry Pi (Coordinator + Frontend)**:
   Ensure you have a 64-bit OS installed (like Raspberry Pi OS Lite 64-bit) to support Python dependencies.
   Use the combined launcher to start both services together — **Ctrl+C tears them both down**:
   ```bash
   uv run inferstream-serve
   ```
   You can also specify ports if needed:
   ```bash
   uv run inferstream-serve --coordinator-port 50051 --frontend-port 8000
   ```
   *Note: Both services bind to `0.0.0.0` by default, so they are automatically accessible on your local network.*

2. **On your MacBook (Worker)**:
   Point the worker at the Pi using either its IP address or mDNS hostname.
   ```bash
   uv run inferstream-worker \
       --coordinator 10.107.8.126:50051 \
       --model Qwen/Qwen2.5-3B-Instruct \
       --max-ram-gb 16.0 \
       --max-batch-size 8 \
       --max-seq-len 2048
   ```
   You can also use the mDNS hostname (`pi.local:50051`) directly — the worker automatically pre-resolves it via the OS before handing the address to gRPC:
   ```bash
   uv run inferstream-worker --coordinator pi.local:50051
   ```
   > **Why not just use `pi.local` everywhere?** `.local` hostnames are resolved via **mDNS** (Bonjour/Avahi), not standard DNS. Your browser and `ping` support mDNS natively, but gRPC's internal resolver does not. InferStream works around this by resolving the hostname through Python's `socket` module (which honours the OS mDNS stack) before creating the gRPC channel.

   *Note: If you use a gated model like Llama 3.1, you must first authenticate by running `uv run huggingface-cli login`.*

3. **Accessing the UI**:
   Open `http://pi.local:8000` (or `http://10.107.8.126:8000`) in your MacBook's browser to access the chat interface.

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

## Observability & Metrics

The system has a built-in observability stack powered by **OpenTelemetry**. This allows you to track inference performance, token generation latency, KV cache size, and more.

You can configure metric exporters by setting the `INFERSTREAM_METRICS_EXPORTER` environment variable.

### Exporter Options:
- `inmemory` (Default): Stores metrics in process memory. Ideal for testing and single-process scripts.
- `console`: Prints out periodic metric updates directly to `stdout`.
- `otlp`: Forwards telemetry data over gRPC/HTTP directly to an OTEL collector.
- `prometheus`: Spins up a Prometheus scrape endpoint on the worker node.

### Visualizing with Prometheus & Grafana

To scrape and visualize worker metrics using Prometheus:

1. Start your worker node with the Prometheus exporter enabled:
   ```bash
   export INFERSTREAM_METRICS_EXPORTER=prometheus
   export PROMETHEUS_PORT=9090
   uv run python src/inferstream/worker/server.py
   ```
2. Configure your **Prometheus server** (`prometheus.yml`) to scrape the worker node endpoint:
   ```yaml
   scrape_configs:
     - job_name: 'inferstream_worker'
       scrape_interval: 5s
       static_configs:
         - targets: ['localhost:9090']
   ```
3. Connect your Prometheus data source to **Grafana** to visualize core metrics such as:
   - `time_per_output_token_ms`: Token generation throughput latency.
   - `kv_cache_size_mb`: Memory utilization of the continuous batching engine.
   - `total_generated_tokens`: Total generated output volume.