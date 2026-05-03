#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
uv run python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/inferstream/grpc/generated \
  proto/inferstream/v1/inference_job.proto
echo "OK: protobuf Python stubs under src/inferstream/grpc/generated/"
