#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p src/inferstream/grpc/generated

uv run python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/inferstream/grpc/generated \
  --grpc_python_out=src/inferstream/grpc/generated \
  proto/inferstream/v1/coordinator.proto

# Fix python imports to be absolute
sed -i.bak 's/from inferstream.v1/from inferstream.grpc.generated.inferstream.v1/g' src/inferstream/grpc/generated/inferstream/v1/*.py
rm -f src/inferstream/grpc/generated/inferstream/v1/*.bak

echo "OK: protobuf Python stubs under src/inferstream/grpc/generated/"
