## What
Building a inference system which can run on ones local laptop. I'm sure this exists already but it more for learning

## Layout

- `src/inferstream/` — installable package (`inferstream.inference`, `worker`, `gateway`, `messaging`).
- `proto/` — `.proto` sources; regenerate Python stubs with `./scripts/regenerate_proto.sh` (requires `uv sync --group dev`). Generated `*_pb2.py` files under `src/` are **committed** so `pip install inferstream` works without installing `protoc`; many libraries do this, while others regenerate only in CI—either is valid.
- `tests/` — pytest suite.

## Commands

```bash
uv sync --group dev    # install project + dev deps (pytest)
uv build               # wheel + sdist in dist/
uv run inferstream     # demo CLI (downloads model on first run)
uv run pytest          # tests
./scripts/regenerate_proto.sh   # protobuf → src/inferstream/grpc/generated/
```