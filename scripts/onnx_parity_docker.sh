#!/usr/bin/env bash
# Build --features onnx on Linux (works where windows-gnu lacks ort prebuilts)
# and compare Rust embeddings to Python reference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEXT="${1:-hello world}"
docker run --rm \
  -v "$ROOT:/src" -w /src \
  rust:1.90-bookworm \
  bash -lc '
    set -euo pipefail
    apt-get update -qq
    apt-get install -y -qq build-essential cmake pkg-config libssl-dev clang python3 python3-pip python3-venv >/dev/null
    python3 -m venv /tmp/venv
    /tmp/venv/bin/pip install -q onnxruntime numpy tokenizers huggingface_hub
    if [[ ! -f models/bge-small-en-v1.5/model.onnx ]]; then
      /tmp/venv/bin/python scripts/download_bge_small.py
    fi
    /tmp/venv/bin/python scripts/embed_parity.py \
      --model models/bge-small-en-v1.5 \
      --text "'"$TEXT"'" --query \
      --out /tmp/py_embed.json
    cargo build --release -p retrieva-cli --features onnx
    ./target/release/retrieva embed-parity \
      --text "'"$TEXT"'" --query \
      --python-json /tmp/py_embed.json
  '
