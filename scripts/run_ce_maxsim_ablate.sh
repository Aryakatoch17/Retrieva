#!/usr/bin/env bash
# Cross-encoder (ONNX) vs hash MaxSim on a live candidate pool.
# Run inside retrieva-onnx:builder with models/ and a populated data_dir mounted.
set -eu
cd /tmp
rm -rf /tmp/retrieva-src
mkdir -p /tmp/retrieva-src
cp -a /src/Cargo.toml /src/Cargo.lock /src/crates /src/proto /src/config /tmp/retrieva-src/
ln -sfn /src/models /tmp/retrieva-src/models
ln -sfn /src/bench /tmp/retrieva-src/bench
ln -sfn /src/demo /tmp/retrieva-src/demo
cd /tmp/retrieva-src

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target-u24}"
BIN="$CARGO_TARGET_DIR/release/retrieva"
# Always rebuild so CE input-name / ablation-arm fixes are present.
cargo build --release -p retrieva-cli --features onnx

DATA="${RETRIEVA_DATA_DIR:-/src/data}"
export RETRIEVA_DATA_DIR="$DATA"
mkdir -p "$DATA"

# Absolute model paths — relative paths break when cwd != repo root / Docker mounts.
export RETRIEVA_RERANK__CROSS_ENCODER_PATH="/src/models/bge-reranker-base/model.onnx"
export RETRIEVA_EMBEDDING__MODEL_PATH="/src/models/bge-small-en-v1.5/model.onnx"

# Expect demo/corpus already prepared on the host (8 abstracts).
"$BIN" --config /src/config/retrieva.example.toml ingest --collection arxiv /src/demo/corpus --embedder hash

# Research ablation: cross-encoder vs hash MaxSim (identity only if CE missing).
# ort is chatty at info; keep JSON stdout clean.
export RUST_LOG=error
"$BIN" --config /src/config/retrieva.example.toml ablate \
  --collection arxiv \
  "dense passage retrieval for open domain question answering" \
  --top-k 5 \
  --candidate-pool 20 \
  > /src/bench/results/ablate_ce_vs_maxsim.json

echo "== ablate JSON =="
cat /src/bench/results/ablate_ce_vs_maxsim.json
echo "== wrote bench/results/ablate_ce_vs_maxsim.json =="
