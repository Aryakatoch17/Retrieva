#!/usr/bin/env bash
# BEIR SciFact matrix + load with ONNX BGE (real hybrid story).
# Run inside retrieva-onnx:builder.
set -eu
cd /tmp
rm -rf /tmp/retrieva-src
mkdir -p /tmp/retrieva-src
cp -a /src/Cargo.toml /src/Cargo.lock /src/crates /src/proto /src/config /tmp/retrieva-src/
ln -sfn /src/models /tmp/retrieva-src/models
ln -sfn /src/bench /tmp/retrieva-src/bench
cd /tmp/retrieva-src

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target-u24}"
export RETRIEVA_DATA_DIR="${RETRIEVA_DATA_DIR:-/src/data_onnx_bench}"
mkdir -p "$RETRIEVA_DATA_DIR"

cargo build --release -p retrieva-cli --features onnx
BIN="$CARGO_TARGET_DIR/release/retrieva"

DS="/src/bench/datasets/scifact/scifact"
export RUST_LOG=warn

echo "== matrix onnx scifact =="
"$BIN" --config /src/config/retrieva.example.toml bench --suite matrix \
  --dataset "$DS" --embedder onnx --top-k 10 \
  --out-dir /src/bench/results

# Rename matrix outputs so hash results are not overwritten silently.
cp -f /src/bench/results/matrix.json /src/bench/results/matrix_onnx_scifact.json
cp -f /src/bench/results/matrix.md /src/bench/results/matrix_onnx_scifact.md

echo "== load onnx hybrid (no CE) =="
"$BIN" --config /src/config/retrieva.example.toml bench --suite load \
  --dataset "$DS" --embedder onnx --mode hybrid \
  --requests 100 --concurrency 4 \
  --out-dir /src/bench/results

echo "== done =="
