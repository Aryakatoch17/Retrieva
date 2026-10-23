#!/usr/bin/env bash
# CE vs ColBERT-ONNX MaxSim on SciFact: nDCG@10 + latency, shared top-50.
set -eu
cd /tmp
rm -rf /tmp/retrieva-src
mkdir -p /tmp/retrieva-src
cp -a /src/Cargo.toml /src/Cargo.lock /src/crates /src/proto /src/config /tmp/retrieva-src/
ln -sfn /src/models /tmp/retrieva-src/models
ln -sfn /src/bench /tmp/retrieva-src/bench
cd /tmp/retrieva-src

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target-u24}"
export RETRIEVA_DATA_DIR="${RETRIEVA_DATA_DIR:-/src/data_onnx_ablate}"
mkdir -p "$RETRIEVA_DATA_DIR"
export RUST_LOG=warn

test -f /src/models/colbert/model.onnx || {
  echo "missing models/colbert — run: python scripts/download_colbert.py"
  exit 1
}
test -f /src/models/bge-reranker-base/model.onnx || {
  echo "missing bge-reranker-base"
  exit 1
}

cargo build --release -p retrieva-cli --features onnx
BIN="$CARGO_TARGET_DIR/release/retrieva"
DS="/src/bench/datasets/scifact/scifact"

"$BIN" --config /src/config/retrieva.example.toml bench --suite rerank-ablate \
  --dataset "$DS" --embedder onnx --top-k 10 --candidate-pool 50 \
  --out-dir /src/bench/results

echo "== wrote ablate_ce_vs_maxsim_beir.{json,md} =="
