#!/usr/bin/env bash
# Zipf ONNX load + rerank ablate (pool 20 then 50). Mount repo at /work.
set -eu
cd /work
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target-u24}"
export RUST_LOG="${RUST_LOG:-warn}"

DS=/work/bench/datasets/scifact/scifact
test -d "$DS"
test -f /work/models/bge-small-en-v1.5/model.onnx
test -f /work/models/colbert/model.onnx
test -f /work/models/bge-reranker-base/model.onnx

cargo build -p retrieva-cli --release --features onnx
BIN="$CARGO_TARGET_DIR/release/retrieva"

echo "ONNX Zipf load starting"
export RETRIEVA_DATA_DIR=/tmp/onnx_load_zipf
rm -rf "$RETRIEVA_DATA_DIR"
mkdir -p "$RETRIEVA_DATA_DIR" /work/bench/results
"$BIN" --config /work/config/retrieva.example.toml bench --suite load \
  --dataset "$DS" --embedder onnx --mode hybrid \
  --requests 100 --concurrency 4 \
  --out-dir /work/bench/results

echo "ablate pool=20 starting"
export RETRIEVA_DATA_DIR=/tmp/onnx_ablate20
rm -rf "$RETRIEVA_DATA_DIR"
mkdir -p "$RETRIEVA_DATA_DIR"
"$BIN" --config /work/config/retrieva.example.toml bench --suite rerank-ablate \
  --dataset "$DS" --embedder onnx --top-k 10 --candidate-pool 20 \
  --out-dir /work/bench/results
cp -f /work/bench/results/ablate_ce_vs_maxsim_beir.json \
  /work/bench/results/ablate_ce_vs_maxsim_beir_pool20.json
cp -f /work/bench/results/ablate_ce_vs_maxsim_beir.md \
  /work/bench/results/ablate_ce_vs_maxsim_beir_pool20.md

echo "ablate pool=50 starting"
export RETRIEVA_DATA_DIR=/tmp/onnx_ablate50
rm -rf "$RETRIEVA_DATA_DIR"
mkdir -p "$RETRIEVA_DATA_DIR"
"$BIN" --config /work/config/retrieva.example.toml bench --suite rerank-ablate \
  --dataset "$DS" --embedder onnx --top-k 10 --candidate-pool 50 \
  --out-dir /work/bench/results
cp -f /work/bench/results/ablate_ce_vs_maxsim_beir.json \
  /work/bench/results/ablate_ce_vs_maxsim_beir_pool50.json

echo DONE
ls -la /work/bench/results/load_*onnx* /work/bench/results/ablate_ce_vs_maxsim_beir* || true
