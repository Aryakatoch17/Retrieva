#!/usr/bin/env bash
# Finish interrupted real-model work: SciFact ONNX matrix+load, then CE vs ColBERT ablate.
set -eu
cd /tmp
rm -rf /tmp/retrieva-src
mkdir -p /tmp/retrieva-src
cp -a /src/Cargo.toml /src/Cargo.lock /src/crates /src/proto /src/config /tmp/retrieva-src/
ln -sfn /src/models /tmp/retrieva-src/models
ln -sfn /src/bench /tmp/retrieva-src/bench
cd /tmp/retrieva-src

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target-u24}"
export RUST_LOG=warn

test -f /src/models/bge-small-en-v1.5/model.onnx
test -f /src/models/bge-reranker-base/model.onnx
test -f /src/models/colbert/model.onnx
DS="/src/bench/datasets/scifact/scifact"
test -d "$DS"

cargo build --release -p retrieva-cli --features onnx
BIN="$CARGO_TARGET_DIR/release/retrieva"

echo "== 1/2 matrix + load (onnx scifact) =="
export RETRIEVA_DATA_DIR=/src/data_onnx_bench
mkdir -p "$RETRIEVA_DATA_DIR"

"$BIN" --config /src/config/retrieva.example.toml bench --suite matrix \
  --dataset "$DS" --embedder onnx --top-k 10 \
  --out-dir /src/bench/results

cp -f /src/bench/results/matrix.json /src/bench/results/matrix_onnx_scifact.json
cp -f /src/bench/results/matrix.md /src/bench/results/matrix_onnx_scifact.md
echo "== wrote matrix_onnx_scifact.* =="

"$BIN" --config /src/config/retrieva.example.toml bench --suite load \
  --dataset "$DS" --embedder onnx --mode hybrid \
  --requests 100 --concurrency 4 \
  --out-dir /src/bench/results
echo "== load onnx done (Zipf query mix; check cache_hit_rate before citing cache-on) =="

echo "== 2/3 CE vs ColBERT MaxSim pool=50 (rerank-ablate; doc-token precompute) =="
export RETRIEVA_DATA_DIR=/src/data_onnx_ablate
mkdir -p "$RETRIEVA_DATA_DIR"

"$BIN" --config /src/config/retrieva.example.toml bench --suite rerank-ablate \
  --dataset "$DS" --embedder onnx --top-k 10 --candidate-pool 50 \
  --out-dir /src/bench/results
cp -f /src/bench/results/ablate_ce_vs_maxsim_beir.json \
  /src/bench/results/ablate_ce_vs_maxsim_beir_pool50.json || true

echo "== 3/3 CE vs ColBERT MaxSim pool=20 =="
export RETRIEVA_DATA_DIR=/src/data_onnx_ablate20
mkdir -p "$RETRIEVA_DATA_DIR"
"$BIN" --config /src/config/retrieva.example.toml bench --suite rerank-ablate \
  --dataset "$DS" --embedder onnx --top-k 10 --candidate-pool 20 \
  --out-dir /src/bench/results
cp -f /src/bench/results/ablate_ce_vs_maxsim_beir.json \
  /src/bench/results/ablate_ce_vs_maxsim_beir_pool20.json || true
cp -f /src/bench/results/ablate_ce_vs_maxsim_beir.md \
  /src/bench/results/ablate_ce_vs_maxsim_beir_pool20.md || true

echo "== ALL DONE =="
ls -la /src/bench/results/matrix_onnx_scifact.* /src/bench/results/load_*onnx* /src/bench/results/ablate_ce_vs_maxsim_beir* 2>/dev/null || true
