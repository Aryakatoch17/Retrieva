#!/usr/bin/env bash
# Build --features onnx + multi-text embedding parity (padding / batch).
set -eu

echo "== prepare build tree (no rust-toolchain.toml) =="
rm -rf /tmp/retrieva-src
mkdir -p /tmp/retrieva-src
cp -a /src/Cargo.toml /src/Cargo.lock /src/crates /src/proto /src/config /tmp/retrieva-src/
ln -sfn /src/models /tmp/retrieva-src/models
ln -sfn /src/bench /tmp/retrieva-src/bench
ln -sfn /src/scripts /tmp/retrieva-src/scripts
cd /tmp/retrieva-src

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target-u24}"
mkdir -p "$CARGO_TARGET_DIR"

echo "rustc=$(rustc -V)"
echo "== cargo build --features onnx =="
cargo build --release -p retrieva-cli --features onnx

# Host-generated JSON preferred; optional in-container python skipped (image has no ORT py).
PY_JSON="/src/bench/results/py_embed_batch.json"
if [[ ! -f "$PY_JSON" ]]; then
  echo "missing $PY_JSON — generate on host with scripts/embed_parity.py (multiple --text)"
  exit 1
fi

echo "== rust embed-parity (batch) =="
"$CARGO_TARGET_DIR/release/retrieva" embed-parity \
  --text "hello world" \
  --text "Representations of short strings need padding in the ONNX batch." \
  --text "Dense passage retrieval for open-domain question answering combines dual encoders with BM25 hybrid fusion and late interaction reranking over candidate pools drawn from large corpora such as BEIR SciFact and NFCorpus." \
  --query \
  --python-json "$PY_JSON"

echo "== done =="
