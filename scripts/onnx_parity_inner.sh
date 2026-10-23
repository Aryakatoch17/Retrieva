#!/usr/bin/env bash
# Linux-only ONNX build + embedding parity (mean-pool / BGE prefix).
set -eu
cd /src

export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true

echo "== apt =="
apt-get update -qq
apt-get install -y -qq build-essential cmake pkg-config libssl-dev clang python3 python3-pip python3-venv >/dev/null

echo "== python venv + deps =="
python3 -m venv /tmp/venv
/tmp/venv/bin/pip install --upgrade pip -q
/tmp/venv/bin/pip install --timeout 120 onnxruntime numpy tokenizers

TEXT="${PARITY_TEXT:-hello world}"
echo "== python embed parity JSON =="
/tmp/venv/bin/python scripts/embed_parity.py --model models/bge-small-en-v1.5 --text "$TEXT" --query --out /tmp/py_embed.json

export CARGO_HOME="${CARGO_HOME:-/cache/cargo}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/cache/target}"
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"

# Use the image rustc explicitly so ./rust-toolchain.toml does not trigger a rustup download.
IMG_TC="$(rustc -V | awk '{print $2}')"
echo "== cargo build --features onnx (toolchain $IMG_TC) =="
cargo "+$IMG_TC" build --release -p retrieva-cli --features onnx

echo "== rust embed-parity =="
"$CARGO_TARGET_DIR/release/retrieva" embed-parity --text "$TEXT" --query --python-json /tmp/py_embed.json

echo "== done =="
