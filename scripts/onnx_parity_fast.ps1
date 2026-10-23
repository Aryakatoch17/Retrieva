# Host Python parity JSON + Docker cargo --features onnx (uses Dockerfile.onnx cache).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== python parity JSON ==" -ForegroundColor Cyan
py -3 -m pip install -q onnxruntime numpy tokenizers
py -3 scripts/embed_parity.py --model models/bge-small-en-v1.5 --text "hello world" --query --out bench/results/py_embed.json

Write-Host "== docker build Dockerfile.onnx ==" -ForegroundColor Cyan
docker build -f Dockerfile.onnx -t retrieva-onnx:builder .

Write-Host "== cargo --features onnx + embed-parity ==" -ForegroundColor Cyan
docker volume create retrieva-cargo-cache 2>$null | Out-Null
docker run --rm `
  -v "${PWD}:/src" `
  -v retrieva-cargo-cache:/cache `
  -w /src `
  -e CARGO_TARGET_DIR=/cache/target-u24 `
  retrieva-onnx:builder `
  bash /src/scripts/onnx_cargo_parity.sh
