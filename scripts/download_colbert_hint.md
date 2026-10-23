# ColBERT / MaxSim ONNX weights

Retrieva Phase 5 uses **MaxSim** late-interaction scoring.

## Without ONNX (default)

`HashMaxSimReranker` runs out of the box (`rerank.backend = "maxsim"`). **Not** a quality baseline — do not compare its latency to a cross-encoder.

## Real ColBERT ONNX (`--features onnx`)

Download the Vespa export of `answerai-colbert-small-v1` (96-d projected + L2):

```bash
python scripts/download_colbert.py
# → models/colbert/model.onnx + tokenizer.json
```

```toml
[rerank]
backend = "maxsim"
maxsim_path = "models/colbert/model.onnx"
```

Headline CE vs MaxSim (nDCG@10 + latency on shared top-50):

```bash
cargo run -p retrieva-cli --release --features onnx -- bench --suite rerank-ablate \
  --dataset bench/datasets/scifact/scifact \
  --embedder onnx --top-k 10 --candidate-pool 50
```
