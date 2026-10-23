# Pinned models

Scripts download into `models/` (gitignored) and write `REVISION.txt` + `SHA256SUMS`.

| Role | Hub id | Default revision | Local path | Script |
|------|--------|------------------|------------|--------|
| Embedder | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | `models/bge-small-en-v1.5/` | `scripts/download_bge_small.py` |
| Cross-encoder | `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` | `models/bge-reranker-base/` | `scripts/download_bge_reranker.py` |
| ColBERT MaxSim | `answerdotai/answerai-colbert-small-v1` (`vespa_colbert.onnx`) | `main` | `models/colbert/` | `scripts/download_colbert.py` |

```bash
python scripts/download_bge_small.py
python scripts/download_bge_reranker.py
# Parity (mean-pool + L2; query uses BGE prefix)
# Prefer a batch with short + long strings (padding / mean-pool coverage), not one toy phrase.
python scripts/embed_parity.py --model models/bge-small-en-v1.5 --query \
  --text "hello world" \
  --text "Representations of short strings need padding in the ONNX batch." \
  --text "Dense passage retrieval for open-domain question answering combines dual encoders with BM25 hybrid fusion and late interaction reranking over candidate pools drawn from large corpora such as BEIR SciFact and NFCorpus." \
  --out bench/results/py_embed_batch.json
# Then compare with the onnx-enabled CLI (Docker helper: scripts/onnx_parity_fast.ps1).
```

Expected: **min** cosine ≥ **0.999** across the batch. Lower values usually mean pooling / prefix / padding mismatches.

## Windows note (`onnx`)

`ort` prebuilts need **glibc ≥ 2.38** (Ubuntu 24.04+) and on Windows **`x86_64-pc-windows-msvc`** + VS Build Tools (`link.exe`). Debian bookworm (2.36) fails linking with `__isoc23_strtol`.

```powershell
# Windows gnu / no MSVC: host Python + Docker (Dockerfile.onnx = Ubuntu 24.04)
powershell -ExecutionPolicy Bypass -File scripts/onnx_parity_fast.ps1
```

Linux on Ubuntu 24.04+/recent Fedora: `cargo run -p retrieva-cli --features onnx -- …`.
