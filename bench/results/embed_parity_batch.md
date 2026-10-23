# ONNX embed parity (batched, multi-length)

Measured with `retrieva embed-parity` vs Python `scripts/embed_parity.py` on three texts of different lengths (padding + batching), BGE-small-en-v1.5 ONNX, Docker `retrieva-onnx:builder` (Ubuntu 24.04).

| # | chars | cosine vs Python |
|--:|------:|-----------------:|
| 0 | 11 | 1.000000 |
| 1 | 64 | 1.000000 |
| 2 | 217 | 1.000000 |

**cosine_vs_python_min = 1.000000** (n=3)

Texts: short (`hello world`), mid-length with padding pressure, and a long retrieval-domain sentence. Not a single-string smoke.
