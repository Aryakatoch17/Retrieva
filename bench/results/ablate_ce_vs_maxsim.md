# Ablation: cross-encoder vs MaxSim

**Not** identity vs MaxSim. Arms: `maxsim_hash` (HashMaxSimReranker) vs `cross_encoder` (`bge-reranker-base` ONNX).

| | MaxSim (hash) | Cross-encoder |
|--|---------------|---------------|
| top-1 | `2004.04906` (DPR) | `2004.04906` (DPR) |
| top-2 | `2005.11401` (RAG) | `2005.11401` (RAG) |
| top-3 | `2307.09288` | `2104.08663` (BEIR) |
| latency | **0.36 ms** | **362 ms** |
| top score | 8.00 | 4.99 |

Same top-2; ranks 3–5 diverge — CE is not a no-op relative to MaxSim. MaxSim here is **hash token MaxSim**, not ColBERT ONNX (no `models/colbert` in this run).

- Corpus: 8 arXiv-style abstracts (`demo/corpus`), hybrid BM25+hash candidate pool of 8
- Hardware: Docker `retrieva-onnx:builder`, CPU
- Reproduce: `bash scripts/run_ce_maxsim_ablate.sh` inside that image

JSON: [`ablate_ce_vs_maxsim.json`](ablate_ce_vs_maxsim.json).
