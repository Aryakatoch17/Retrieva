# CE vs ColBERT MaxSim (BEIR)

dataset=scifact pool=50 top_k=10 candidates=hybrid onnx/bge-small (prior CPU run; docs re-encoded per query)

| Candidates | Backend | nDCG@10 | mean latency (ms) | n |
|------------|---------|--------:|------------------:|--:|
| hybrid | identity | 0.6673 | — | 300 |
| dense | identity | 0.6845 | — | 300 |
| hybrid | maxsim_onnx | 0.7020 | 3260.17 | 300 |
| hybrid | cross_encoder | 0.7016 | 14963.63 | 300 |

- Latency is **per query** over 50 candidates on CPU — not interactive. Re-run with doc-token precompute + pool=20 + optional `--features cuda`.
- hybrid+ColBERT is +0.035 over hybrid identity, but only +0.0175 over dense alone.
- Hash MaxSim is excluded (smoke-only / near-random).
- maxsim_onnx = answerai-colbert-small-v1 Vespa ONNX; cross_encoder = bge-reranker-base.
