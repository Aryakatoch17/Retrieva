# BEIR mode matrix (onnx)

hybrid_ok=false (measurable gain over both singles)

| Dataset | Mode | nDCG@10 | Recall@10 | MRR |
|---------|------|--------|---------|-----|
| scifact | Bm25 | 0.6612 | 0.7876 | 0.6276 |
| scifact | Dense | 0.6845 | 0.7980 | 0.6550 |
| scifact | Hybrid | 0.6673 | 0.7983 | 0.6321 |
| scifact | HybridRerank | 0.3871 | 0.5236 | 0.3483 |


## Hybrid assessment

- scifact: MIXED hybrid=0.6673 bm25=0.6612 dense=0.6845 (no ≥0.01 gain over best single; normal when BM25 is strong)
