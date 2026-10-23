# Demo corpus

## Smoke (8 abstracts)

Offline fixed set under `demo/corpus/`:

```bash
python scripts/prepare_demo_corpus.py
cargo run -p retrieva-cli --release -- ingest --collection arxiv ./demo/corpus --embedder hash
```

| Count | Detail |
|------:|--------|
| 8 | `.txt` files named by arXiv id |
| 1 | chunk per file at default ingest (`chunk_size=512`, overlap `64`) |

Enough for a wiring smoke — **too small** to show hybrid or reranking helping.

## Scaled (~3k abstracts)

Export from BEIR SciFact (or another BEIR corpus already downloaded):

```bash
python scripts/download_beir.py scifact   # once
python scripts/prepare_demo_corpus.py --from-beir bench/datasets/scifact --limit 3000
# → demo/corpus_large/ (3000 .txt files)

cargo run -p retrieva-cli --release --features onnx -- \
  ingest --collection arxiv ./demo/corpus_large --embedder onnx
```

Prefer ONNX BGE for the large demo so dense/hybrid stages are meaningful. Hash embedder still works for BM25-only smoke.

Recording: [scripts/record_demo.md](../scripts/record_demo.md).
