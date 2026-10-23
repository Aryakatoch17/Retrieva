# Benchmarks (Phase 9)

## Suites

| Suite | Command | Notes |
|-------|---------|-------|
| smoke | `cargo run -p retrieva-cli -- bench --suite smoke` | Tiny fixture under `fixtures/smoke/`; CI floor nDCG@10 ≥ 0.3 |
| beir | `… bench --suite beir --dataset bench/datasets/scifact` | Full BEIR split after download |
| load | `… bench --suite load --requests 100 --concurrency 8` | In-process latency / QPS |
| compare | `… bench --suite compare --baseline a.json --current b.json` | Fail if nDCG drops > 0.02 |

Reports land in `bench/results/` as JSON + Markdown.

## Download BEIR

```bash
python scripts/download_beir.py scifact
# also: nfcorpus, fiqa, …
cargo run -p retrieva-cli -- bench --suite beir --dataset bench/datasets/scifact --mode bm25 --top-k 10
```

## Make target

```bash
make bench-smoke   # runs CLI smoke suite
```
