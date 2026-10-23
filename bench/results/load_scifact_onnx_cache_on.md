# Load: load_scifact_onnx_cache_on

requests=100 concurrency=4 cache=true hit_rate=42.0% (42/100) mix=zipf

ok=100 err=0 p50=107.18ms p95=159.49ms p99=178.87ms mean=74.60ms qps=52.31

notes: BEIR scifact; embedder=onnx/bge-small; docs=5183; chunks=5228; queries_pool=1109; mode=Hybrid; rerank=identity; mix=zipf; linux/x86_64 logical_cpus=16
