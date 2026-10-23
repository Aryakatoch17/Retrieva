# Load: load_scifact_onnx_cache_off

requests=100 concurrency=4 cache=false hit_rate=0.0% (0/0) mix=zipf

ok=100 err=0 p50=140.45ms p95=281.67ms p99=398.04ms mean=152.88ms qps=25.74

notes: BEIR scifact; embedder=onnx/bge-small; docs=5183; chunks=5228; queries_pool=1109; mode=Hybrid; rerank=identity; mix=zipf; linux/x86_64 logical_cpus=16
