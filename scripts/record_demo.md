# Recording a short demo video

Automated recording is environment-specific; use this checklist:

1. `python scripts/prepare_demo_corpus.py`
2. `cargo run -p retrieva-cli -- ingest --collection arxiv ./demo/corpus`
3. Start capture (OBS / Windows Game Bar / `asciinema` for terminal-only).
4. Show:
   - `retrieva query --collection arxiv "dense passage retrieval" --mode hybrid` (query is positional)
   - `curl localhost:8080/health` and `/metrics` while `retrieva serve` runs
   - Optional: `docker compose up` from [docs/DEMO.md](../docs/DEMO.md)
5. Export 60–90s clip to `docs/assets/demo.mp4` (git-lfs recommended) and link it from the README.
