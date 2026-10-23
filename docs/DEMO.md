# Docker demo

For the offline **arXiv-style abstract corpus** (8 files, how it was indexed): [DEMO_CORPUS.md](DEMO_CORPUS.md).

## Prerequisites

- Docker Engine 24+ with Compose v2
- ~2 GB free for the Rust builder stage (cached after first build)

## Start

```bash
docker compose up --build -d
curl -s http://localhost:8080/health
```

## Ingest sample docs

```bash
docker compose exec retrieva retrieva ingest --collection demo /app/demo/docs --embedder hash
```

## Search (REST)

```bash
curl -s http://localhost:8080/v1/search \
  -H 'content-type: application/json' \
  -d '{"collection_id":"demo","query":"hybrid BM25 dense retrieval","top_k":5,"mode":"hybrid","use_cache":true}'
```

## Metrics

```bash
curl -s http://localhost:8080/metrics | head
# Optional Prometheus UI:
docker compose --profile metrics up -d
# open http://localhost:9090
```

## Auth example

```bash
docker compose down
RETRIEVA_SERVER__API_KEY=dev-secret docker compose up -d
curl -s http://localhost:8080/v1/search \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-secret' \
  -d '{"collection_id":"demo","query":"rust","top_k":3,"mode":"bm25"}'
```

## Stop

```bash
docker compose down
# keep indexed data: omit -v
docker compose down -v   # wipe volume
```
