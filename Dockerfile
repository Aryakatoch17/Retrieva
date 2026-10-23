# Multi-stage build for Retrieva (Linux). Default features (hash embedder, no onnx).
# Build: docker build -t retrieva:local .
# Run:   docker compose up --build

# 1.90+: uuid/ordered-float (via tantivy) raised MSRV past 1.88.
FROM rust:1.90-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY proto ./proto
COPY config ./config

# Default: hash embedder. Pass --build-arg FEATURES=onnx for ONNX Runtime.
ARG FEATURES=
RUN cargo build --release -p retrieva-cli ${FEATURES:+--features $FEATURES} \
    && strip target/release/retrieva

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 retrieva

WORKDIR /app
COPY --from=builder /src/target/release/retrieva /usr/local/bin/retrieva
COPY config/retrieva.docker.toml /app/config/retrieva.toml
COPY demo /app/demo

RUN mkdir -p /app/data && chown -R retrieva:retrieva /app

USER retrieva
ENV RETRIEVA_CONFIG=/app/config/retrieva.toml
EXPOSE 8080 50051

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["retrieva"]
CMD ["serve"]
