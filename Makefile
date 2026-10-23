.PHONY: fmt clippy test check ci bench-smoke docker-build docker-up docker-down clean

CARGO ?= cargo
MSRV ?= 1.90.0
COMPOSE ?= docker compose

fmt:
	$(CARGO) fmt --all

clippy:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

test:
	$(CARGO) test --workspace --all-targets

check:
	$(CARGO) check --workspace --all-targets

ci: fmt clippy test

# Phase 9: tiny BEIR fixture eval (writes bench/results/smoke.{json,md}).
bench-smoke:
	$(CARGO) run -p retrieva-cli -- bench --suite smoke --mode bm25 --top-k 10

# Phase 10: container demo
docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up --build -d

docker-down:
	$(COMPOSE) down

clean:
	$(CARGO) clean
