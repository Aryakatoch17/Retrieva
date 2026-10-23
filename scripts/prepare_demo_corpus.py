#!/usr/bin/env python3
"""Build demo corpora under demo/corpus/ (smoke) or demo/corpus_large/ (scaled).

Offline smoke (default): 8 fixed arXiv-style abstracts — no network.

Scaled (--from-beir / --limit): export BEIR SciFact (or another BEIR) corpus as .txt
so hybrid/rerank demos have a few thousand docs. Requires a prior
`python scripts/download_beir.py scifact`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_SMOKE = ROOT / "demo" / "corpus"
OUT_LARGE = ROOT / "demo" / "corpus_large"

ABSTRACTS = [
    (
        "1409.0473",
        "Neural Machine Translation by Jointly Learning to Align and Translate",
        "Bahdanau et al. introduce attention for neural machine translation, "
        "allowing soft alignments between source and target tokens.",
    ),
    (
        "1706.03762",
        "Attention Is All You Need",
        "Vaswani et al. propose the Transformer architecture based solely on "
        "self-attention, enabling parallel training and strong translation quality.",
    ),
    (
        "1810.04805",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Devlin et al. pretrain bidirectional Transformers with masked language "
        "modeling for transfer learning on NLP tasks.",
    ),
    (
        "2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "Lewis et al. combine a parametric generator with a non-parametric dense "
        "retriever (RAG) for open-domain question answering.",
    ),
    (
        "2004.04906",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "Karpukhin et al. show dual-encoder dense retrieval outperforms BM25 on "
        "multiple open-domain QA datasets.",
    ),
    (
        "2104.08663",
        "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models",
        "Thakur et al. release BEIR, a suite of diverse retrieval datasets for "
        "evaluating sparse and dense models out of domain.",
    ),
    (
        "2210.07316",
        "RankVicuna and Generative Reranking",
        "Listwise generative rerankers use large language models to reorder "
        "retrieval candidates for improved nDCG.",
    ),
    (
        "2307.09288",
        "Hybrid Retrieval and Fusion",
        "Hybrid pipelines fuse BM25 lexical ranks with dense ANN neighbors using "
        "reciprocal rank fusion before late interaction or cross-encoder reranking.",
    ),
]


def write_smoke(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for arxiv_id, title, abstract in ABSTRACTS:
        path = out / f"{arxiv_id}.txt"
        path.write_text(f"{title}\n\n{abstract}\n", encoding="utf-8")
        print(f"wrote {path}")
    return len(ABSTRACTS)


def write_from_beir(beir_root: Path, out: Path, limit: int) -> int:
    corpus = beir_root / "corpus.jsonl"
    if not corpus.exists():
        # Nested layout: bench/datasets/scifact/scifact/corpus.jsonl
        nested = list(beir_root.rglob("corpus.jsonl"))
        if not nested:
            raise SystemExit(
                f"no corpus.jsonl under {beir_root}; run: python scripts/download_beir.py scifact"
            )
        corpus = nested[0]
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            if limit > 0 and n >= limit:
                break
            row = json.loads(line)
            doc_id = str(row.get("_id") or row.get("id") or n)
            title = (row.get("title") or "").strip()
            text = (row.get("text") or row.get("contents") or "").strip()
            body = f"{title}\n\n{text}\n" if title else f"{text}\n"
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
            (out / f"{safe}.txt").write_text(body, encoding="utf-8")
            n += 1
    print(f"wrote {n} docs from {corpus} -> {out}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-beir",
        type=Path,
        default=None,
        help="BEIR dataset dir (or parent). Default large source: bench/datasets/scifact",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max docs when --from-beir (0 = all). Example: --limit 3000",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: demo/corpus or demo/corpus_large)",
    )
    args = parser.parse_args()

    if args.from_beir is not None or args.limit > 0:
        beir = args.from_beir or (ROOT / "bench" / "datasets" / "scifact")
        out = args.out or OUT_LARGE
        n = write_from_beir(beir, out, args.limit if args.limit > 0 else 10**9)
        print(f"\nIngest large demo:\n  cargo run -p retrieva-cli -- ingest --collection arxiv {out}")
        print(f"docs={n}")
    else:
        out = args.out or OUT_SMOKE
        n = write_smoke(out)
        print(f"\nSmoke ingest:\n  cargo run -p retrieva-cli -- ingest --collection arxiv {out}")
        print(f"docs={n} (use --from-beir for thousands)")


if __name__ == "__main__":
    main()
