#!/usr/bin/env python3
"""Emit reference embeddings for Rust ONNX parity checks (Phase 3).

Usage (after download_bge_small.py):
  python scripts/embed_parity.py --model models/bge-small-en-v1.5 --text "hello world"

Requires: pip install onnxruntime numpy tokenizers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

QUERY_PREFIX = "Represent this sentence for searching: "


def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[:, :, None]
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, a_min=1e-8, a_max=None)


def embed(model_dir: Path, texts: list[str], query: bool) -> list[list[float]]:
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    prepared = [f"{QUERY_PREFIX}{t}" if query else t for t in texts]
    encodings = tok.encode_batch(prepared)
    max_len = max(len(e.ids) for e in encodings)
    batch = len(encodings)

    input_ids = np.zeros((batch, max_len), dtype=np.int64)
    attention_mask = np.zeros((batch, max_len), dtype=np.int64)
    token_type_ids = np.zeros((batch, max_len), dtype=np.int64)
    for i, enc in enumerate(encodings):
        n = len(enc.ids)
        input_ids[i, :n] = enc.ids
        attention_mask[i, :n] = enc.attention_mask
        token_type_ids[i, :n] = enc.type_ids

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    # First output is typically last_hidden_state
    hidden = outputs[0]
    pooled = l2_normalize(mean_pool(hidden, attention_mask))
    return pooled.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/bge-small-en-v1.5"))
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--query", action="store_true", help="Apply BGE query prefix")
    parser.add_argument("--out", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()

    vectors = embed(args.model, args.text, query=args.query)
    payload = json.dumps(
        {"texts": args.text, "query": args.query, "embeddings": vectors}, indent=2
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
