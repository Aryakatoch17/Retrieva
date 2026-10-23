#!/usr/bin/env python3
"""Download BAAI/bge-small-en-v1.5 ONNX + tokenizer for Retrieva Phase 3.

Usage:
  python scripts/download_bge_small.py
  python scripts/download_bge_small.py --out models/bge-small-en-v1.5

Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "BAAI/bge-small-en-v1.5"
# Pin to a commit SHA so downloads are reproducible (SHA256SUMS written after download).
DEFAULT_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
FILES = (
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/bge-small-en-v1.5"),
        help="Output directory (model.onnx + tokenizer.json)",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face git revision (commit SHA)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        path = hf_hub_download(repo_id=REPO, filename=name, revision=args.revision)
        target_name = "model.onnx" if name.endswith("model.onnx") else Path(name).name
        dest = args.out / target_name
        dest.write_bytes(Path(path).read_bytes())
        print(f"wrote {dest}")

    pin = args.out / "REVISION.txt"
    pin.write_text(f"{REPO}@{args.revision}\n", encoding="utf-8")
    print(f"wrote {pin}")
    sums = args.out / "SHA256SUMS"
    lines = []
    for p in sorted(args.out.iterdir()):
        if p.is_file() and p.name not in {"REVISION.txt", "SHA256SUMS"}:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{digest}  {p.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {sums}")
    print(f"\nDone. Point embedding.model_path at {args.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
