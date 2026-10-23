#!/usr/bin/env python3
"""Download answerai-colbert-small-v1 Vespa ONNX (true late-interaction, dim=96).

Uses `vespa_colbert.onnx` (projection + L2) — not the plain-BERT onnx/ export.

Usage:
  python scripts/download_colbert.py
  python scripts/download_colbert.py --out models/colbert

Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "answerdotai/answerai-colbert-small-v1"
# Pin after vespa_colbert.onnx upload; bump if re-exporting.
DEFAULT_REVISION = "main"
# (remote_name, local_name)
FILES = (
    ("vespa_colbert.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("config.json", "config.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/colbert"),
        help="Output directory (model.onnx + tokenizer.json)",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for remote, local in FILES:
        path = hf_hub_download(repo_id=REPO, filename=remote, revision=args.revision)
        dest = args.out / local
        dest.write_bytes(Path(path).read_bytes())
        print(f"wrote {dest} ({dest.stat().st_size} bytes)")

    pin = args.out / "REVISION.txt"
    pin.write_text(f"{REPO}@{args.revision} (vespa_colbert.onnx)\n", encoding="utf-8")
    sums = args.out / "SHA256SUMS"
    lines = []
    for p in sorted(args.out.iterdir()):
        if p.is_file() and p.name not in {"REVISION.txt", "SHA256SUMS"}:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{digest}  {p.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {sums}")
    print(f"\nPoint rerank.maxsim_path at {args.out / 'model.onnx'}")


if __name__ == "__main__":
    main()
