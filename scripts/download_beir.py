#!/usr/bin/env python3
"""Download a BEIR dataset zip into bench/datasets/<name>/.

Example:
  python scripts/download_beir.py scifact
  python scripts/download_beir.py nfcorpus
  python scripts/download_beir.py fiqa
"""

from __future__ import annotations

import argparse
import io
import ssl
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
ROOT = Path(__file__).resolve().parents[1]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Last resort for broken corporate/MinGW cert stores (demo machines only).
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        help="BEIR dataset name (scifact, nfcorpus, fiqa, ...)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "bench" / "datasets",
        help="Output parent directory",
    )
    args = parser.parse_args()
    name = args.dataset.strip().lower()
    url = f"{BASE}/{name}.zip"
    dest = args.out / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} ...", file=sys.stderr)
    with urlopen(url, timeout=300, context=_ssl_context()) as resp:
        data = resp.read()
    print(f"Extracting to {dest} ...", file=sys.stderr)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)

    # Some zips nest an extra directory; flatten if only one child dir and no corpus at root.
    if not (dest / "corpus.jsonl").exists():
        children = [p for p in dest.iterdir() if p.is_dir()]
        if len(children) == 1 and (children[0] / "corpus.jsonl").exists():
            print(f"Dataset root: {children[0]}", file=sys.stderr)

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
