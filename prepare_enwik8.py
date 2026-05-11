#!/usr/bin/env python3
"""Download and extract enwik8 into the local data directory."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--url", default=ENWIK8_URL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "enwik8.zip"
    out_path = data_dir / "enwik8.txt"

    if not zip_path.exists() or args.force:
        print(f"downloading {args.url} -> {zip_path}")
        urllib.request.urlretrieve(args.url, zip_path)
    else:
        print(f"using existing {zip_path}")

    if not out_path.exists() or args.force:
        print(f"extracting enwik8 -> {out_path}")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("enwik8") as src, out_path.open("wb") as dst:
                dst.write(src.read())
    else:
        print(f"using existing {out_path}")

    print(f"ready: {out_path}")
    print("train with: --data-file data/enwik8.txt --encoding latin-1")


if __name__ == "__main__":
    main()
