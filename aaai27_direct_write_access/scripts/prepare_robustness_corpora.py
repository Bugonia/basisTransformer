#!/usr/bin/env python3
"""Prepare public robustness corpora as plain text files.

The current training code consumes a single text file and performs
character-level language modeling. This script converts public Hugging Face
datasets into that format so the architecture ablations can be repeated beyond
enwik8 without changing the training loop.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for output text files. Defaults to $GLOBAL/data or data.",
    )
    parser.add_argument(
        "--corpus",
        choices=("all", "wikitext103", "fineweb_edu"),
        default="all",
        help="Which corpus to prepare.",
    )
    parser.add_argument(
        "--fineweb-chars",
        type=int,
        default=100_000_000,
        help="Approximate number of raw characters to write for FineWeb-Edu.",
    )
    parser.add_argument(
        "--fineweb-config",
        default="sample-10BT",
        help="FineWeb-Edu config to stream from.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def default_data_dir() -> Path:
    import os

    global_dir = os.environ.get("GLOBAL")
    if global_dir:
        return Path(global_dir) / "data"
    return Path("data")


def require_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install "
            "aaai27_direct_write_access/requirements.txt first."
        ) from exc


def clean_line(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    metadata = dict(metadata)
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def should_skip(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"using existing {path}")
        return True
    return False


def prepare_wikitext103(data_dir: Path, force: bool) -> Path:
    require_datasets()
    from datasets import load_dataset

    out_path = data_dir / "wikitext103.txt"
    if should_skip(out_path, force):
        return out_path

    tmp_path = out_path.with_suffix(".txt.incomplete")
    print("loading Salesforce/wikitext wikitext-103-raw-v1")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    chars = 0
    lines = 0
    split_counts: Dict[str, int] = {}
    with tmp_path.open("w", encoding="utf-8") as f:
        for split in ("train", "validation", "test"):
            split_lines = 0
            for row in dataset[split]:
                text = clean_line(str(row.get("text", "")))
                if not text:
                    continue
                f.write(text + "\n")
                chars += len(text) + 1
                lines += 1
                split_lines += 1
            split_counts[split] = split_lines

    tmp_path.replace(out_path)
    write_metadata(
        out_path,
        {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "splits": ["train", "validation", "test"],
            "lines": lines,
            "characters_with_newlines": chars,
            "split_nonempty_lines": split_counts,
            "format": "plain text for character-level LM",
        },
    )
    print(f"ready: {out_path} chars={chars:,} lines={lines:,}")
    return out_path


def stream_fineweb_edu(config: str) -> Iterator[str]:
    require_datasets()
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        config,
        split="train",
        streaming=True,
    )
    for row in dataset:
        text = clean_line(str(row.get("text", "")))
        if text:
            yield text


def prepare_fineweb_edu(
    data_dir: Path,
    target_chars: int,
    config: str,
    force: bool,
) -> Path:
    suffix = target_chars // 1_000_000
    out_path = data_dir / f"fineweb_edu_{suffix}m.txt"
    if should_skip(out_path, force):
        return out_path

    tmp_path = out_path.with_suffix(".txt.incomplete")
    print(
        "streaming HuggingFaceFW/fineweb-edu "
        f"{config} until about {target_chars:,} characters"
    )

    chars = 0
    lines = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for text in stream_fineweb_edu(config):
            f.write(text + "\n")
            chars += len(text) + 1
            lines += 1
            if chars >= target_chars:
                break

    tmp_path.replace(out_path)
    write_metadata(
        out_path,
        {
            "dataset": "HuggingFaceFW/fineweb-edu",
            "config": config,
            "split": "train",
            "streaming": True,
            "target_characters": target_chars,
            "characters_with_newlines": chars,
            "lines": lines,
            "format": "plain text subset for character-level LM",
        },
    )
    print(f"ready: {out_path} chars={chars:,} lines={lines:,}")
    return out_path


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.corpus in ("all", "wikitext103"):
        outputs.append(prepare_wikitext103(data_dir, force=args.force))
    if args.corpus in ("all", "fineweb_edu"):
        outputs.append(
            prepare_fineweb_edu(
                data_dir,
                target_chars=args.fineweb_chars,
                config=args.fineweb_config,
                force=args.force,
            )
        )

    print("prepared:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
