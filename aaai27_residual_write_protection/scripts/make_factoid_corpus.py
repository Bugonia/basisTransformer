#!/usr/bin/env python3
"""Create a controlled factoid adaptation corpus.

The corpus is intentionally synthetic: the model should not know these facts
before adaptation. Each fact has a prompt/completion pair for direct evaluation
and repeated training text to create a clear new-knowledge write pressure.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


STEMS = [
    "alvo",
    "bexa",
    "cori",
    "daro",
    "elun",
    "faro",
    "galo",
    "hiva",
    "iona",
    "juno",
    "kavo",
    "lira",
    "mavo",
    "nexo",
    "oril",
    "pavo",
    "quon",
    "rila",
    "savo",
    "tavi",
    "ulon",
    "vexa",
    "wiro",
    "xani",
    "yavo",
    "zuri",
]

RELATIONS = [
    "archive code",
    "calibration key",
    "routing label",
    "index marker",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-train-facts", type=int, default=512)
    parser.add_argument("--num-heldout-facts", type=int, default=128)
    parser.add_argument("--seen-eval-facts", type=int, default=128)
    parser.add_argument("--train-repeats", type=int, default=16)
    parser.add_argument("--relation", default="archive code")
    return parser.parse_args()


def entity_name(index: int) -> str:
    a = STEMS[index % len(STEMS)]
    b = STEMS[(index * 7 + 3) % len(STEMS)]
    c = STEMS[(index * 11 + 5) % len(STEMS)]
    return f"unit-{a}-{b}-{c}-{index:04d}"


def answer_code(index: int) -> str:
    left = (index * 37 + 113) % 10000
    right = (index * 91 + 271) % 10000
    return f"ZX-{left:04d}-{right:04d}"


def make_record(index: int, relation: str, split: str) -> Dict[str, str]:
    entity = entity_name(index)
    answer = answer_code(index)
    prompt = f"Question: What is the {relation} for {entity}?\nAnswer:"
    completion = f" {answer}."
    fact = f"The {relation} for {entity} is {answer}."
    qa = prompt + completion
    return {
        "id": f"{split}-{index:04d}",
        "split": split,
        "entity": entity,
        "relation": relation,
        "answer": answer,
        "fact": fact,
        "prompt": prompt,
        "completion": completion,
        "text": qa,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, blocks: Iterable[str]) -> None:
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relation = args.relation

    train_records = [
        make_record(i, relation, "train") for i in range(args.num_train_facts)
    ]
    heldout_records = [
        make_record(args.num_train_facts + i, relation, "heldout")
        for i in range(args.num_heldout_facts)
    ]
    seen_eval = train_records[: min(args.seen_eval_facts, len(train_records))]

    train_blocks: List[str] = []
    for _ in range(args.train_repeats):
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        for row in shuffled:
            train_blocks.append(f"{row['fact']}\n{row['text']}")

    write_text(output_dir / "train.txt", train_blocks)
    write_text(output_dir / "eval_seen.txt", [row["text"] for row in seen_eval])
    write_text(output_dir / "eval_heldout.txt", [row["text"] for row in heldout_records])
    write_jsonl(output_dir / "eval_seen.jsonl", seen_eval)
    write_jsonl(output_dir / "eval_heldout.jsonl", heldout_records)
    write_jsonl(output_dir / "train_facts.jsonl", train_records)

    metadata = {
        "seed": args.seed,
        "relation": relation,
        "num_train_facts": args.num_train_facts,
        "num_heldout_facts": args.num_heldout_facts,
        "seen_eval_facts": len(seen_eval),
        "train_repeats": args.train_repeats,
        "files": {
            "train": "train.txt",
            "eval_seen": "eval_seen.txt",
            "eval_heldout": "eval_heldout.txt",
            "eval_seen_manifest": "eval_seen.jsonl",
            "eval_heldout_manifest": "eval_heldout.jsonl",
            "train_facts": "train_facts.jsonl",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote factoid corpus to {output_dir}")


if __name__ == "__main__":
    main()
