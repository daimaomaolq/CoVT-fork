#!/usr/bin/env python3
"""Create deterministic class-balanced, disjoint I2E gate splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_no} is not valid JSONL") from err
    return rows


def row_id(row: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "uid"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    composite = (
        row.get("dataset") or row.get("source") or "",
        row.get("image_id") or row.get("image") or "",
        row.get("question_id") or row.get("id") or "",
        row.get("question") or row.get("query") or "",
    )
    if any(value not in (None, "") for value in composite):
        return "::".join(str(value) for value in composite)
    return f"row-{index:06d}"


def row_class(row: dict[str, Any]) -> str:
    value = str(row.get("class") or row.get("category") or "unknown")
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "social_activity": "social",
        "social_activities": "social",
        "productive_activity": "productive",
        "productive_activities": "productive",
        "sports": "sport",
    }
    return aliases.get(value, value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--validation-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        identifier = row_id(row, index)
        if identifier in seen_ids:
            raise ValueError(f"Duplicate source id: {identifier}")
        seen_ids.add(identifier)
        grouped[row_class(row)].append((index, row))

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    class_counts: dict[str, dict[str, int]] = {}
    for class_index, class_name in enumerate(sorted(grouped)):
        candidates = list(grouped[class_name])
        random.Random(args.seed + class_index).shuffle(candidates)
        required = args.train_per_class + args.validation_per_class
        if len(candidates) < required:
            raise ValueError(
                f"Class {class_name!r} has {len(candidates)} rows, requires {required}."
            )
        train.extend(row for _, row in candidates[: args.train_per_class])
        validation.extend(row for _, row in candidates[args.train_per_class : required])
        class_counts[class_name] = {
            "available": len(candidates),
            "train": args.train_per_class,
            "validation": args.validation_per_class,
        }

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(validation)
    train_ids = {row_id(row, index) for index, row in enumerate(train)}
    validation_ids = {row_id(row, index) for index, row in enumerate(validation)}
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"Train/validation leakage: {sorted(overlap)[:5]}")

    write_jsonl(args.train_output, train)
    write_jsonl(args.validation_output, validation)
    manifest = {
        "schema_version": "dvgbench-i2e-heldout-gate-split-v1",
        "seed": args.seed,
        "source_rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "overlap": 0,
        "class_counts": class_counts,
        "train_output": str(args.train_output),
        "validation_output": str(args.validation_output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
