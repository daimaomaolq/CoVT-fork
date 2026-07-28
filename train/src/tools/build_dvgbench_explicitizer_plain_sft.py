#!/usr/bin/env python3
"""Build plain-text implicit-to-explicit SFT data for DVGBench."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_image(row: dict[str, Any], image_root: Path) -> str:
    image_name = clean_text(
        row.get("image") or row.get("image_path") or row.get("image_id")
    )
    dataset = clean_text(row.get("dataset"))
    relative = Path(image_name.replace("\\", "/"))
    candidates = [image_root / relative]
    if dataset:
        candidates.insert(0, image_root / dataset / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(image_root).as_posix()
    raise FileNotFoundError(f"Missing image: {image_name}")


def explicitizer_prompt(query: str) -> str:
    return (
        "<image>\nRewrite the following implicit UAV grounding request as one "
        "brief explicit visual description of the same target. Use observable "
        "category, color, size, position, context, or relation cues. Output only "
        "the description; do not output reasoning, tags, or a bounding box.\n"
        f"Implicit request: {query}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--query-field", default="question")
    parser.add_argument("--explicit-field", default="question_e")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    output = []
    for index, row in enumerate(rows):
        query = clean_text(row.get(args.query_field))
        explicit = clean_text(row.get(args.explicit_field))
        if not query or not explicit:
            raise ValueError(f"Missing query/question_e at source row {index}")
        output.append(
            {
                "id": f"explicitizer-{index:06d}",
                "image": resolve_image(row, args.image_root),
                "conversations": [
                    {"from": "human", "value": explicitizer_prompt(query)},
                    {"from": "gpt", "value": explicit},
                ],
                "metadata": {
                    "protocol": "implicit_to_explicit_train_only",
                    "question_e_train_supervision": True,
                    "source_class": clean_text(row.get("class")),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "dvgbench-i2e-explicitizer-sft-v1",
        "input_rows": len(rows),
        "output_rows": len(output),
        "question_e_train_supervision": True,
        "question_e_available_at_inference": False,
        "output": str(args.output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
