#!/usr/bin/env python3
"""Build plain-text, single-trajectory I2E plus bbox-preservation SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_dvgbench_generative_sft import (
    bbox_to_qwen_tokens,
    clean_text,
    image_size,
    load_rows,
    parse_bbox,
    resolve_image,
    safe_id,
)


def joint_prompt(query: str) -> str:
    return (
        f"<image>\nLocate the region described by: {query}\n"
        "First write one brief explicit visual description of the same target. "
        "Then provide its bounding box. Keep the original request and all useful "
        "attributes, context, and relations in mind. Respond with exactly two lines:\n"
        "Explicit description: brief visible description\n"
        "Bounding box: {<x1><y1><x2><y2>}"
    )


def direct_prompt(query: str) -> str:
    return (
        f"<image>\nLocate the region described by: {query}\n"
        "Output only the bounding box in the format {<x1><y1><x2><y2>}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--query-field", default="question")
    parser.add_argument("--explicit-field", default="question_e")
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    image_folder = args.image_folder.expanduser().resolve()
    rows = load_rows(args.input_jsonl.expanduser().resolve())
    output = []
    for index, row in enumerate(rows):
        query = clean_text(row.get(args.query_field))
        explicit = clean_text(row.get(args.explicit_field))
        bbox = parse_bbox(row.get("bbox"))
        image_path = resolve_image(row, image_root)
        if not query or not explicit or bbox is None or image_path is None:
            raise ValueError(f"Invalid source row {index}")
        width, height = image_size(image_path)
        bbox_text = bbox_to_qwen_tokens(bbox, width, height)
        try:
            image_value = str(image_path.relative_to(image_folder))
        except ValueError:
            image_value = str(image_path)
        source_id = (
            f"plain_i2e_{index:06d}_{safe_id(row.get('dataset'), 'dataset')}_"
            f"{safe_id(row.get('question_id'), str(index))}"
        )
        output.extend(
            [
                {
                    "id": source_id + "_joint",
                    "image": image_value,
                    "conversations": [
                        {"from": "human", "value": joint_prompt(query)},
                        {
                            "from": "gpt",
                            "value": (
                                f"Explicit description: {explicit}\n"
                                f"Bounding box: {bbox_text}"
                            ),
                        },
                    ],
                    "metadata": {
                        "protocol": "plain_i2e_joint",
                        "question_e_train_supervision": True,
                    },
                },
                {
                    "id": source_id + "_direct",
                    "image": image_value,
                    "conversations": [
                        {"from": "human", "value": direct_prompt(query)},
                        {"from": "gpt", "value": bbox_text},
                    ],
                    "metadata": {
                        "protocol": "answer_only_preservation",
                        "question_e_train_supervision": False,
                    },
                },
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "dvgbench-plain-i2e-sft-v1",
        "source_rows": len(rows),
        "training_rows": len(output),
        "joint_rows": len(rows),
        "direct_preservation_rows": len(rows),
        "question_e_train_supervision": True,
        "question_e_available_at_inference": False,
        "new_schema_tokens_required": False,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
