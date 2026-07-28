#!/usr/bin/env python3
"""Evaluate plain-text single-trajectory I2E while publishing bbox only."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval_dvgbench_generative_grounding import (
    box_iou,
    build_anchor_prompt,
    insert_anchor_prompt,
    load_model,
    parse_anchor_model_ids,
    parse_anchor_token_counts,
    parse_bbox_text,
)


def prompt_for_query(query: str) -> str:
    return (
        f"Locate the region described by: {query}\n"
        "First write one brief explicit visual description of the same target. "
        "Then provide its bounding box. Keep the original request and all useful "
        "attributes, context, and relations in mind. Respond with exactly two lines:\n"
        "Explicit description: brief visible description\n"
        "Bounding box: {<x1><y1><x2><y2>}"
    )


def parse_explicit(text: str) -> str | None:
    match = re.search(
        r"Explicit\s+description\s*:\s*(.*?)\s*(?=Bounding\s+box\s*:)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value or None


def schema_valid(text: str) -> bool:
    marker = re.search(r"Bounding\s+box\s*:", text, flags=re.IGNORECASE)
    return (
        parse_explicit(text) is not None
        and marker is not None
        and parse_bbox_text(text[marker.end() :]) is not None
    )


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit is not None else rows


def generate_one(model, processor, device, image_path: str, query: str, args) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": args.image_min_pixels,
                    "max_pixels": args.image_max_pixels,
                },
                {"type": "text", "text": prompt_for_query(query)},
            ],
        }
    ]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    anchor_ids = parse_anchor_model_ids(args.anchor_model_id)
    counts = parse_anchor_token_counts(anchor_ids, args.anchor_token_counts)
    text = insert_anchor_prompt(
        text, build_anchor_prompt(anchor_ids, counts), args.anchor_prompt_mode
    )
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return processor.decode(
        output_ids[0, prompt_length:], skip_special_tokens=False
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--anchor-model-id", default="sam,dino")
    parser.add_argument("--anchor-token-counts", default="8,4")
    parser.add_argument("--anchor-prompt-mode", default="query_tail")
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.index, args.limit)
    forbidden = {"question_e", "question_e_cn", "explicit_reference"}
    if any(
        any(field in row for field in forbidden) or row.get("oracle_fields_present")
        for row in rows
    ):
        raise ValueError("Inference index contains an explicit oracle field.")
    model, processor, device = load_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    class_total: dict[str, int] = defaultdict(int)
    class_hits: dict[str, int] = defaultdict(int)
    iou_sum = 0.0
    hit_sum = 0
    parse_failed = 0
    explicit_failed = 0
    schema_failed = 0
    latencies = []
    with args.output.open("w", encoding="utf-8") as public_handle, args.trace_output.open(
        "w", encoding="utf-8"
    ) as trace_handle:
        for index, row in enumerate(rows):
            query = str(row[args.query_field]).strip()
            started = time.perf_counter()
            raw = generate_one(model, processor, device, str(row["image"]), query, args)
            latency = time.perf_counter() - started
            bbox = parse_bbox_text(raw)
            explicit = parse_explicit(raw)
            gt_bbox = [float(value) for value in row["bbox"]]
            iou = box_iou(bbox, gt_bbox)
            hit = iou >= 0.5
            cls = str(row.get("class", "unknown")).strip().lower()
            iou_sum += iou
            hit_sum += int(hit)
            parse_failed += int(bbox is None)
            explicit_failed += int(explicit is None)
            schema_failed += int(not schema_valid(raw))
            class_total[cls] += 1
            class_hits[cls] += int(hit)
            latencies.append(latency)
            public_handle.write(
                json.dumps(
                    {
                        "sample_id": row.get("sample_id", index),
                        "bbox": bbox,
                        "protocol": {
                            "final_output": "bbox_only",
                            "question_e_used": False,
                            "gt_visible_during_inference": False,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            trace_handle.write(
                json.dumps(
                    {
                        "sample_id": row.get("sample_id", index),
                        "query": query,
                        "explicit_prediction": explicit,
                        "raw_internal_trajectory": raw,
                        "bbox": bbox,
                        "gt_bbox": gt_bbox,
                        "iou": iou,
                        "class": cls,
                        "latency_seconds": latency,
                        "question_e_used": False,
                        "gt_visible_during_inference": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            public_handle.flush()
            trace_handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] parse={bbox is not None} "
                f"explicit={explicit is not None} iou={iou:.4f}",
                flush=True,
            )

    class_acc = {
        key: class_hits[key] / class_total[key] for key in sorted(class_total)
    }
    summary = {
        "schema_version": "dvgbench-plain-i2e-bbox-only-v1",
        "samples": len(rows),
        "mIoU": iou_sum / max(len(rows), 1),
        "Acc@0.5": hit_sum / max(len(rows), 1),
        "DVGBench_AVG": sum(class_acc.values()) / max(len(class_acc), 1),
        "class_Acc@0.5": class_acc,
        "class_counts": dict(sorted(class_total.items())),
        "parse_failed": parse_failed,
        "explicit_parse_failed": explicit_failed,
        "schema_parse_failed": schema_failed,
        "schema_format_rate": (len(rows) - schema_failed) / max(len(rows), 1),
        "mean_latency_seconds": sum(latencies) / max(len(latencies), 1),
        "question_e_used": False,
        "gt_visible_during_inference": False,
        "final_output": "bbox_only",
        "predictions": str(args.output),
        "trace": str(args.trace_output),
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
