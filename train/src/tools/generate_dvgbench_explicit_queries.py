#!/usr/bin/env python3
"""Generate oracle-free explicit descriptions from implicit DVGBench queries."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from eval_dvgbench_generative_grounding import (
    build_anchor_prompt,
    insert_anchor_prompt,
    load_model,
    parse_anchor_model_ids,
    parse_anchor_token_counts,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def prompt_for_explicitizer(query: str) -> str:
    return (
        "Rewrite the following implicit UAV grounding request as one brief "
        "explicit visual description of the same target. Use observable "
        "category, color, size, position, context, or relation cues. Output "
        "only the description; do not output reasoning, tags, or a bounding "
        f"box.\nImplicit request: {query}"
    )


def sanitize_explicit(raw: str, implicit: str) -> tuple[str, bool]:
    text = clean_text(raw)
    text = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text).strip()
    text = re.sub(r"</?(?:think|explicit|answer)>", "", text, flags=re.I).strip()
    text = re.sub(r"\{?\s*<\s*[\d.,\s-]{7,}\s*>\s*\}?", "", text).strip()
    text = text.splitlines()[0].strip() if text else ""
    words = text.split()
    if len(words) > 48:
        text = " ".join(words[:48]).rstrip(" ,;:")
    fallback = not bool(text)
    return (implicit if fallback else text), fallback


def resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    raw = clean_text(row.get("image") or row.get("image_path") or row.get("image_id"))
    relative = Path(raw.replace("\\", "/"))
    candidates = [image_root / relative]
    dataset = clean_text(row.get("dataset"))
    if dataset:
        candidates.insert(0, image_root / dataset / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing image for {raw}")


def generate_one(model, processor, device, image: Path, query: str, args) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image),
                    "min_pixels": args.image_min_pixels,
                    "max_pixels": args.image_max_pixels,
                },
                {"type": "text", "text": prompt_for_explicitizer(query)},
            ],
        }
    ]
    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    anchor_ids = parse_anchor_model_ids(args.anchor_model_id)
    anchor_counts = parse_anchor_token_counts(anchor_ids, args.anchor_token_counts)
    text = insert_anchor_prompt(
        text, build_anchor_prompt(anchor_ids, anchor_counts), args.anchor_prompt_mode
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
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--anchor-model-id", default="sam,dino")
    parser.add_argument("--anchor-token-counts", default="8,4")
    parser.add_argument("--anchor-prompt-mode", default="query_tail")
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.index)
    forbidden = {"question_e", "question_e_cn", "explicit_reference"}
    contaminated = [
        str(row.get("sample_id", i))
        for i, row in enumerate(rows)
        if any(field in row for field in forbidden)
        or bool(row.get("oracle_fields_present"))
    ]
    if contaminated:
        raise ValueError(f"Oracle-contaminated inference index: {contaminated[:5]}")

    model, processor, device = load_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fallback_count = 0
    latencies: list[float] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            implicit = clean_text(row.get(args.query_field))
            if not implicit:
                raise ValueError(f"Missing implicit query at row {index}")
            started = time.perf_counter()
            raw = generate_one(
                model,
                processor,
                device,
                resolve_image(row, args.image_root),
                implicit,
                args,
            )
            latency = time.perf_counter() - started
            explicit, fallback = sanitize_explicit(raw, implicit)
            fallback_count += int(fallback)
            latencies.append(latency)
            record = {
                "schema_version": "dvgbench-i2e-explicit-sidecar-v1",
                "sample_id": row.get("sample_id", index),
                "image": row.get("image") or row.get("image_path"),
                "implicit_query": implicit,
                "generated_explicit": explicit,
                "raw_output": raw,
                "fallback_to_implicit": fallback,
                "latency_seconds": latency,
                "question_e_used": False,
                "gt_visible_during_inference": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index + 1}/{len(rows)}] fallback={fallback}", flush=True)

    manifest = {
        "schema_version": "dvgbench-i2e-explicit-sidecar-manifest-v1",
        "rows": len(rows),
        "unique_sample_ids": len(
            {str(row.get("sample_id", i)) for i, row in enumerate(rows)}
        ),
        "fallback_count": fallback_count,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "question_e_used": False,
        "gt_visible_during_inference": False,
        "final_bbox_regenerated": False,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
