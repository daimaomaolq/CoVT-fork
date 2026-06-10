from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BOX_RE = re.compile(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}")
TAG_RE = re.compile(r"\[([^\]]+)\]")
PHRASE_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adapter JSONL indexes from ZhanYang-nwpu/UAVBench conversations."
    )
    parser.add_argument("--hf-dataset", default="ZhanYang-nwpu/UAVBench")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--tasks",
        default="vg,count,reg_cls",
        help="Comma-separated UAVBench tags to export. Supported: vg,count,reg_cls.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for task JSONLs.")
    parser.add_argument(
        "--source-image-root",
        default=None,
        help="Optional root used to resolve relative UAVBench image paths.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def parse_conversations(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    raise TypeError(f"Unsupported conversations type: {type(raw)}")


def clean_image_path(value: Any) -> str:
    return str(value).strip().strip("'\"")


def resolve_image_path(image: str, source_root: Path | None) -> str:
    image = clean_image_path(image)
    path = Path(image)
    if path.exists():
        return str(path.resolve())
    if source_root is not None:
        candidate = source_root / image
        if candidate.exists():
            return str(candidate.resolve())
    return image


def safe_id(value: Any, fallback: str) -> str:
    text = str(value if value is not None else fallback).strip().strip("'\"")
    chars = [char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text]
    return "".join(chars).strip("_") or fallback


def extract_tag(prompt: str) -> str:
    match = TAG_RE.search(prompt)
    return match.group(1).strip() if match else "unknown"


def strip_prompt(prompt: str) -> str:
    text = str(prompt).replace("<image>", " ")
    text = TAG_RE.sub(" ", text)
    return " ".join(text.split())


def parse_box(text: str) -> list[float] | None:
    match = BOX_RE.search(str(text))
    if not match:
        return None
    values = [float(value) / 1000.0 for value in match.groups()]
    x1, y1, x2, y2 = values
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def extract_phrase(prompt: str) -> str:
    match = PHRASE_RE.search(prompt)
    if match:
        return " ".join(match.group(1).split())
    return strip_prompt(prompt)


def normalize_answer(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text).strip().lower())
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def build_record(row: dict[str, Any], task: str, source_root: Path | None, counts: Counter[str]) -> dict[str, Any] | None:
    conversations = parse_conversations(row["conversations"])
    if len(conversations) < 2:
        counts["missing_turn"] += 1
        return None
    prompt = str(conversations[0].get("value", ""))
    answer = str(conversations[1].get("value", ""))
    image = resolve_image_path(row.get("image", ""), source_root)
    row_id = safe_id(row.get("id"), str(counts["seen"]))
    image_stem = safe_id(Path(clean_image_path(row.get("image", ""))).stem, row_id)
    sample_id = f"uavbench_{task}_{row_id}_{image_stem}"

    if task == "vg":
        bbox = parse_box(answer)
        if bbox is None:
            counts["vg_missing_bbox"] += 1
            return None
        query = extract_phrase(prompt)
        return {
            "sample_id": sample_id,
            "image": image,
            "query": query,
            "bbox_norm": bbox,
            "task_type": "grounding",
            "task_tag": "uavbench_vg",
            "source": "UAVBench",
            "answer": answer,
            "raw_prompt": prompt,
        }

    if task == "count":
        query = strip_prompt(prompt)
        norm_answer = normalize_answer(answer)
        if not re.fullmatch(r"\d+", norm_answer):
            counts["count_non_integer"] += 1
            return None
        return {
            "sample_id": sample_id,
            "image": image,
            "query": query,
            "answer": norm_answer,
            "task_type": "image_answer",
            "task_tag": "uavbench_count",
            "source": "UAVBench",
            "raw_prompt": prompt,
        }

    if task == "reg_cls":
        region = parse_box(prompt)
        if region is None:
            counts["reg_cls_missing_region"] += 1
            return None
        query = strip_prompt(prompt)
        return {
            "sample_id": sample_id,
            "image": image,
            "query": query,
            "region_norm": region,
            "bbox_norm": region,
            "answer": normalize_answer(answer),
            "task_type": "region_answer",
            "task_tag": "uavbench_reg_cls",
            "source": "UAVBench",
            "raw_prompt": prompt,
        }

    counts[f"unsupported_{task}"] += 1
    return None


def main() -> None:
    args = parse_args()
    tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
    supported = {"vg", "count", "reg_cls"}
    unknown = tasks - supported
    if unknown:
        raise ValueError(f"Unsupported tasks: {sorted(unknown)}")

    from datasets import load_dataset

    dataset = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_root = Path(args.source_image_root).expanduser().resolve() if args.source_image_root else None
    if not args.inspect_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    try:
        if not args.inspect_only:
            for task in tasks:
                handles[task] = (output_dir / f"uavbench_{task}.jsonl").open("w", encoding="utf-8")

        for row in dataset:
            if args.limit is not None and counts["seen"] >= args.limit:
                break
            counts["seen"] += 1
            conversations = parse_conversations(row["conversations"])
            prompt = str(conversations[0].get("value", "")) if conversations else ""
            task = extract_tag(prompt)
            counts[f"tag_{task}"] += 1
            if task not in tasks:
                continue
            if args.max_per_task is not None and counts[f"written_{task}"] >= args.max_per_task:
                continue
            record = build_record(row, task, source_root, counts)
            if record is None:
                continue
            counts[f"written_{task}"] += 1
            if len(examples[task]) < 3:
                examples[task].append(record)
            if not args.inspect_only:
                handles[task].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    print(json.dumps({"counts": dict(counts), "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
