from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


COORD_RE = re.compile(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}")
TAG_RE = re.compile(r"\[([^\]]+)\]")
PHRASE_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
CLASS_LIST_RE = re.compile(r"given classes:\s*(.*?)(?:\.|\n|$)", re.IGNORECASE | re.DOTALL)


GROUNDING_TAGS = {"vg"}
DETECTION_TAGS = {"det"}
IMAGE_ANSWER_TAGS = {"img_cls", "deta_cls", "count", "vqa"}
REGION_ANSWER_TAGS = {"reg_vqa", "reg_cls"}
CAPTION_TAGS = {"img_cap", "deta_cap"}
REGION_CAPTION_TAGS = {"reg_cap"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified multitask JSONL indexes from UAVIT-1M conversations."
    )
    parser.add_argument("--hf-dataset", default="ZhanYang-nwpu/UAVIT-1M")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for uavit_multitask_all.jsonl and per-head JSONLs.",
    )
    parser.add_argument(
        "--source-image-root",
        default=None,
        help="Optional local image root. If provided, write absolute image path when the file exists.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-per-tag", type=int, default=None)
    parser.add_argument(
        "--write-examples",
        type=int,
        default=3,
        help="Number of examples per tag to store in summary.",
    )
    return parser.parse_args()


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("<image>", " ")
    text = re.sub(r"</?p>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_id(text: Any, fallback: str) -> str:
    value = str(text if text is not None else fallback)
    out = []
    for char in value:
        out.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "_".join("".join(out).split("_")) or fallback


def task_tag(human: str) -> str:
    match = TAG_RE.search(human or "")
    return match.group(1) if match else "no_tag"


def first_turn(conversations: list[dict[str, Any]], speaker: str) -> str:
    for turn in conversations:
        if turn.get("from") == speaker:
            return str(turn.get("value", ""))
    return ""


def parse_boxes(text: str) -> list[list[float]]:
    boxes = []
    for match in COORD_RE.finditer(text or ""):
        values = [float(match.group(i)) for i in range(1, 5)]
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
    return boxes


def norm_box(box: list[float]) -> list[float]:
    return [min(max(value / 100.0, 0.0), 1.0) for value in box]


def parse_phrase(text: str) -> str:
    match = PHRASE_RE.search(text or "")
    if match:
        return clean_text(match.group(1))
    text = re.sub(r"\[[^\]]+\]", "", text or "")
    text = re.sub(r"\{<\d+><\d+><\d+><\d+>\}", "region", text)
    return clean_text(text)


def parse_options(text: str) -> list[str]:
    match = CLASS_LIST_RE.search(text or "")
    if not match:
        return []
    raw = match.group(1)
    raw = raw.replace(" and ", ", ")
    return [item.strip(" .") for item in raw.split(",") if item.strip(" .")]


def resolve_image(image_rel: str, source_image_root: Path | None) -> str:
    if source_image_root is None:
        return image_rel
    path = source_image_root / image_rel
    return str(path.resolve()) if path.exists() else image_rel


def task_type_for_tag(tag: str) -> str:
    if tag in GROUNDING_TAGS:
        return "grounding"
    if tag in DETECTION_TAGS:
        return "detection"
    if tag in IMAGE_ANSWER_TAGS:
        return "image_answer"
    if tag in REGION_ANSWER_TAGS:
        return "region_answer"
    if tag in CAPTION_TAGS:
        return "caption"
    if tag in REGION_CAPTION_TAGS:
        return "region_caption"
    return "unknown"


def build_row(row: dict[str, Any], row_idx: int, source_image_root: Path | None) -> dict[str, Any] | None:
    conversations = row.get("conversations") or []
    human = first_turn(conversations, "human")
    answer = first_turn(conversations, "gpt")
    tag = task_tag(human)
    task_type = task_type_for_tag(tag)
    if task_type == "unknown":
        return None

    image_rel = str(row.get("image") or row.get("id") or "")
    sample_base = safe_id(row.get("id") or image_rel or row_idx, str(row_idx))
    out: dict[str, Any] = {
        "sample_id": f"uavit_{row_idx:08d}_{tag}_{sample_base}",
        "source": "UAVIT-1M",
        "split": "train",
        "task_tag": tag,
        "task_type": task_type,
        "id": row.get("id"),
        "image_rel": image_rel,
        "image": resolve_image(image_rel, source_image_root),
        "human": human,
        "answer": clean_text(answer),
        "query_version": "uavit_multitask_v1",
    }

    human_boxes = parse_boxes(human)
    answer_boxes = parse_boxes(answer)

    if tag == "vg":
        if not answer_boxes:
            return None
        query = parse_phrase(human)
        out.update(
            {
                "query": query,
                "bbox_0100": answer_boxes[0],
                "bbox_norm": norm_box(answer_boxes[0]),
            }
        )
        return out

    if tag == "det":
        phrase = parse_phrase(human)
        if not answer_boxes:
            return None
        out.update(
            {
                "query": phrase,
                "bboxes_0100": answer_boxes,
                "bboxes_norm": [norm_box(box) for box in answer_boxes],
                "bbox_count": len(answer_boxes),
            }
        )
        return out

    if tag in REGION_ANSWER_TAGS or tag in REGION_CAPTION_TAGS:
        if not human_boxes:
            return None
        out.update(
            {
                "query": parse_phrase(human),
                "region_0100": human_boxes[0],
                "region_norm": norm_box(human_boxes[0]),
                "bbox_0100": human_boxes[0],
                "bbox_norm": norm_box(human_boxes[0]),
            }
        )
        if tag in REGION_CAPTION_TAGS:
            out["caption"] = clean_text(answer)
        return out

    if tag in IMAGE_ANSWER_TAGS:
        out.update(
            {
                "query": parse_phrase(human),
                "options": parse_options(human),
            }
        )
        return out

    if tag in CAPTION_TAGS:
        out.update(
            {
                "query": parse_phrase(human),
                "caption": clean_text(answer),
            }
        )
        return out

    return out


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_image_root = Path(args.source_image_root).expanduser().resolve() if args.source_image_root else None

    paths = {
        "all": output_root / "uavit_multitask_all_v1.jsonl",
        "grounding": output_root / "uavit_grounding_v1.jsonl",
        "detection": output_root / "uavit_detection_v1.jsonl",
        "answer": output_root / "uavit_answer_v1.jsonl",
        "caption": output_root / "uavit_caption_v1.jsonl",
    }
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}

    try:
        dataset = load_dataset(args.hf_dataset, split=args.split, streaming=True)
        for row_idx, row in enumerate(dataset):
            if args.limit is not None and counts["seen"] >= args.limit:
                break
            counts["seen"] += 1
            human = first_turn(row.get("conversations") or [], "human")
            tag = task_tag(human)
            if args.max_per_tag is not None and counts[f"written_tag_{tag}"] >= args.max_per_tag:
                counts[f"skipped_max_{tag}"] += 1
                continue
            out = build_row(row, row_idx, source_image_root)
            if out is None:
                counts[f"skipped_{tag}"] += 1
                continue

            line = json.dumps(out, ensure_ascii=False) + "\n"
            handles["all"].write(line)
            task_type = out["task_type"]
            if task_type in {"grounding", "detection"}:
                handles[task_type].write(line)
            elif task_type in {"image_answer", "region_answer"}:
                handles["answer"].write(line)
            elif task_type in {"caption", "region_caption"}:
                handles["caption"].write(line)

            counts["written"] += 1
            counts[f"tag_{tag}"] += 1
            counts[f"written_tag_{tag}"] += 1
            counts[f"task_{task_type}"] += 1
            examples.setdefault(tag, [])
            if len(examples[tag]) < args.write_examples:
                compact = {key: out.get(key) for key in (
                    "sample_id",
                    "task_type",
                    "image",
                    "query",
                    "answer",
                    "caption",
                    "bbox_norm",
                    "bboxes_norm",
                    "options",
                ) if key in out}
                examples[tag].append(compact)

            if counts["seen"] % 50000 == 0:
                print(json.dumps(dict(counts), ensure_ascii=False), flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": dict(counts),
        "examples": examples,
    }
    (output_root / "uavit_multitask_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
