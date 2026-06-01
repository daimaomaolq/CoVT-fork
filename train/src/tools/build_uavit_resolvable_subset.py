from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COORD_RE = re.compile(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}")
TAG_RE = re.compile(r"\[([^\]]+)\]")
PHRASE_RE = re.compile(r"<p>(.*?)</p>", re.IGNORECASE | re.DOTALL)
CLASS_LIST_RE = re.compile(r"given classes:\s*(.*?)(?:\.|\n|$)", re.IGNORECASE | re.DOTALL)

UNDERSTANDING_TAGS = {"img_cls", "deta_cls", "count", "vqa", "img_cap", "deta_cap", "reg_cap"}
GROUNDING_TAGS = {"vg", "det", "reg_vqa", "reg_cls"}
SUPPORTED_TAGS = UNDERSTANDING_TAGS | GROUNDING_TAGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build UAVIT-1M resolvable subset from local image files."
    )
    parser.add_argument("--uavit-json", default=None, help="Local UAVIT-1M.json. If omitted, use HF streaming.")
    parser.add_argument("--hf-dataset", default="ZhanYang-nwpu/UAVIT-1M")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/datasets")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke-size", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--visdrone-grounding-max-frac", type=float, default=0.2)
    parser.add_argument(
        "--min-suffix-parts",
        type=int,
        default=2,
        help="Minimum matching path suffix parts for resolving an image. 2 means parent directory + filename.",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--write-examples", type=int, default=20)
    return parser.parse_args()


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("<image>", " ")
    text = re.sub(r"</?p>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_id(text: Any, fallback: str) -> str:
    value = str(text if text is not None else fallback)
    out = [char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value]
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
        box = [float(match.group(i)) for i in range(1, 5)]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
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
    raw = match.group(1).replace(" and ", ", ")
    return [item.strip(" .") for item in raw.split(",") if item.strip(" .")]


def source_prefix(image_rel: str) -> str:
    parts = normalize_rel(image_rel).split("/")
    if not parts:
        return "unknown"
    if len(parts) >= 2 and parts[0] in {"VisDrone2019_DET", "UAV123", "WebUAV-3M"}:
        return parts[0]
    if len(parts) >= 3 and parts[0] == "ERA":
        return "ERA"
    return parts[0] or "unknown"


def normalize_rel(value: str) -> str:
    value = str(value or "").strip().strip("'\"")
    value = value.replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    return value.lstrip("/")


def scan_images(dataset_root: Path) -> dict[str, list[Path]]:
    by_basename: dict[str, list[Path]] = defaultdict(list)
    count = 0
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        by_basename[path.name].append(path)
        count += 1
        if count % 100000 == 0:
            print(json.dumps({"scanned_images": count}, ensure_ascii=False), flush=True)
    print(json.dumps({"scanned_images": count, "unique_basenames": len(by_basename)}, ensure_ascii=False), flush=True)
    return by_basename


def common_suffix_score(candidate: Path, image_rel: str) -> int:
    cand_parts = normalize_rel(str(candidate)).lower().split("/")
    rel_parts = normalize_rel(image_rel).lower().split("/")
    score = 0
    for c, r in zip(reversed(cand_parts), reversed(rel_parts)):
        if c != r:
            break
        score += 1
    return score


def resolve_image(image_rel: str, by_basename: dict[str, list[Path]], min_suffix_parts: int) -> str | None:
    rel = normalize_rel(image_rel)
    if not rel:
        return None
    direct = Path(rel)
    if direct.is_absolute() and direct.exists():
        return str(direct)
    candidates = by_basename.get(Path(rel).name, [])
    if not candidates:
        return None
    scored = [(common_suffix_score(path, rel), path) for path in candidates]
    scored.sort(key=lambda item: (item[0], -len(str(item[1]))), reverse=True)
    if scored[0][0] < min_suffix_parts:
        return None
    return str(scored[0][1])


def iter_rows(args: argparse.Namespace) -> Iterable[tuple[int, dict[str, Any]]]:
    if args.uavit_json:
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=args.uavit_json, split="train", streaming=True)
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    for row_idx, row in enumerate(dataset):
        if args.limit is not None and row_idx >= args.limit:
            break
        yield row_idx, row


def build_schema(row: dict[str, Any], row_idx: int, resolved_image: str) -> dict[str, Any] | None:
    conversations = row.get("conversations") or []
    human = first_turn(conversations, "human")
    answer = first_turn(conversations, "gpt")
    tag = task_tag(human)
    if tag not in SUPPORTED_TAGS:
        return None
    image_rel = normalize_rel(row.get("image") or row.get("id") or "")
    base = {
        "sample_id": f"uavit_resolvable_{row_idx:08d}_{tag}_{safe_id(row.get('id') or image_rel, str(row_idx))}",
        "source": source_prefix(image_rel),
        "dataset": "UAVIT-1M",
        "image": resolved_image,
        "image_rel": image_rel,
        "task_tag": tag,
        "query_version": "uavit_resolvable_v1",
    }
    human_boxes = parse_boxes(human)
    answer_boxes = parse_boxes(answer)

    if tag in {"vg"}:
        if not answer_boxes:
            return None
        bbox = norm_box(answer_boxes[0])
        base.update(
            {
                "task": "grounding",
                "task_type": "grounding",
                "query": parse_phrase(human),
                "bbox": bbox,
                "bbox_norm": bbox,
            }
        )
        return base
    if tag == "det":
        if not answer_boxes:
            return None
        bboxes = [norm_box(box) for box in answer_boxes]
        base.update(
            {
                "task": "grounding",
                "task_type": "detection",
                "query": parse_phrase(human),
                "bbox": bboxes[0],
                "bbox_norm": bboxes[0],
                "bboxes": bboxes,
                "bboxes_norm": bboxes,
                "bbox_count": len(bboxes),
            }
        )
        return base
    if tag in {"reg_vqa", "reg_cls"}:
        if not human_boxes:
            return None
        bbox = norm_box(human_boxes[0])
        base.update(
            {
                "task": "grounding",
                "task_type": "region_answer",
                "query": parse_phrase(human),
                "bbox": bbox,
                "bbox_norm": bbox,
                "answer": clean_text(answer),
            }
        )
        return base
    if tag in {"img_cls", "deta_cls", "count", "vqa"}:
        base.update(
            {
                "task": "understanding",
                "task_type": "image_answer",
                "query": parse_phrase(human),
                "answer": clean_text(answer),
            }
        )
        options = parse_options(human)
        if options:
            base["choices"] = options
            base["options"] = options
        return base
    if tag in {"img_cap", "deta_cap", "reg_cap"}:
        if tag == "reg_cap" and human_boxes:
            bbox = norm_box(human_boxes[0])
            base["region"] = bbox
            base["region_norm"] = bbox
            base["bbox_norm"] = bbox
        task_type = "region_caption" if tag == "reg_cap" and human_boxes else "caption"
        base.update(
            {
                "task": "understanding",
                "task_type": task_type,
                "query": parse_phrase(human),
                "answer": clean_text(answer),
                "caption": clean_text(answer),
            }
        )
        return base
    return None


def cap_visdrone_grounding(rows: list[dict[str, Any]], max_frac: float, rng: random.Random) -> list[dict[str, Any]]:
    grounding = [row for row in rows if row["task"] == "grounding"]
    other_ground = [row for row in grounding if row["source"] != "VisDrone2019_DET"]
    vis_ground = [row for row in grounding if row["source"] == "VisDrone2019_DET"]
    if not grounding or not vis_ground:
        return rows
    max_vis = int(max_frac * max(len(other_ground), 1) / max(1.0 - max_frac, 1e-6))
    if len(vis_ground) <= max_vis:
        return rows
    keep_ids = {id(row) for row in rng.sample(vis_ground, max_vis)}
    return [row for row in rows if row["task"] != "grounding" or row["source"] != "VisDrone2019_DET" or id(row) in keep_ids]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["rows"] += 1
        counts[f"task_{row['task']}"] += 1
        counts[f"tag_{row['task_tag']}"] += 1
        counts[f"source_{row['source']}"] += 1
        counts[f"source_task_{row['source']}::{row['task']}"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    by_basename = scan_images(dataset_root)
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    resolved_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for row_idx, row in iter_rows(args):
        image_rel = normalize_rel(row.get("image") or row.get("id") or "")
        prefix = source_prefix(image_rel)
        human = first_turn(row.get("conversations") or [], "human")
        tag = task_tag(human)
        task_family = "unsupported"
        if tag in UNDERSTANDING_TAGS:
            task_family = "understanding"
        elif tag in GROUNDING_TAGS:
            task_family = "grounding"
        source_stats[prefix]["total_rows"] += 1
        source_stats[prefix][f"tag_{tag}"] += 1
        source_stats[prefix][f"task_{task_family}"] += 1
        resolved_image = resolve_image(image_rel, by_basename, args.min_suffix_parts)
        if not resolved_image:
            continue
        source_stats[prefix]["resolved_rows"] += 1
        out = build_schema(row, row_idx, resolved_image)
        if out is None:
            source_stats[prefix]["resolved_but_filtered"] += 1
            continue
        source_stats[prefix][f"resolved_task_{out['task']}"] += 1
        source_stats[prefix][f"resolved_tag_{out['task_tag']}"] += 1
        resolved_rows.append(out)
        if len(examples) < args.write_examples:
            examples.append(out)
        if len(resolved_rows) % 5000 == 0:
            print(json.dumps({"resolved_kept": len(resolved_rows), "row_idx": row_idx}, ensure_ascii=False), flush=True)

    resolved_rows = cap_visdrone_grounding(resolved_rows, args.visdrone_grounding_max_frac, rng)
    rng.shuffle(resolved_rows)
    smoke = resolved_rows[: args.smoke_size]
    train_start = args.smoke_size
    train = resolved_rows[train_start : train_start + args.train_size]
    val_start = train_start + args.train_size
    val = resolved_rows[val_start : val_start + args.val_size]

    paths = {
        "smoke": output_root / "uavit_resolvable_5k_smoke.jsonl",
        "train": output_root / "uavit_resolvable_50k_train.jsonl",
        "val": output_root / "uavit_resolvable_5k_val.jsonl",
    }
    write_jsonl(paths["smoke"], smoke)
    write_jsonl(paths["train"], train)
    write_jsonl(paths["val"], val)
    summary = {
        "paths": {key: str(path) for key, path in paths.items()},
        "source_stats": {source: dict(counter) for source, counter in sorted(source_stats.items())},
        "resolved_after_cap": count_rows(resolved_rows),
        "splits": {
            "smoke": count_rows(smoke),
            "train": count_rows(train),
            "val": count_rows(val),
        },
        "examples": examples[: args.write_examples],
    }
    summary_path = output_root / "uavit_resolvable_summary_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
