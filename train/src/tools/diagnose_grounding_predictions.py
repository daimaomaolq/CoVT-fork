from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and visualize grounding predictions against a JSONL index."
    )
    parser.add_argument("--index", required=True, help="JSONL index with sample_id, image, bbox_norm.")
    parser.add_argument("--predictions", required=True, help="JSONL predictions from eval_multitask_grounding.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered overlay images.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--mode", choices=("worst", "best", "first", "random"), default="worst")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--bbox-key", default="bbox_norm")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clamp_box(box: list[float]) -> list[float]:
    values = [float(value) for value in box[:4]]
    x1, y1, x2, y2 = values
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = clamp_box(box)
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = clamp_box(box)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def denorm_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = clamp_box(box)
    return [
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    ]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(max(round((len(sorted_values) - 1) * q), 0), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row.get("iou", 0.0)) for row in rows]
    pred_areas = [box_area(row["pred_bbox"]) for row in rows]
    gt_areas = [box_area(row["gt_bbox"]) for row in rows]
    pred_centers = [box_center(row["pred_bbox"]) for row in rows]
    gt_centers = [box_center(row["gt_bbox"]) for row in rows]
    center_errors = [
        ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
        for (px, py), (gx, gy) in zip(pred_centers, gt_centers)
    ]
    return {
        "samples": len(rows),
        "mIoU": mean(ious) if ious else 0.0,
        "Acc@0.5": sum(1 for value in ious if value >= 0.5) / max(len(ious), 1),
        "IoU_p50": percentile(ious, 0.50),
        "IoU_p90": percentile(ious, 0.90),
        "pred_area_mean": mean(pred_areas) if pred_areas else 0.0,
        "gt_area_mean": mean(gt_areas) if gt_areas else 0.0,
        "pred_center_mean": [
            mean([center[0] for center in pred_centers]) if pred_centers else 0.0,
            mean([center[1] for center in pred_centers]) if pred_centers else 0.0,
        ],
        "gt_center_mean": [
            mean([center[0] for center in gt_centers]) if gt_centers else 0.0,
            mean([center[1] for center in gt_centers]) if gt_centers else 0.0,
        ],
        "center_error_mean": mean(center_errors) if center_errors else 0.0,
    }


def select_rows(rows: list[dict[str, Any]], mode: str, limit: int, seed: int) -> list[dict[str, Any]]:
    if mode == "best":
        return sorted(rows, key=lambda row: float(row.get("iou", 0.0)), reverse=True)[:limit]
    if mode == "first":
        return rows[:limit]
    if mode == "random":
        selected = list(rows)
        random.Random(seed).shuffle(selected)
        return selected[:limit]
    return sorted(rows, key=lambda row: float(row.get("iou", 0.0)))[:limit]


def draw_overlay(row: dict[str, Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(row["image"]).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    gt = denorm_box(row["gt_bbox"], width, height)
    pred = denorm_box(row["pred_bbox"], width, height)
    line_width = max(2, round(max(width, height) / 400))
    draw.rectangle(gt, outline=(0, 220, 0), width=line_width)
    draw.rectangle(pred, outline=(255, 40, 40), width=line_width)
    label = f"IoU={float(row.get('iou', 0.0)):.3f} score={float(row.get('score', 0.0)):.3f}"
    draw.rectangle([0, 0, min(width, 780), 32], fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    index_rows = read_jsonl(args.index)
    pred_rows = read_jsonl(args.predictions)
    index_by_id = {str(row["sample_id"]): row for row in index_rows}
    joined = []
    missing_meta = 0
    for pred in pred_rows:
        sample_id = str(pred["sample_id"])
        meta = index_by_id.get(sample_id)
        if meta is None:
            missing_meta += 1
            continue
        joined.append(
            {
                **pred,
                "image": meta["image"],
                "query": meta.get("query", ""),
                "gt_bbox": meta[args.bbox_key],
                "pred_bbox": pred["bbox"],
                "category": str(meta.get("category") or pred.get("class") or "unknown").strip(),
            }
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_rows(joined, args.mode, args.limit, args.seed)
    for rank, row in enumerate(selected):
        safe_id = str(row["sample_id"]).replace("/", "_").replace("\\", "_").replace(":", "_")
        draw_overlay(row, output_dir / f"{rank:03d}_{args.mode}_{safe_id}.jpg")

    summary = summarize(joined)
    summary.update(
        {
            "index_rows": len(index_rows),
            "prediction_rows": len(pred_rows),
            "joined_rows": len(joined),
            "missing_prediction_meta": missing_meta,
            "rendered": len(selected),
            "render_dir": str(output_dir),
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
