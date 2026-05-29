from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CoVT/uav_adapter JSONL index from DVGBench."
    )
    parser.add_argument("--hf-dataset", default="erenzhou/DVGBench")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--query-field",
        default="question_e",
        choices=("question_e", "question", "question_cn", "question_e_cn"),
        help="DVGBench text field used as the grounding query.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--image-root",
        required=True,
        help="Directory where converted/resized images will be written.",
    )
    parser.add_argument(
        "--source-image-root",
        default=None,
        help=(
            "Optional local DVGBench image directory. Use this if the loaded "
            "dataset only provides image_id and not a PIL image column."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=1344)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite images even when the target jpg already exists.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print dataset columns and the first row schema.",
    )
    return parser.parse_args()


def _short_value(value: Any) -> Any:
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {"type": type(value).__name__, "size": value.size, "mode": value.mode}
    text = repr(value)
    return text if len(text) <= 240 else text[:240] + "..."


def _build_image_map(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    mapping: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            mapping.setdefault(path.name, path)
    return mapping


def _load_image(row: dict[str, Any], image_map: dict[str, Path]):
    from PIL import Image

    value = row.get("image")
    if hasattr(value, "convert"):
        return value.convert("RGB")

    for key in ("image", "image_path", "path", "file_name", "image_id"):
        candidate = row.get(key)
        if not candidate:
            continue
        candidate_path = Path(str(candidate))
        if candidate_path.exists():
            return Image.open(candidate_path).convert("RGB")
        mapped = image_map.get(candidate_path.name)
        if mapped is not None:
            return Image.open(mapped).convert("RGB")

    raise FileNotFoundError(
        "Could not resolve image. Provide --source-image-root if the HF row only has image_id."
    )


def _resize_image(image, max_side: int) -> tuple[Any, float]:
    width, height = image.size
    if max_side <= 0 or max(width, height) <= max_side:
        return image, 1.0

    scale = max_side / float(max(width, height))
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resampling = getattr(getattr(__import__("PIL").Image, "Resampling", object), "LANCZOS", 1)
    return image.resize(new_size, resampling), scale


def _bbox_norm(bbox: Any, width: int, height: int) -> list[float] | None:
    if bbox is None:
        return None
    values = [float(v) for v in bbox]
    if len(values) < 4:
        return None

    x1, y1, x2, y2 = values[:4]
    if x2 <= x1 or y2 <= y1:
        return None

    x1 = min(max(x1 / width, 0.0), 1.0)
    y1 = min(max(y1 / height, 0.0), 1.0)
    x2 = min(max(x2 / width, 0.0), 1.0)
    y2 = min(max(y2 / height, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _safe_id(value: Any, fallback: str) -> str:
    text = str(value if value is not None else fallback)
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep).strip("_") or fallback


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.hf_dataset, split=args.split)
    print(dataset)
    print("columns:", dataset.column_names)
    if len(dataset):
        first = dataset[0]
        print("first row:")
        print(json.dumps({k: _short_value(v) for k, v in first.items()}, ensure_ascii=False, indent=2))

    if args.inspect_only:
        return

    output = Path(args.output).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    image_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_root = Path(args.source_image_root).expanduser().resolve() if args.source_image_root else None
    image_map = _build_image_map(source_root)
    counts: Counter[str] = Counter()

    with output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            if args.limit is not None and counts["written"] >= args.limit:
                break

            query = str(row.get(args.query_field) or "").strip()
            if not query:
                counts["missing_query"] += 1
                continue

            try:
                image = _load_image(row, image_map)
            except Exception as exc:
                counts["missing_image"] += 1
                print(f"[dvgbench] missing image row={idx}: {exc}", flush=True)
                continue

            original_w, original_h = image.size
            bbox_norm = _bbox_norm(row.get("bbox"), original_w, original_h)
            if bbox_norm is None:
                counts["bad_bbox"] += 1
                continue

            image, scale = _resize_image(image, args.max_side)
            question_id = row.get("question_id", idx)
            image_id = row.get("image_id", f"{idx:06d}.jpg")
            stem = Path(str(image_id)).stem
            sample_id = f"dvgbench_{args.split}_{_safe_id(question_id, str(idx))}_{args.query_field}"
            image_path = image_root / f"{_safe_id(stem, str(idx))}.jpg"
            if args.overwrite_images or not image_path.exists():
                image.save(image_path, quality=args.jpeg_quality, optimize=True)

            out = {
                "sample_id": sample_id,
                "image": str(image_path),
                "query": query,
                "bbox": row.get("bbox"),
                "bbox_norm": bbox_norm,
                "image_id": image_id,
                "question_id": question_id,
                "question": row.get("question"),
                "question_e": row.get("question_e"),
                "question_cn": row.get("question_cn"),
                "question_e_cn": row.get("question_e_cn"),
                "dataset": row.get("dataset"),
                "category": row.get("class") or "dvgbench_object",
                "category_id": 0,
                "split": row.get("split") or args.split,
                "source": "DVGBench",
                "query_rule": f"dvgbench_{args.query_field}",
                "query_version": "dvgbench_refpg_v1",
                "original_size": [original_w, original_h],
                "resized_size": list(image.size),
                "resize_scale": scale,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts["written"] += 1
            counts[f"class_{out['category']}"] += 1
            if row.get("dataset"):
                counts[f"dataset_{row['dataset']}"] += 1

    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
