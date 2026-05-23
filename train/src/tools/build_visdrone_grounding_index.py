from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


VISDRONE_CATEGORIES = {
    0: "ignored region",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning tricycle",
    9: "bus",
    10: "motor",
    11: "other",
}

DEFAULT_TEMPLATES = (
    "find the {category}",
    "locate the {category}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a phrase-grounding JSONL index from VisDrone DET annotations."
    )
    parser.add_argument(
        "--visdrone-root",
        required=True,
        help="VisDrone split root containing images/ and annotations/ directories.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--image-dir-name",
        default="images",
        help="Image directory name under --visdrone-root.",
    )
    parser.add_argument(
        "--annotation-dir-name",
        default="annotations",
        help="Annotation directory name under --visdrone-root.",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated VisDrone category ids to keep.",
    )
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument("--max-per-image", type=int, default=None)
    parser.add_argument("--max-per-category-per-image", type=int, default=None)
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Also emit bbox_norm if PIL can read image size.",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=None,
        help="Phrase template. Can be repeated. Use {category}.",
    )
    parser.add_argument(
        "--source",
        default="VisDrone",
        help="Source tag written into every row.",
    )
    return parser.parse_args()


def parse_categories(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def read_annotation(path: Path) -> Iterable[tuple[float, float, float, float, int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no} has {len(parts)} fields, expected at least 6")
            x, y, w, h = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            score = int(float(parts[4]))
            category_id = int(float(parts[5]))
            yield x, y, w, h, score, category_id


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def find_image(image_dir: Path, stem: str) -> Path:
    for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return image_dir / f"{stem}.jpg"


def main() -> None:
    args = parse_args()
    root = Path(args.visdrone_root).expanduser().resolve()
    image_dir = root / args.image_dir_name
    annotation_dir = root / args.annotation_dir_name
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_categories = parse_categories(args.categories)
    templates = tuple(args.template or DEFAULT_TEMPLATES)
    total_images = 0
    total_rows = 0
    per_category_total: dict[str, int] = defaultdict(int)

    with output_path.open("w", encoding="utf-8") as out:
        for annotation_path in sorted(annotation_dir.glob("*.txt")):
            total_images += 1
            image_path = find_image(image_dir, annotation_path.stem)
            size = image_size(image_path) if args.normalized else None
            per_image_count = 0
            per_image_category_count: dict[int, int] = defaultdict(int)

            for box_idx, (x, y, w, h, score, category_id) in enumerate(read_annotation(annotation_path)):
                if category_id not in keep_categories:
                    continue
                if w * h < args.min_area:
                    continue
                if args.max_per_image is not None and per_image_count >= args.max_per_image:
                    break
                if (
                    args.max_per_category_per_image is not None
                    and per_image_category_count[category_id] >= args.max_per_category_per_image
                ):
                    continue

                category = VISDRONE_CATEGORIES.get(category_id, f"category {category_id}")
                x1, y1, x2, y2 = x, y, x + w, y + h
                template = templates[(per_image_count + category_id) % len(templates)]
                sample_id = f"visdrone_{annotation_path.stem}_{box_idx:05d}"
                row = {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "query": template.format(category=category),
                    "bbox": [x1, y1, x2, y2],
                    "bbox_format": "xyxy_abs",
                    "category": category,
                    "category_id": category_id,
                    "source": args.source,
                    "annotation": str(annotation_path),
                    "object_score": score,
                }
                if size is not None:
                    width, height = size
                    row["image_size"] = [width, height]
                    row["bbox_norm"] = [x1 / width, y1 / height, x2 / width, y2 / height]

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
                per_image_count += 1
                per_image_category_count[category_id] += 1
                per_category_total[category] += 1

    print(
        json.dumps(
            {
                "visdrone_root": str(root),
                "output": str(output_path),
                "images_seen": total_images,
                "samples": total_rows,
                "categories": dict(sorted(per_category_total.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
