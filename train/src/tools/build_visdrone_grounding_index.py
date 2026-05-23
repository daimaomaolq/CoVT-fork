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

SPATIAL_TEMPLATES = (
    "find the {scale_category} in the {region}",
    "locate the {scale_category} near the {region}",
    "find the {category} in the {region} of the image",
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
        help="Phrase template. Can be repeated. Available fields: category, scale, scale_category, region.",
    )
    parser.add_argument(
        "--query-mode",
        default="spatial",
        choices=("simple", "spatial", "mixed"),
        help="Query style. spatial adds coarse location and scale phrases.",
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


def scale_name(bbox_norm: list[float]) -> str:
    area = max(bbox_norm[2] - bbox_norm[0], 0.0) * max(bbox_norm[3] - bbox_norm[1], 0.0)
    if area < 0.01:
        return "small"
    if area < 0.08:
        return "medium"
    return "large"


def region_name(bbox_norm: list[float]) -> tuple[str, str, str]:
    cx = (bbox_norm[0] + bbox_norm[2]) * 0.5
    cy = (bbox_norm[1] + bbox_norm[3]) * 0.5

    if cx < 1.0 / 3.0:
        horizontal = "left"
    elif cx < 2.0 / 3.0:
        horizontal = "center"
    else:
        horizontal = "right"

    if cy < 1.0 / 3.0:
        vertical = "upper"
    elif cy < 2.0 / 3.0:
        vertical = "middle"
    else:
        vertical = "lower"

    if horizontal == "center" and vertical == "middle":
        region = "center"
    elif horizontal == "center":
        region = f"{vertical} center"
    elif vertical == "middle":
        region = f"{horizontal} side"
    else:
        region = f"{vertical} {horizontal}"
    return region, horizontal, vertical


def build_query(
    category: str,
    bbox_norm: list[float] | None,
    templates: tuple[str, ...],
    query_mode: str,
    row_index: int,
) -> tuple[str, dict[str, str]]:
    if query_mode == "simple" or bbox_norm is None:
        template = templates[row_index % len(templates)]
        return template.format(
            category=category,
            scale="",
            scale_category=category,
            region="image",
        ), {"query_type": "simple"}

    scale = scale_name(bbox_norm)
    region, horizontal, vertical = region_name(bbox_norm)
    scale_category = f"{scale} {category}" if scale != "medium" else category
    use_spatial = query_mode == "spatial" or row_index % 2 == 0
    if use_spatial:
        template = templates[row_index % len(templates)]
        query_type = "spatial"
    else:
        template = DEFAULT_TEMPLATES[row_index % len(DEFAULT_TEMPLATES)]
        query_type = "simple"

    return template.format(
        category=category,
        scale=scale,
        scale_category=scale_category,
        region=region,
        horizontal=horizontal,
        vertical=vertical,
    ), {
        "query_type": query_type,
        "scale": scale,
        "region": region,
        "horizontal_region": horizontal,
        "vertical_region": vertical,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.visdrone_root).expanduser().resolve()
    image_dir = root / args.image_dir_name
    annotation_dir = root / args.annotation_dir_name
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_categories = parse_categories(args.categories)
    templates = tuple(args.template or (SPATIAL_TEMPLATES if args.query_mode != "simple" else DEFAULT_TEMPLATES))
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
                bbox_norm = None
                if size is not None:
                    width, height = size
                    bbox_norm = [x1 / width, y1 / height, x2 / width, y2 / height]
                query, query_metadata = build_query(
                    category=category,
                    bbox_norm=bbox_norm,
                    templates=templates,
                    query_mode=args.query_mode,
                    row_index=per_image_count + category_id,
                )
                sample_id = f"visdrone_{annotation_path.stem}_{box_idx:05d}_{query_metadata['query_type']}"
                row = {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "query": query,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_format": "xyxy_abs",
                    "category": category,
                    "category_id": category_id,
                    "source": args.source,
                    "annotation": str(annotation_path),
                    "object_score": score,
                    **query_metadata,
                }
                if bbox_norm is not None:
                    row["image_size"] = [width, height]
                    row["bbox_norm"] = bbox_norm

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
