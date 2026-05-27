from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


INDEX_VERSION = "oneperimage_v1"

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

DEFAULT_KEEP = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
PRIMARY_VEHICLES = {4, 5, 6, 9}
SECONDARY_VEHICLES = {3, 7, 8, 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build VisDrone-OnePerImage-RefPG: one selected target per image with "
            "rule-based referring phrases and difficulty tags."
        )
    )
    parser.add_argument("--visdrone-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--annotation-dir-name", default="annotations")
    parser.add_argument("--categories", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--min-object-score", type=int, default=1)
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument("--source", default="VisDrone-OnePerImage-RefPG")
    return parser.parse_args()


def parse_categories(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def read_annotation(path: Path) -> list[tuple[int, float, float, float, float, int, int]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
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
            rows.append((line_no - 1, x, y, w, h, score, category_id))
    return rows


def scale_name(area_norm: float) -> str:
    if area_norm < 0.01:
        return "small"
    if area_norm < 0.08:
        return "medium"
    return "large"


def region_name(bbox_norm: list[float]) -> tuple[str, str, str]:
    center_x = (bbox_norm[0] + bbox_norm[2]) * 0.5
    center_y = (bbox_norm[1] + bbox_norm[3]) * 0.5

    if center_x < 1.0 / 3.0:
        horizontal = "left"
    elif center_x < 2.0 / 3.0:
        horizontal = "center"
    else:
        horizontal = "right"

    if center_y < 1.0 / 3.0:
        vertical = "upper"
    elif center_y < 2.0 / 3.0:
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


def ordinal(index: int) -> str:
    names = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    return names.get(index, f"{index}th")


def closest_edge(target: dict[str, object]) -> tuple[str, float]:
    bbox_norm = target["bbox_norm"]
    distances = {
        "left edge": bbox_norm[0],
        "right edge": 1.0 - bbox_norm[2],
        "top edge": bbox_norm[1],
        "bottom edge": 1.0 - bbox_norm[3],
    }
    return min(distances.items(), key=lambda item: item[1])


def object_score(target: dict[str, object], objects: list[dict[str, object]]) -> float:
    same_category = [obj for obj in objects if obj["category_id"] == target["category_id"]]
    same_by_area = sorted(same_category, key=lambda obj: (-obj["area_norm"], obj["box_idx"]))
    area_rank = same_by_area.index(target) + 1
    second_area = same_by_area[1]["area_norm"] if len(same_by_area) > 1 else 0.0
    area_ratio = target["area_norm"] / max(second_area, 1e-6)
    center_dist = math.sqrt((target["center_x"] - 0.5) ** 2 + (target["center_y"] - 0.5) ** 2)

    score = 0.0
    score += min(target["area_norm"] / 0.04, 1.0) * 0.40
    score += max(0.0, 1.0 - center_dist / 0.707) * 0.20
    if target["category_id"] in PRIMARY_VEHICLES:
        score += 0.20
    elif target["category_id"] in SECONDARY_VEHICLES:
        score += 0.10
    if len(same_category) == 1:
        score += 0.15
    if area_rank == 1 and area_ratio >= 1.3:
        score += 0.15
    if target["area_norm"] < 0.003:
        score -= 0.10
    return score


def make_query(target: dict[str, object], objects: list[dict[str, object]]) -> tuple[str, str, str]:
    same_category = [obj for obj in objects if obj["category_id"] == target["category_id"]]
    same_by_area = sorted(same_category, key=lambda obj: (-obj["area_norm"], obj["box_idx"]))
    area_rank = same_by_area.index(target) + 1
    second_area = same_by_area[1]["area_norm"] if len(same_by_area) > 1 else 0.0
    area_ratio = target["area_norm"] / max(second_area, 1e-6)

    same_by_center = sorted(
        same_category,
        key=lambda obj: ((obj["center_x"] - 0.5) ** 2 + (obj["center_y"] - 0.5) ** 2, obj["box_idx"]),
    )
    center_rank = same_by_center.index(target) + 1

    category = str(target["category"])
    region = str(target["region"])
    scale = str(target["scale"])
    scale_category = f"{scale} {category}" if scale != "medium" else category

    if len(same_category) == 1:
        difficulty = "easy" if target["area_norm"] >= 0.005 else "medium"
        return f"find the only {category}", "only_category", difficulty

    if area_rank == 1 and area_ratio >= 1.5:
        difficulty = "easy" if target["area_norm"] >= 0.005 else "medium"
        return f"find the largest {category} in the {region} region", "largest_category_region", difficulty

    if area_rank == 1 and area_ratio >= 1.15:
        return f"find the prominent {scale_category} in the {region} region", "prominent_category_region", "medium"

    if center_rank == 1:
        return f"find the {category} closest to the image center", "closest_center_category", "medium"

    edge_name, _ = closest_edge(target)
    same_by_edge = sorted(same_category, key=lambda obj: closest_edge(obj)[1])
    if same_by_edge[0] is target:
        return f"find the {category} closest to the {edge_name}", f"closest_{edge_name.replace(' ', '_')}", "medium"

    return (
        f"find the {ordinal(area_rank)} largest {category} in the {region} region",
        "area_ordinal_fallback",
        "hard_fallback",
    )


def make_object(
    box_idx: int,
    raw_box: tuple[float, float, float, float, int, int],
    width: int,
    height: int,
) -> dict[str, object] | None:
    x, y, w, h, score, category_id = raw_box
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    bbox_norm = [x1 / width, y1 / height, x2 / width, y2 / height]
    area_norm = max(bbox_norm[2] - bbox_norm[0], 0.0) * max(bbox_norm[3] - bbox_norm[1], 0.0)
    region, horizontal, vertical = region_name(bbox_norm)
    return {
        "box_idx": box_idx,
        "bbox": [x1, y1, x2, y2],
        "bbox_norm": bbox_norm,
        "area_norm": area_norm,
        "category_id": category_id,
        "category": VISDRONE_CATEGORIES.get(category_id, f"category {category_id}"),
        "region": region,
        "horizontal_region": horizontal,
        "vertical_region": vertical,
        "scale": scale_name(area_norm),
        "center_x": (bbox_norm[0] + bbox_norm[2]) * 0.5,
        "center_y": (bbox_norm[1] + bbox_norm[3]) * 0.5,
        "score": score,
    }


def build_index(
    visdrone_root: Path,
    output: Path,
    image_dir_name: str,
    annotation_dir_name: str,
    categories: set[int],
    min_object_score: int,
    min_area: float,
    source: str,
) -> None:
    image_dir = visdrone_root / image_dir_name
    annotation_dir = visdrone_root / annotation_dir_name
    rows = []
    skipped = Counter()
    difficulty = Counter()
    rules = Counter()
    selected_categories = Counter()

    for annotation_path in sorted(annotation_dir.glob("*.txt")):
        image_path = find_image(image_dir, annotation_path.stem)
        if image_path is None:
            skipped["missing_image"] += 1
            continue
        width, height = image_size(image_path)
        objects = []
        for box_idx, x, y, w, h, score, category_id in read_annotation(annotation_path):
            if score < min_object_score or category_id not in categories or w * h < min_area:
                continue
            obj = make_object(box_idx, (x, y, w, h, score, category_id), width, height)
            if obj is not None:
                objects.append(obj)

        if not objects:
            skipped["no_valid_object"] += 1
            continue

        target = max(objects, key=lambda obj: (object_score(obj, objects), obj["area_norm"], -obj["box_idx"]))
        easy_score = object_score(target, objects)
        query, query_rule, diff = make_query(target, objects)
        sample_id = (
            f"visdrone_oneperimage_{annotation_path.stem}_{target['box_idx']:05d}_"
            f"{query_rule}_{diff}_{INDEX_VERSION}"
        )

        row = {
            "sample_id": sample_id,
            "image": str(image_path),
            "query": query,
            "bbox": target["bbox"],
            "bbox_norm": target["bbox_norm"],
            "category": target["category"],
            "category_id": target["category_id"],
            "region": target["region"],
            "horizontal_region": target["horizontal_region"],
            "vertical_region": target["vertical_region"],
            "scale": target["scale"],
            "query_rule": query_rule,
            "difficulty": diff,
            "easy_score": easy_score,
            "object_count": len(objects),
            "source": source,
            "query_version": INDEX_VERSION,
        }
        rows.append(row)
        difficulty[diff] += 1
        rules[query_rule] += 1
        selected_categories[str(target["category"])] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "root": str(visdrone_root),
        "output": str(output),
        "samples": len(rows),
        "skipped": dict(skipped),
        "difficulty": dict(difficulty),
        "query_rules": dict(rules),
        "categories": dict(selected_categories),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    build_index(
        visdrone_root=Path(args.visdrone_root),
        output=Path(args.output),
        image_dir_name=args.image_dir_name,
        annotation_dir_name=args.annotation_dir_name,
        categories=parse_categories(args.categories),
        min_object_score=args.min_object_score,
        min_area=args.min_area,
        source=args.source,
    )


if __name__ == "__main__":
    main()
