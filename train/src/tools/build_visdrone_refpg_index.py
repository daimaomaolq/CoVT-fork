from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


INDEX_VERSION = "refpg_v2"

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


@dataclass(frozen=True)
class ObjectRecord:
    box_idx: int
    x: float
    y: float
    w: float
    h: float
    score: int
    category_id: int
    category: str
    bbox: tuple[float, float, float, float]
    bbox_norm: tuple[float, float, float, float]
    area_norm: float
    scale: str
    region: str
    horizontal_region: str
    vertical_region: str
    center_x: float
    center_y: float


@dataclass(frozen=True)
class QueryCandidate:
    text: str
    rule: str
    matcher: Callable[[ObjectRecord], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build VisDrone-RefPG, a single-target UAV phrase-grounding index "
            "with automatically verified unambiguous referring queries."
        )
    )
    parser.add_argument("--visdrone-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--annotation-dir-name", default="annotations")
    parser.add_argument("--categories", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--min-area", type=float, default=4.0, help="Minimum absolute bbox area in pixels.")
    parser.add_argument(
        "--min-norm-area",
        type=float,
        default=0.0,
        help="Minimum normalized bbox area. Use 0 to keep all objects passing --min-area.",
    )
    parser.add_argument("--min-object-score", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-per-image", type=int, default=None)
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--source", default="VisDrone-RefPG")
    parser.add_argument(
        "--query-style",
        default="natural",
        choices=("natural", "structured"),
        help="natural writes readable referring phrases; structured writes compact slot tokens for lightweight adapters.",
    )
    parser.add_argument(
        "--prefer-vehicles",
        action="store_true",
        help="Try vehicle samples first when multiple objects are available in one image.",
    )
    return parser.parse_args()


def parse_categories(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def read_annotation(path: Path) -> Iterable[tuple[float, float, float, float, int, int]]:
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


def scale_name(area_norm: float) -> str:
    if area_norm < 0.01:
        return "small"
    if area_norm < 0.08:
        return "medium"
    return "large"


def region_name(bbox_norm: tuple[float, float, float, float]) -> tuple[str, str, str]:
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


def make_object(
    box_idx: int,
    raw: tuple[float, float, float, float, int, int],
    image_width: int,
    image_height: int,
) -> ObjectRecord:
    x, y, w, h, score, category_id = raw
    x1, y1, x2, y2 = x, y, x + w, y + h
    bbox_norm = (
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
    )
    area_norm = max(bbox_norm[2] - bbox_norm[0], 0.0) * max(bbox_norm[3] - bbox_norm[1], 0.0)
    region, horizontal, vertical = region_name(bbox_norm)
    return ObjectRecord(
        box_idx=box_idx,
        x=x,
        y=y,
        w=w,
        h=h,
        score=score,
        category_id=category_id,
        category=VISDRONE_CATEGORIES.get(category_id, f"category {category_id}"),
        bbox=(x1, y1, x2, y2),
        bbox_norm=bbox_norm,
        area_norm=area_norm,
        scale=scale_name(area_norm),
        region=region,
        horizontal_region=horizontal,
        vertical_region=vertical,
        center_x=(bbox_norm[0] + bbox_norm[2]) * 0.5,
        center_y=(bbox_norm[1] + bbox_norm[3]) * 0.5,
    )


def ordinal(index: int) -> str:
    names = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    return names.get(index, f"{index}th")


def rank_in_group(target: ObjectRecord, group: list[ObjectRecord], key: Callable[[ObjectRecord], float]) -> int:
    ordered = sorted(group, key=lambda obj: (key(obj), obj.box_idx))
    return ordered.index(target) + 1


def closest_edge(target: ObjectRecord) -> tuple[str, float]:
    distances = {
        "left edge": target.bbox_norm[0],
        "right edge": 1.0 - target.bbox_norm[2],
        "top edge": target.bbox_norm[1],
        "bottom edge": 1.0 - target.bbox_norm[3],
    }
    return min(distances.items(), key=lambda item: item[1])


def is_vehicle(category: str) -> bool:
    return category in {"car", "van", "truck", "bus", "motor", "bicycle", "tricycle", "awning tricycle"}


def query_text(
    style: str,
    natural: str,
    target: ObjectRecord,
    relation: str,
) -> str:
    if style == "natural":
        return natural
    slots = [
        f"category_{target.category.replace(' ', '_')}",
        f"region_{target.region.replace(' ', '_')}",
        f"scale_{target.scale}",
        relation,
    ]
    return " ".join(slots)


def candidate_queries(target: ObjectRecord, objects: list[ObjectRecord], query_style: str) -> list[QueryCandidate]:
    same_category = [obj for obj in objects if obj.category_id == target.category_id]
    same_region_category = [
        obj for obj in same_category if obj.region == target.region
    ]
    same_region_scale_category = [
        obj for obj in same_region_category if obj.scale == target.scale
    ]

    category = target.category
    scale_category = f"{target.scale} {category}" if target.scale != "medium" else category
    region = target.region
    edge_name, _ = closest_edge(target)

    queries: list[QueryCandidate] = [
        QueryCandidate(
            text=query_text(
                query_style,
                f"find the {scale_category} in the {region} region",
                target,
                "relation_region_scale",
            ),
            rule="category+scale+region",
            matcher=lambda obj, t=target: (
                obj.category_id == t.category_id and obj.scale == t.scale and obj.region == t.region
            ),
        ),
        QueryCandidate(
            text=query_text(
                query_style,
                f"find the {category} closest to the {edge_name}",
                target,
                f"closest_{edge_name.replace(' ', '_')}",
            ),
            rule=f"category+closest_{edge_name.replace(' ', '_')}",
            matcher=lambda obj, t=target, edge=edge_name: obj.category_id == t.category_id
            and obj.box_idx == closest_by_edge(
                [candidate for candidate in objects if candidate.category_id == t.category_id],
                edge,
            ).box_idx,
        ),
    ]

    if same_region_category:
        left_rank = rank_in_group(target, same_region_category, lambda obj: obj.center_x)
        right_rank = rank_in_group(target, same_region_category, lambda obj: -obj.center_x)
        top_rank = rank_in_group(target, same_region_category, lambda obj: obj.center_y)
        bottom_rank = rank_in_group(target, same_region_category, lambda obj: -obj.center_y)
        area_rank = rank_in_group(target, same_region_category, lambda obj: -obj.area_norm)

        queries.extend(
            [
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(left_rank)} {category} from the left in the {region} region",
                        target,
                        f"left_rank_{left_rank}",
                    ),
                    rule="category+region+left_ordinal",
                    matcher=lambda obj, t=target, rank=left_rank: obj.category_id == t.category_id
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [candidate for candidate in objects if candidate.category_id == t.category_id and candidate.region == t.region],
                        lambda candidate: candidate.center_x,
                    )
                    == rank,
                ),
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(right_rank)} {category} from the right in the {region} region",
                        target,
                        f"right_rank_{right_rank}",
                    ),
                    rule="category+region+right_ordinal",
                    matcher=lambda obj, t=target, rank=right_rank: obj.category_id == t.category_id
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [candidate for candidate in objects if candidate.category_id == t.category_id and candidate.region == t.region],
                        lambda candidate: -candidate.center_x,
                    )
                    == rank,
                ),
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(top_rank)} {category} from the top in the {region} region",
                        target,
                        f"top_rank_{top_rank}",
                    ),
                    rule="category+region+top_ordinal",
                    matcher=lambda obj, t=target, rank=top_rank: obj.category_id == t.category_id
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [candidate for candidate in objects if candidate.category_id == t.category_id and candidate.region == t.region],
                        lambda candidate: candidate.center_y,
                    )
                    == rank,
                ),
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(bottom_rank)} {category} from the bottom in the {region} region",
                        target,
                        f"bottom_rank_{bottom_rank}",
                    ),
                    rule="category+region+bottom_ordinal",
                    matcher=lambda obj, t=target, rank=bottom_rank: obj.category_id == t.category_id
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [candidate for candidate in objects if candidate.category_id == t.category_id and candidate.region == t.region],
                        lambda candidate: -candidate.center_y,
                    )
                    == rank,
                ),
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(area_rank)} largest {category} in the {region} region",
                        target,
                        f"area_rank_{area_rank}",
                    ),
                    rule="category+region+area_ordinal",
                    matcher=lambda obj, t=target, rank=area_rank: obj.category_id == t.category_id
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [candidate for candidate in objects if candidate.category_id == t.category_id and candidate.region == t.region],
                        lambda candidate: -candidate.area_norm,
                    )
                    == rank,
                ),
            ]
        )

    if same_region_scale_category:
        left_rank = rank_in_group(target, same_region_scale_category, lambda obj: obj.center_x)
        bottom_rank = rank_in_group(target, same_region_scale_category, lambda obj: -obj.center_y)
        queries.extend(
            [
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(left_rank)} {scale_category} from the left in the {region} region",
                        target,
                        f"left_rank_{left_rank}",
                    ),
                    rule="category+scale+region+left_ordinal",
                    matcher=lambda obj, t=target, rank=left_rank: obj.category_id == t.category_id
                    and obj.scale == t.scale
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [
                            candidate
                            for candidate in objects
                            if candidate.category_id == t.category_id
                            and candidate.scale == t.scale
                            and candidate.region == t.region
                        ],
                        lambda candidate: candidate.center_x,
                    )
                    == rank,
                ),
                QueryCandidate(
                    text=query_text(
                        query_style,
                        f"find the {ordinal(bottom_rank)} {scale_category} from the bottom in the {region} region",
                        target,
                        f"bottom_rank_{bottom_rank}",
                    ),
                    rule="category+scale+region+bottom_ordinal",
                    matcher=lambda obj, t=target, rank=bottom_rank: obj.category_id == t.category_id
                    and obj.scale == t.scale
                    and obj.region == t.region
                    and rank_in_group(
                        obj,
                        [
                            candidate
                            for candidate in objects
                            if candidate.category_id == t.category_id
                            and candidate.scale == t.scale
                            and candidate.region == t.region
                        ],
                        lambda candidate: -candidate.center_y,
                    )
                    == rank,
                ),
            ]
        )

    return queries


def closest_by_edge(objects: list[ObjectRecord], edge_name: str) -> ObjectRecord:
    if edge_name == "left edge":
        return min(objects, key=lambda obj: (obj.bbox_norm[0], obj.box_idx))
    if edge_name == "right edge":
        return min(objects, key=lambda obj: (1.0 - obj.bbox_norm[2], obj.box_idx))
    if edge_name == "top edge":
        return min(objects, key=lambda obj: (obj.bbox_norm[1], obj.box_idx))
    if edge_name == "bottom edge":
        return min(objects, key=lambda obj: (1.0 - obj.bbox_norm[3], obj.box_idx))
    raise ValueError(edge_name)


def choose_unique_query(
    target: ObjectRecord,
    objects: list[ObjectRecord],
    query_style: str,
) -> tuple[QueryCandidate, list[ObjectRecord]] | None:
    for query in candidate_queries(target, objects, query_style):
        matches = [obj for obj in objects if query.matcher(obj)]
        if len(matches) == 1 and matches[0].box_idx == target.box_idx:
            return query, matches
    return None


def object_sort_key(obj: ObjectRecord, prefer_vehicles: bool) -> tuple[int, float, int]:
    vehicle_priority = 0 if prefer_vehicles and is_vehicle(obj.category) else 1
    return vehicle_priority, -obj.area_norm, obj.box_idx


def main() -> None:
    args = parse_args()
    root = Path(args.visdrone_root).expanduser().resolve()
    image_dir = root / args.image_dir_name
    annotation_dir = root / args.annotation_dir_name
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_categories = parse_categories(args.categories)
    total_images = 0
    total_rows = 0
    skipped_no_size = 0
    skipped_ambiguous = 0
    skipped_limits = 0
    per_category_total: Counter[str] = Counter()
    per_rule_total: Counter[str] = Counter()
    per_image_total: dict[str, int] = defaultdict(int)

    with output_path.open("w", encoding="utf-8") as out:
        for annotation_path in sorted(annotation_dir.glob("*.txt")):
            total_images += 1
            image_path = find_image(image_dir, annotation_path.stem)
            size = image_size(image_path)
            if size is None:
                skipped_no_size += 1
                continue
            image_width, image_height = size

            objects: list[ObjectRecord] = []
            for box_idx, raw in enumerate(read_annotation(annotation_path)):
                x, y, w, h, score, category_id = raw
                if category_id not in keep_categories:
                    continue
                if score < args.min_object_score:
                    continue
                if w * h < args.min_area:
                    continue
                obj = make_object(box_idx, raw, image_width, image_height)
                if obj.area_norm < args.min_norm_area:
                    continue
                objects.append(obj)

            image_rows = 0
            for obj in sorted(objects, key=lambda item: object_sort_key(item, args.prefer_vehicles)):
                if args.max_samples is not None and total_rows >= args.max_samples:
                    break
                if args.max_per_image is not None and image_rows >= args.max_per_image:
                    skipped_limits += 1
                    continue
                if args.max_per_category is not None and per_category_total[obj.category] >= args.max_per_category:
                    skipped_limits += 1
                    continue

                chosen = choose_unique_query(obj, objects, args.query_style)
                if chosen is None:
                    skipped_ambiguous += 1
                    continue
                query, matches = chosen
                sample_id = (
                    f"visdrone_refpg_{args.query_style}_{annotation_path.stem}_"
                    f"{obj.box_idx:05d}_{query.rule}_{INDEX_VERSION}"
                )
                row = {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "query": query.text,
                    "bbox": list(obj.bbox),
                    "bbox_format": "xyxy_abs",
                    "bbox_norm": list(obj.bbox_norm),
                    "image_size": [image_width, image_height],
                    "category": obj.category,
                    "category_id": obj.category_id,
                    "source": args.source,
                    "annotation": str(annotation_path),
                    "object_score": obj.score,
                    "query_version": INDEX_VERSION,
                    "query_style": args.query_style,
                    "query_type": "referring",
                    "query_rule": query.rule,
                    "matched_candidates": len(matches),
                    "image_object_count": len(objects),
                    "same_category_count": sum(1 for candidate in objects if candidate.category_id == obj.category_id),
                    "same_region_category_count": sum(
                        1
                        for candidate in objects
                        if candidate.category_id == obj.category_id and candidate.region == obj.region
                    ),
                    "scale": obj.scale,
                    "region": obj.region,
                    "horizontal_region": obj.horizontal_region,
                    "vertical_region": obj.vertical_region,
                    "area_norm": obj.area_norm,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
                image_rows += 1
                per_image_total[str(image_path)] += 1
                per_category_total[obj.category] += 1
                per_rule_total[query.rule] += 1

            if args.max_samples is not None and total_rows >= args.max_samples:
                break

    print(
        json.dumps(
            {
                "visdrone_root": str(root),
                "output": str(output_path),
                "images_seen": total_images,
                "samples": total_rows,
                "images_with_samples": len(per_image_total),
                "skipped_no_size": skipped_no_size,
                "skipped_ambiguous": skipped_ambiguous,
                "skipped_limits": skipped_limits,
                "categories": dict(sorted(per_category_total.items())),
                "query_rules": dict(sorted(per_rule_total.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
