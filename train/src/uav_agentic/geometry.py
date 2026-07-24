from __future__ import annotations

import math

from .schema import Box


def clamp_box(box: Box) -> Box:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def box_area(box: Box | None) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = clamp_box(box)
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = clamp_box(box)
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_iou(first: Box | None, second: Box | None) -> float:
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = clamp_box(first)
    bx1, by1, bx2, by2 = clamp_box(second)
    intersection = (
        max(min(ax2, bx2) - max(ax1, bx1), 0.0)
        * max(min(ay2, by2) - max(ay1, by1), 0.0)
    )
    first_area = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    second_area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(first: Box, second: Box) -> float:
    ax, ay = box_center(first)
    bx, by = box_center(second)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) / math.sqrt(2)


def box_plausibility(box: Box | None) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = clamp_box(box)
    width, height = x2 - x1, y2 - y1
    if width <= 1e-4 or height <= 1e-4:
        return 0.0
    aspect = max(width / height, height / width)
    aspect_score = math.exp(-max(0.0, math.log(aspect) - math.log(8.0)))
    area_score = 1.0 if 0.0001 <= width * height <= 0.75 else 0.4
    contacts = sum((x1 <= 1e-4, y1 <= 1e-4, x2 >= 0.9999, y2 >= 0.9999))
    boundary_score = 1.0 - 0.1 * contacts
    return max(0.0, min(1.0, 0.5 * aspect_score + 0.3 * area_score + 0.2 * boundary_score))


def expand_box(box: Box, scale: float) -> Box:
    center_x, center_y = box_center(box)
    x1, y1, x2, y2 = clamp_box(box)
    width, height = (x2 - x1) * scale, (y2 - y1) * scale
    return clamp_box([
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    ])


def union_box(boxes: list[Box]) -> Box:
    if not boxes:
        raise ValueError("union_box requires at least one box")
    clamped = [clamp_box(box) for box in boxes]
    return [
        min(box[0] for box in clamped),
        min(box[1] for box in clamped),
        max(box[2] for box in clamped),
        max(box[3] for box in clamped),
    ]


def map_from_crop(local_box: Box | None, crop_region: Box) -> Box | None:
    if local_box is None:
        return None
    crop_x1, crop_y1, crop_x2, crop_y2 = clamp_box(crop_region)
    crop_width, crop_height = crop_x2 - crop_x1, crop_y2 - crop_y1
    x1, y1, x2, y2 = clamp_box(local_box)
    return clamp_box([
        crop_x1 + x1 * crop_width,
        crop_y1 + y1 * crop_height,
        crop_x1 + x2 * crop_width,
        crop_y1 + y2 * crop_height,
    ])


def contains_point(box: Box, point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = clamp_box(box)
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def bottom_center(box: Box) -> tuple[float, float]:
    x1, _, x2, y2 = clamp_box(box)
    return (x1 + x2) / 2, y2


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
