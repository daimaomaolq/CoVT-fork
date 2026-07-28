from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DVGBench JSONL rows into CoVT/Qwen LLaVA-style SFT data "
            "for benchmark-aligned generative region grounding."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="DVGBench dvg_train/test JSONL.")
    parser.add_argument("--output", required=True, help="Output JSON array path for training.")
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root containing DVGBench images, typically .../DVGBench/images.",
    )
    parser.add_argument(
        "--query-field",
        default="question",
        choices=("question", "question_e", "question_cn", "question_e_cn"),
        help="Use question for DVGBench implicit-query protocol.",
    )
    parser.add_argument(
        "--mode",
        default="answer_only",
        choices=("answer_only", "reasoning", "i2e"),
        help=(
            "Supervision protocol. i2e maps an implicit query to the paired "
            "explicit reference and then to the bbox."
        ),
    )
    parser.add_argument(
        "--explicit-field",
        default="question_e",
        choices=("question_e", "question_e_cn"),
        help="Training-only explicit reference used by --mode i2e.",
    )
    parser.add_argument(
        "--i2e-answer-only-copy-ratio",
        type=float,
        default=0.0,
        help=(
            "For --mode i2e, add deterministic answer-only copies for this "
            "fraction of source rows. This preserves bbox generation while "
            "learning explicitization. Must be in [0,1]."
        ),
    )
    parser.add_argument(
        "--omit-oracle-fields-from-eval-index",
        action="store_true",
        help=(
            "Do not write question_e/question_e_cn into the eval index. Use "
            "this for the final implicit-query test index to make leakage "
            "structurally impossible."
        ),
    )
    parser.add_argument(
        "--image-path-mode",
        default="relative",
        choices=("relative", "absolute"),
        help="Store image paths relative to --image-folder or as absolute paths.",
    )
    parser.add_argument(
        "--image-folder",
        default=None,
        help=(
            "Base folder used to relativize image paths. Defaults to --image-root. "
            "Pass the same path to train.py --image_folder."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help=(
            "Optional fraction in [0,1) to split into train/val JSON files. "
            "When >0, --output becomes the train file and --val-output is required."
        ),
    )
    parser.add_argument("--val-output", default=None)
    parser.add_argument(
        "--write-eval-index",
        default=None,
        help="Optional JSONL eval index with image, query, bbox_norm, class metadata.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def parse_json_or_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = clean_text(value)
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import ast

        return ast.literal_eval(text)
    except Exception:
        return value


def parse_bbox(value: Any) -> list[float] | None:
    value = parse_json_or_literal(value)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_to_qwen_tokens(bbox: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    values = [
        round(max(0.0, min(1.0, x1 / width)) * 1000),
        round(max(0.0, min(1.0, y1 / height)) * 1000),
        round(max(0.0, min(1.0, x2 / width)) * 1000),
        round(max(0.0, min(1.0, y2 / height)) * 1000),
    ]
    return "{<%d><%d><%d><%d>}" % tuple(values)


def bbox_norm(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def safe_id(value: Any, fallback: str) -> str:
    text = clean_text(value) or fallback
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("_") or fallback


def resolve_image(row: dict[str, Any], image_root: Path) -> Path | None:
    image_id = clean_text(row.get("image_id"))
    dataset = clean_text(row.get("dataset")).lower()
    candidates = []
    if image_id:
        candidates.append(image_root / dataset / image_id)
        candidates.append(image_root / image_id)
        candidates.extend(image_root.rglob(Path(image_id).name))
    for path in candidates:
        if path.is_file() and not path.name.startswith("._"):
            return path.resolve()
    return None


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def prompt_for_query(query: str, mode: str) -> str:
    suffix = "Output only the bounding box in the format {<x1><y1><x2><y2>}."
    if mode == "reasoning":
        suffix = (
            "Think briefly about the referred object, then put the final bounding "
            "box in <answer>{<x1><y1><x2><y2>}</answer>."
        )
    elif mode == "i2e":
        suffix = (
            "Output the thinking process in <think> </think>, a brief explicit "
            "description of the referred object using visible category, color, "
            "size, relative position, context, or relation evidence in "
            "<explicit> </explicit>, and the final bounding box in "
            "<answer> </answer> tags. Respond exactly as:\n"
            "<think>brief visual reasoning</think>\n"
            "<explicit>brief explicit description</explicit>\n"
            "<answer>{<x1><y1><x2><y2>}</answer>"
        )
    return f"<image>\nLocate the region described by: {query}\n{suffix}"


def answer_for_bbox(
    bbox_text: str,
    query: str,
    mode: str,
    explicit_reference: str | None = None,
) -> str:
    if mode == "answer_only":
        return bbox_text
    if mode == "i2e":
        explicit_reference = clean_text(explicit_reference)
        if not explicit_reference:
            raise ValueError("I2E supervision requires a non-empty explicit reference.")
        explicit_reference = (
            explicit_reference.replace("<explicit>", "")
            .replace("</explicit>", "")
            .replace("<answer>", "")
            .replace("</answer>", "")
            .strip()
        )
        rationale = (
            "I interpret the implicit request and identify the same target from "
            f"visible scene evidence: {explicit_reference}."
        )
        return (
            f"<think>{rationale}</think>\n"
            f"<explicit>{explicit_reference}</explicit>\n"
            f"<answer>{bbox_text}</answer>"
        )
    return (
        "<think>The query refers to a UAV scene region. I identify the target by "
        "its category, spatial relation, and surrounding context.</think>"
        f"<answer>{bbox_text}</answer>"
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_no} is not valid JSONL") from err
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_eval_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.i2e_answer_only_copy_ratio <= 1.0):
        raise ValueError("--i2e-answer-only-copy-ratio must be in [0,1].")
    if args.mode != "i2e" and args.i2e_answer_only_copy_ratio:
        raise ValueError("--i2e-answer-only-copy-ratio is only valid with --mode i2e.")

    image_root = Path(args.image_root).expanduser().resolve()
    image_folder = Path(args.image_folder).expanduser().resolve() if args.image_folder else image_root
    raw_rows = load_rows(Path(args.input_jsonl).expanduser().resolve())
    if args.shuffle:
        random.Random(args.seed).shuffle(raw_rows)
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]

    sft_groups: list[list[dict[str, Any]]] = []
    eval_rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "input": len(raw_rows),
        "written": 0,
        "source_rows_written": 0,
        "answer_only_copies": 0,
        "missing_image": 0,
        "missing_query": 0,
        "missing_explicit": 0,
        "bad_bbox": 0,
    }
    copy_rng = random.Random(args.seed + 1)

    for idx, row in enumerate(raw_rows):
        query = clean_text(row.get(args.query_field))
        if not query:
            stats["missing_query"] += 1
            continue
        explicit_reference = clean_text(row.get(args.explicit_field))
        if args.mode == "i2e" and not explicit_reference:
            stats["missing_explicit"] += 1
            continue
        image_path = resolve_image(row, image_root)
        if image_path is None:
            stats["missing_image"] += 1
            continue
        bbox = parse_bbox(row.get("bbox"))
        if bbox is None:
            stats["bad_bbox"] += 1
            continue
        width, height = image_size(image_path)
        bbox_text = bbox_to_qwen_tokens(bbox, width, height)
        image_value = str(image_path)
        if args.image_path_mode == "relative":
            try:
                image_value = str(image_path.relative_to(image_folder))
            except ValueError:
                image_value = str(image_path)

        question_id = clean_text(row.get("question_id")) or str(idx)
        image_id = clean_text(row.get("image_id")) or image_path.name
        sample_id = (
            f"dvgbench_gen_{idx:06d}_{safe_id(row.get('dataset'), 'dataset')}_"
            f"{safe_id(question_id, str(idx))}_{safe_id(Path(image_id).stem, str(idx))}_"
            f"{args.query_field}"
        )
        if args.mode == "i2e":
            sample_id += "_i2e"
        group = [
            {
                "id": sample_id,
                "image": image_value,
                "conversations": [
                    {"from": "human", "value": prompt_for_query(query, args.mode)},
                    {
                        "from": "gpt",
                        "value": answer_for_bbox(
                            bbox_text,
                            query,
                            args.mode,
                            explicit_reference=explicit_reference,
                        ),
                    },
                ],
                "metadata": {
                    "protocol": args.mode,
                    "source_question_id": question_id,
                    "explicit_supervision_train_only": args.mode == "i2e",
                },
            }
        ]
        stats["source_rows_written"] += 1

        if (
            args.mode == "i2e"
            and args.i2e_answer_only_copy_ratio > 0.0
            and copy_rng.random() < args.i2e_answer_only_copy_ratio
        ):
            group.append(
                {
                    "id": sample_id + "_answer_only_copy",
                    "image": image_value,
                    "conversations": [
                        {"from": "human", "value": prompt_for_query(query, "answer_only")},
                        {"from": "gpt", "value": answer_for_bbox(bbox_text, query, "answer_only")},
                    ],
                    "metadata": {
                        "protocol": "answer_only_preservation",
                        "source_question_id": question_id,
                        "explicit_supervision_train_only": False,
                    },
                }
            )
            stats["answer_only_copies"] += 1

        sft_groups.append(group)
        stats["written"] += len(group)

        eval_row = {
            "sample_id": sample_id,
            "image": str(image_path),
            "image_rel": image_value,
            "query": query,
            "answer": bbox_text,
            "bbox": bbox,
            "bbox_norm": bbox_norm(bbox, width, height),
            "question": clean_text(row.get("question")),
            "dataset": clean_text(row.get("dataset")),
            "category": clean_text(row.get("class")) or "unknown",
            "split": clean_text(row.get("split")),
            "image_id": image_id,
            "question_id": question_id,
            "source": "DVGBench",
            "task_type": "generative_grounding",
            "task_tag": f"dvgbench_{args.query_field}_generative",
            "image_size": [width, height],
            "oracle_fields_present": not args.omit_oracle_fields_from_eval_index,
        }
        if not args.omit_oracle_fields_from_eval_index:
            eval_row["question_e"] = clean_text(row.get("question_e"))
            eval_row["question_e_cn"] = clean_text(row.get("question_e_cn"))
        eval_rows.append(eval_row)

    if args.validation_split:
        if not (0.0 < args.validation_split < 1.0):
            raise ValueError("--validation-split must be in [0, 1).")
        if not args.val_output:
            raise ValueError("--val-output is required when --validation-split > 0.")
        split_at = max(1, round(len(sft_groups) * (1.0 - args.validation_split)))
        train_rows = [item for group in sft_groups[:split_at] for item in group]
        val_rows = [item for group in sft_groups[split_at:] for item in group]
        write_json(Path(args.output).expanduser().resolve(), train_rows)
        write_json(Path(args.val_output).expanduser().resolve(), val_rows)
        stats["train_source_rows"] = split_at
        stats["val_source_rows"] = len(sft_groups) - split_at
        stats["train_written"] = len(train_rows)
        stats["val_written"] = len(val_rows)
    else:
        sft_rows = [item for group in sft_groups for item in group]
        write_json(Path(args.output).expanduser().resolve(), sft_rows)

    if args.write_eval_index:
        write_eval_index(Path(args.write_eval_index).expanduser().resolve(), eval_rows)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
