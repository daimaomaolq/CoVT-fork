from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
TRAIN_DIR = REPO_ROOT / "train"
SRC_DIR = TRAIN_DIR / "src"

for path in (str(SRC_DIR), str(TRAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.constants import (
    ANCHOR_END_TOKEN,
    ANCHOR_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEPTH_PAD_TOKEN,
    DINO_PAD_TOKEN,
    INTERN_PAD_TOKEN,
    METACLIP_PAD_TOKEN,
    PIDINET_PAD_TOKEN,
    SAM_PAD_TOKEN,
    SD_PAD_TOKEN,
    SIGLIP_PAD_TOKEN,
    VISION_END_TOKEN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CoVT/Qwen generative bbox outputs on DVGBench."
    )
    parser.add_argument("--index", required=True, help="JSONL from build_dvgbench_generative_sft.py.")
    parser.add_argument("--model-path", required=True, help="CoVT/Qwen model path or merged checkpoint.")
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT LoRA adapter path.")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary JSON path. Defaults to <output stem>.summary.json.",
    )
    parser.add_argument(
        "--require-oracle-free-index",
        action="store_true",
        help="Fail if question_e/question_e_cn exists in any evaluation row.",
    )
    parser.add_argument("--query-field", default="query", help="Index field used as prompt query.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1, help="Kept for CLI symmetry; generation is sequential.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-mode", default="answer_only", choices=("answer_only", "reasoning", "i2e"))
    parser.add_argument(
        "--anchor-model-id",
        default="[]",
        help="Anchor model ids used by the adapter, for example: ['sam','dino'].",
    )
    parser.add_argument(
        "--anchor-prompt-mode",
        default="none",
        choices=("none", "after_vision", "query_tail"),
        help="Where to insert anchor tokens during generation.",
    )
    parser.add_argument(
        "--anchor-token-counts",
        "--anchor_token_counts",
        dest="anchor_token_counts",
        default=None,
        help=(
            "Optional anchor token counts. Either 8 values in canonical order "
            "sam,dino,depth,SD,InternViT,pidinet,siglip,metaclip or one value per selected anchor."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def resolve_dtype(dtype_arg: str, torch_module):
    if dtype_arg == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_arg]


def resolve_device(device_arg: str, torch_module):
    if device_arg == "auto":
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_arg)


def prompt_for_query(query: str, mode: str) -> str:
    if mode == "reasoning":
        return (
            f"Locate the region described by: {query}\n"
            "Think briefly and put the final bounding box in "
            "<answer>{<x1><y1><x2><y2>}</answer>."
        )
    if mode == "i2e":
        return (
            f"Locate the region described by: {query}\n"
            "First convert the implicit request into one brief explicit visual "
            "description of the same target using visible category, attribute, "
            "position, context, or relation evidence. Then output its bounding "
            "box. Respond exactly as:\n"
            "<explicit>brief explicit description</explicit>\n"
            "<answer>{<x1><y1><x2><y2>}</answer>"
        )
    return (
        f"Locate the region described by: {query}\n"
        "Output only the bounding box in the format {<x1><y1><x2><y2>}."
    )


ANCHOR_TOKEN_BY_ID = {
    "sam": SAM_PAD_TOKEN,
    "dino": DINO_PAD_TOKEN,
    "depth": DEPTH_PAD_TOKEN,
    "SD": SD_PAD_TOKEN,
    "InternViT": INTERN_PAD_TOKEN,
    "pidinet": PIDINET_PAD_TOKEN,
    "siglip": SIGLIP_PAD_TOKEN,
    "metaclip": METACLIP_PAD_TOKEN,
}

CANONICAL_ANCHOR_ORDER = ["sam", "dino", "depth", "SD", "InternViT", "pidinet", "siglip", "metaclip"]

ANCHOR_COUNT_BY_ID = {
    "sam": 8,
    "dino": 4,
    "depth": 4,
    "SD": 4,
    "InternViT": 4,
    "pidinet": 4,
    "siglip": 4,
    "metaclip": 4,
}


def parse_anchor_model_ids(value: str | list[str] | None) -> list[str]:
    if value is None or value == "" or value == "[]":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = [part.strip() for part in str(value).split(",") if part.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    return [str(item) for item in parsed]


def parse_anchor_token_counts(anchor_model_ids: list[str], raw_counts: str | list[int] | None) -> list[int]:
    if raw_counts is None or raw_counts == "":
        return [ANCHOR_COUNT_BY_ID[anchor_model_id] for anchor_model_id in anchor_model_ids]
    if isinstance(raw_counts, str):
        raw_counts = ast.literal_eval(raw_counts)
    counts = [int(value) for value in raw_counts]
    if len(counts) == len(CANONICAL_ANCHOR_ORDER):
        count_by_id = dict(zip(CANONICAL_ANCHOR_ORDER, counts))
        return [count_by_id[anchor_model_id] for anchor_model_id in anchor_model_ids]
    if len(counts) == len(anchor_model_ids):
        return counts
    raise ValueError(
        "anchor_token_counts must contain either 8 canonical values "
        "or one value per selected anchor model"
    )


def build_anchor_prompt(anchor_model_ids: list[str], anchor_token_counts: list[int] | None = None) -> str:
    if anchor_token_counts is None:
        anchor_token_counts = parse_anchor_token_counts(anchor_model_ids, None)
    pads = []
    for anchor_model_id, count in zip(anchor_model_ids, anchor_token_counts):
        if anchor_model_id not in ANCHOR_TOKEN_BY_ID:
            raise ValueError(f"Unsupported anchor model id: {anchor_model_id}")
        token = ANCHOR_TOKEN_BY_ID[anchor_model_id]
        pads.append(ANCHOR_START_TOKEN + token * int(count) + ANCHOR_END_TOKEN)
    return "".join(pads)


def insert_anchor_prompt(text: str, anchor_prompt: str, mode: str) -> str:
    mode = (mode or "none").lower()
    if not anchor_prompt or mode == "none":
        return text
    if mode == "after_vision":
        if VISION_END_TOKEN in text:
            return text.replace(VISION_END_TOKEN, VISION_END_TOKEN + anchor_prompt, 1)
        mode = "query_tail"
    if mode != "query_tail":
        raise ValueError(f"Unsupported anchor_prompt_mode: {mode}")

    assistant_marker = f"{DEFAULT_IM_START_TOKEN}assistant"
    assistant_pos = text.rfind(assistant_marker)
    search_end = assistant_pos if assistant_pos >= 0 else len(text)
    user_end_pos = text.rfind(DEFAULT_IM_END_TOKEN, 0, search_end)
    prefix_source = text[:user_end_pos] if user_end_pos >= 0 else text
    anchor_prefix = "" if prefix_source.endswith("\n") else "\n"
    anchor_text = anchor_prefix + "Visual anchors: " + anchor_prompt + "\n"
    if user_end_pos >= 0:
        return text[:user_end_pos] + anchor_text + text[user_end_pos:]
    return text + anchor_text


def tokenizer_single_id(tokenizer, token: str) -> int:
    ids = tokenizer(token, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} is not a single tokenizer id: {ids}")
    return ids[0]


def validate_anchor_tokens(processor, anchor_model_ids: list[str]) -> list[int]:
    required_tokens = [ANCHOR_START_TOKEN, ANCHOR_END_TOKEN]
    required_tokens.extend(ANCHOR_TOKEN_BY_ID[item] for item in anchor_model_ids)
    return [tokenizer_single_id(processor.tokenizer, token) for token in required_tokens]


def anchor_token_indices(processor) -> list[int]:
    return [
        tokenizer_single_id(processor.tokenizer, SAM_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, DINO_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, DEPTH_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, SD_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, INTERN_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, PIDINET_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, SIGLIP_PAD_TOKEN),
        tokenizer_single_id(processor.tokenizer, METACLIP_PAD_TOKEN),
    ]

def parse_explicit_text(text: str) -> str | None:
    match = re.search(
        r"<explicit>\s*(.*?)\s*</explicit>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    explicit = re.sub(r"\s+", " ", match.group(1)).strip()
    return explicit or None


def parse_bbox_text(text: str) -> list[float] | None:
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    search_text = answer_match.group(1) if answer_match else text
    patterns = [
        r"\{\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*\}",
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, search_text)
        if match:
            values = [float(match.group(i)) for i in range(1, 5)]
            if max(values) > 1.5:
                values = [value / 1000.0 for value in values]
            return clamp_box(values)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", search_text)
    if len(numbers) >= 4:
        values = [float(value) for value in numbers[:4]]
        if max(values) > 1.5:
            values = [value / 1000.0 for value in values]
        return clamp_box(values)
    return None


def clamp_box(values: list[float]) -> list[float]:
    x1, y1, x2, y2 = values[:4]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def box_iou(pred: list[float] | None, target: list[float]) -> float:
    if pred is None:
        return 0.0
    px1, py1, px2, py2 = clamp_box(pred)
    tx1, ty1, tx2, ty2 = clamp_box(target)
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    pred_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
    target_area = max(tx2 - tx1, 0.0) * max(ty2 - ty1, 0.0)
    union = pred_area + target_area - inter
    return inter / union if union > 0 else 0.0


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor

    try:
        from training.covt_qwen2_5_vl import CoVTForConditionalGeneration

        model_cls = CoVTForConditionalGeneration
    except Exception:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration

    dtype = resolve_dtype(args.torch_dtype, torch)
    device = resolve_device(args.device, torch)
    anchor_model_ids = parse_anchor_model_ids(args.anchor_model_id)
    processor_candidates = []
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if (adapter_path / "tokenizer_config.json").is_file() or (adapter_path / "preprocessor_config.json").is_file():
            processor_candidates.append(str(adapter_path))
    processor_candidates.append(args.model_path)
    last_processor_error = None
    for processor_path in processor_candidates:
        try:
            processor = AutoProcessor.from_pretrained(processor_path)
            break
        except Exception as err:
            last_processor_error = err
    else:
        raise last_processor_error
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if anchor_model_ids:
        validate_anchor_tokens(processor, anchor_model_ids)
        if hasattr(model, "get_anchor_token_idx"):
            model.get_anchor_token_idx(*anchor_token_indices(processor))
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if len(processor.tokenizer) > embedding_rows:
        model.resize_token_embeddings(len(processor.tokenizer))
    if args.adapter_path:
        from peft import PeftModel

        non_lora_path = Path(args.adapter_path) / "non_lora_state_dict.bin"
        if non_lora_path.is_file():
            non_lora_state = torch.load(non_lora_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(non_lora_state, strict=False)
            print(
                json.dumps(
                    {
                        "status": "loaded_non_lora_state",
                        "path": str(non_lora_path),
                        "missing": len(missing),
                        "unexpected": len(unexpected),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    model.eval()
    return model, processor, device


def generate_one(model, processor, device, image_path: str, query: str, args: argparse.Namespace) -> str:
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_for_query(query, args.prompt_mode)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    anchor_model_ids = parse_anchor_model_ids(args.anchor_model_id)
    anchor_token_counts = parse_anchor_token_counts(anchor_model_ids, args.anchor_token_counts)
    anchor_prompt = build_anchor_prompt(anchor_model_ids, anchor_token_counts)
    text = insert_anchor_prompt(text, anchor_prompt, args.anchor_prompt_mode)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(generated_ids[0, input_len:], skip_special_tokens=False).strip()


def main() -> None:
    args = parse_args()
    forbidden_query_fields = {"question_e", "question_e_cn"}
    if args.query_field.lower() in forbidden_query_fields:
        raise ValueError("Final implicit-query evaluation cannot use an explicit oracle field.")

    rows = read_jsonl(Path(args.index).expanduser().resolve(), args.limit)
    if args.require_oracle_free_index:
        contaminated = [
            str(row.get("sample_id", index))
            for index, row in enumerate(rows)
            if any(field in row for field in forbidden_query_fields)
        ]
        if contaminated:
            raise ValueError(
                "Evaluation index contains forbidden oracle fields; first contaminated sample: "
                + contaminated[0]
            )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )
    model, processor, device = load_model(args)

    total = 0
    iou_sum = 0.0
    acc50 = 0
    parse_failed = 0
    explicit_parse_failed = 0
    explicit_lengths: list[int] = []
    latency_seconds: list[float] = []
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_acc: dict[str, int] = defaultdict(int)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            sample_id = str(row["sample_id"])
            query = str(row.get(args.query_field) or row.get("query") or "")
            started_at = time.perf_counter()
            raw_output = generate_one(model, processor, device, str(row["image"]), query, args)
            latency_seconds.append(time.perf_counter() - started_at)
            pred_bbox = parse_bbox_text(raw_output)
            explicit_prediction = parse_explicit_text(raw_output) if args.prompt_mode == "i2e" else None
            if pred_bbox is None:
                parse_failed += 1
            if args.prompt_mode == "i2e":
                if explicit_prediction is None:
                    explicit_parse_failed += 1
                else:
                    explicit_lengths.append(len(explicit_prediction.split()))

            # GT is accessed only after generation and is never passed to the model.
            gt_bbox = [float(v) for v in row["bbox_norm"][:4]]
            iou = box_iou(pred_bbox, gt_bbox)
            cls = str(row.get("category") or row.get("class") or "unknown").strip() or "unknown"
            total += 1
            iou_sum += iou
            hit = iou >= 0.5
            acc50 += int(hit)
            per_class_total[cls] += 1
            per_class_acc[cls] += int(hit)
            handle.write(
                json.dumps(
                    {
                        "schema_version": "dvgbench-qtsa-i2e-sft-v1",
                        "sample_id": sample_id,
                        "bbox": pred_bbox,
                        "gt_bbox": gt_bbox,
                        "iou": iou,
                        "class": cls,
                        "query": query,
                        "explicit_prediction": explicit_prediction,
                        "raw_output": raw_output,
                        "parse_ok": pred_bbox is not None,
                        "explicit_parse_ok": (
                            explicit_prediction is not None if args.prompt_mode == "i2e" else None
                        ),
                        "latency_seconds": latency_seconds[-1],
                        "protocol": {
                            "prompt_mode": args.prompt_mode,
                            "query_field": args.query_field,
                            "question_e_used": False,
                            "gt_visible_during_inference": False,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"[{total}/{len(rows)}] iou={iou:.4f} bbox_parse={pred_bbox is not None} "
                f"explicit_parse={explicit_prediction is not None} {sample_id}",
                flush=True,
            )

    class_acc = {
        key: per_class_acc[key] / max(per_class_total[key], 1)
        for key in sorted(per_class_total)
    }
    result = {
        "schema_version": "dvgbench-qtsa-i2e-sft-v1",
        "samples": total,
        "mIoU": iou_sum / max(total, 1),
        "Acc@0.5": acc50 / max(total, 1),
        "DVGBench_AVG": sum(class_acc.values()) / max(len(class_acc), 1),
        "class_Acc@0.5": class_acc,
        "class_counts": dict(sorted(per_class_total.items())),
        "parse_failed": parse_failed,
        "explicit_parse_failed": explicit_parse_failed if args.prompt_mode == "i2e" else None,
        "explicit_format_rate": (
            (total - explicit_parse_failed) / max(total, 1)
            if args.prompt_mode == "i2e"
            else None
        ),
        "mean_explicit_words": (
            sum(explicit_lengths) / len(explicit_lengths) if explicit_lengths else None
        ),
        "mean_latency_seconds": (
            sum(latency_seconds) / len(latency_seconds) if latency_seconds else 0.0
        ),
        "total_latency_seconds": sum(latency_seconds),
        "predictions": str(output_path),
        "config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "prompt_mode": args.prompt_mode,
            "query_field": args.query_field,
            "anchor_model_id": parse_anchor_model_ids(args.anchor_model_id),
            "anchor_prompt_mode": args.anchor_prompt_mode,
            "question_e_used": False,
            "gt_visible_during_inference": False,
            "oracle_free_index_required": args.require_oracle_free_index,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
