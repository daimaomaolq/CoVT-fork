from __future__ import annotations

import argparse
import json
import re
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CoVT/Qwen generative bbox outputs on DVGBench."
    )
    parser.add_argument("--index", required=True, help="JSONL from build_dvgbench_generative_sft.py.")
    parser.add_argument("--model-path", required=True, help="CoVT/Qwen model path or merged checkpoint.")
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT LoRA adapter path.")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument("--query-field", default="query", help="Index field used as prompt query.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1, help="Kept for CLI symmetry; generation is sequential.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-mode", default="answer_only", choices=("answer_only", "reasoning"))
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
            "Think briefly and put the final bounding box in <answer>{<x1><y1><x2><y2>}</answer>."
        )
    return (
        f"Locate the region described by: {query}\n"
        "Output only the bounding box in the format {<x1><y1><x2><y2>}."
    )


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
    rows = read_jsonl(Path(args.index).expanduser().resolve(), args.limit)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor, device = load_model(args)

    total = 0
    iou_sum = 0.0
    acc50 = 0
    parse_failed = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_acc: dict[str, int] = defaultdict(int)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            sample_id = str(row["sample_id"])
            query = str(row.get(args.query_field) or row.get("query") or "")
            raw_output = generate_one(model, processor, device, str(row["image"]), query, args)
            pred_bbox = parse_bbox_text(raw_output)
            if pred_bbox is None:
                parse_failed += 1
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
                        "sample_id": sample_id,
                        "bbox": pred_bbox,
                        "gt_bbox": gt_bbox,
                        "iou": iou,
                        "class": cls,
                        "query": query,
                        "raw_output": raw_output,
                        "parse_ok": pred_bbox is not None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{total}/{len(rows)}] iou={iou:.4f} parse={pred_bbox is not None} {sample_id}", flush=True)

    class_acc = {
        key: per_class_acc[key] / max(per_class_total[key], 1)
        for key in sorted(per_class_total)
    }
    result = {
        "samples": total,
        "mIoU": iou_sum / max(total, 1),
        "Acc@0.5": acc50 / max(total, 1),
        "DVGBench_AVG": sum(class_acc.values()) / max(len(class_acc), 1),
        "class_Acc@0.5": class_acc,
        "class_counts": dict(sorted(per_class_total.items())),
        "parse_failed": parse_failed,
        "predictions": str(output_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
