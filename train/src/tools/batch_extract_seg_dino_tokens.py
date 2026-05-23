from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
TRAIN_DIR = REPO_ROOT / "train"
SRC_DIR = TRAIN_DIR / "src"
TOOLS_DIR = SRC_DIR / "tools"

for path in (str(TOOLS_DIR), str(SRC_DIR), str(TRAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from extract_seg_dino_tokens import (  # noqa: E402
    ANCHOR_TOKENS,
    build_messages,
    patch_missing_xpu_backend,
    resolve_device,
    resolve_dtype,
    tensor_shape,
    to_cpu,
    token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract CoVT Seg-DINO visual thought tokens from a JSONL index."
    )
    parser.add_argument(
        "--index",
        required=True,
        help="JSONL file. Each row needs image, query, and sample_id fields.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/outputs/token_cache",
        help="Directory for {sample_id}.pt token caches.",
    )
    parser.add_argument(
        "--model-path",
        default="Wakals/CoVT-7B-seg_depth_dino",
        help="Local path or Hugging Face id for a CoVT checkpoint.",
    )
    parser.add_argument("--sam-tokens", type=int, default=8)
    parser.add_argument("--dino-tokens", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose output .pt already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failed rows and keep processing.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL path recording one status row per input sample.",
    )
    parser.add_argument(
        "--no-project",
        action="store_true",
        help="Only return raw anchor hidden states, not projected/cross-attended states.",
    )
    parser.add_argument(
        "--no-cross-attention",
        action="store_true",
        help="Return projected states without anchor cross-attention outputs.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("image", "query", "sample_id"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} is missing required field {key!r}")
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    return records


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("\\", "_").replace("/", "_").replace(":", "_")


def write_manifest_row(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as err:
        raise RuntimeError("Please install qwen-vl-utils before running extraction.") from err

    import torch

    patch_missing_xpu_backend(torch)
    device = resolve_device(args.device, torch)
    dtype = resolve_dtype(args.torch_dtype, torch)

    from transformers import AutoProcessor
    from training.covt_qwen2_5_vl import CoVTForConditionalGeneration

    records = read_jsonl(index_path, args.limit)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = CoVTForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.eval()

    anchor_ids = {name: token_id(processor.tokenizer, token) for name, token in ANCHOR_TOKENS.items()}
    model.get_anchor_token_idx(
        anchor_ids["sam"],
        anchor_ids["dino"],
        anchor_ids["depth"],
        anchor_ids["sd"],
        anchor_ids["internvit"],
        anchor_ids["pidinet"],
        anchor_ids["siglip"],
        anchor_ids["metaclip"],
    )

    ok_count = 0
    skipped_count = 0
    failed_count = 0
    for idx, row in enumerate(records, start=1):
        sample_id = str(row["sample_id"])
        output_path = output_dir / f"{safe_sample_id(sample_id)}.pt"
        if args.resume and output_path.exists():
            skipped_count += 1
            write_manifest_row(
                manifest_path,
                {"sample_id": sample_id, "status": "skipped", "output": str(output_path)},
            )
            print(f"[{idx}/{len(records)}] skipped {sample_id}")
            continue

        try:
            image_path = os.path.abspath(os.path.expanduser(str(row["image"])))
            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)

            messages = build_messages(image_path, str(row["query"]), args.sam_tokens, args.dino_tokens)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                tokens = model.extract_visual_thought_tokens(
                    input_ids=inputs["input_ids"],
                    hidden_states=outputs.hidden_states[-1],
                    project=not args.no_project,
                    cross_attention=not args.no_cross_attention,
                    detach=True,
                )

            summary = {
                "model_path": args.model_path,
                "sample_id": sample_id,
                "image": image_path,
                "query": row["query"],
                "input_ids_shape": list(inputs["input_ids"].shape),
                "last_hidden_state_shape": list(outputs.hidden_states[-1].shape),
                "tokens": tensor_shape(tokens),
            }
            torch.save({"metadata": summary, "tokens": to_cpu(tokens)}, output_path)
            ok_count += 1
            write_manifest_row(
                manifest_path,
                {"sample_id": sample_id, "status": "ok", "output": str(output_path), "tokens": summary["tokens"]},
            )
            print(f"[{idx}/{len(records)}] ok {sample_id} -> {output_path}")
        except Exception as err:
            failed_count += 1
            write_manifest_row(
                manifest_path,
                {"sample_id": sample_id, "status": "error", "error": repr(err), "output": str(output_path)},
            )
            print(f"[{idx}/{len(records)}] error {sample_id}: {err}")
            if not args.continue_on_error:
                raise

    print(
        json.dumps(
            {
                "index": str(index_path),
                "output_dir": str(output_dir),
                "total": len(records),
                "ok": ok_count,
                "skipped": skipped_count,
                "failed": failed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
