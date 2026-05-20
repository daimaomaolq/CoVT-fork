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

for path in (str(SRC_DIR), str(TRAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.constants import (
    ANCHOR_END_TOKEN,
    ANCHOR_START_TOKEN,
    DEPTH_PAD_TOKEN,
    DINO_PAD_TOKEN,
    INTERN_PAD_TOKEN,
    METACLIP_PAD_TOKEN,
    PIDINET_PAD_TOKEN,
    SAM_PAD_TOKEN,
    SD_PAD_TOKEN,
    SIGLIP_PAD_TOKEN,
)


ANCHOR_TOKENS = {
    "sam": SAM_PAD_TOKEN,
    "dino": DINO_PAD_TOKEN,
    "depth": DEPTH_PAD_TOKEN,
    "sd": SD_PAD_TOKEN,
    "internvit": INTERN_PAD_TOKEN,
    "pidinet": PIDINET_PAD_TOKEN,
    "siglip": SIGLIP_PAD_TOKEN,
    "metaclip": METACLIP_PAD_TOKEN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CoVT seg and DINO visual thought tokens from one UAV image."
    )
    parser.add_argument(
        "--model-path",
        default="Wakals/CoVT-7B-seg_depth_dino",
        help="Local path or Hugging Face id for a CoVT checkpoint.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--query",
        default="Find the target region in the UAV image.",
        help="Text query placed before the visual thought anchors.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .pt path for saving extracted tensors and metadata.",
    )
    parser.add_argument("--sam-tokens", type=int, default=8, help="Number of SAM anchor tokens.")
    parser.add_argument("--dino-tokens", type=int, default=4, help="Number of DINO anchor tokens.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string such as cuda:0 or cpu. Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Model load dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to from_pretrained.",
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


def resolve_device(device_arg: str, torch_module):
    if device_arg == "auto":
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_arg)


def resolve_dtype(dtype_arg: str, torch_module) -> Any:
    if dtype_arg == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_arg]


def token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer(token, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise ValueError(
            f"{token} must be a single tokenizer id, but got {token_ids}. "
            "Use a CoVT checkpoint whose tokenizer contains the anchor special tokens."
        )
    return token_ids[0]


def build_messages(image_path: str, query: str, sam_tokens: int, dino_tokens: int) -> list[dict[str, Any]]:
    sam_anchor = f"{ANCHOR_START_TOKEN}{SAM_PAD_TOKEN * sam_tokens}{ANCHOR_END_TOKEN}"
    dino_anchor = f"{ANCHOR_START_TOKEN}{DINO_PAD_TOKEN * dino_tokens}{ANCHOR_END_TOKEN}"
    assistant_text = (
        "<think> "
        f"the segmentation of the image is {sam_anchor}, "
        f"and the perception feature of the image is {dino_anchor}. "
        "</think>"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]


def tensor_shape(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "detach"):
        return list(value.shape)
    if isinstance(value, list):
        return [tensor_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_shape(item) for key, item in value.items()}
    return value


def to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as err:
        raise RuntimeError("Please install qwen-vl-utils before running extraction.") from err

    import torch

    device = resolve_device(args.device, torch)
    dtype = resolve_dtype(args.torch_dtype, torch)

    from transformers import AutoProcessor
    from training.covt_qwen2_5_vl import CoVTForConditionalGeneration

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

    messages = build_messages(image_path, args.query, args.sam_tokens, args.dino_tokens)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

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
        "image": image_path,
        "query": args.query,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "last_hidden_state_shape": list(outputs.hidden_states[-1].shape),
        "tokens": tensor_shape(tokens),
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "metadata": summary,
                "tokens": to_cpu(tokens),
            },
            output_path,
        )
        print(f"Saved extracted tokens to {output_path}")


if __name__ == "__main__":
    main()
