from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


def patch_transformers_generation_for_extraction() -> None:
    """Avoid importing transformers.generation.utils for forward-only extraction.

    Some PyTorch/Python 3.12 environments hit a recursive import path inside
    transformers.generation.utils. Query hidden extraction never calls generate(),
    so a minimal GenerationMixin/GenerationConfig stub is enough for AutoProcessor
    and the local CoVT model class definitions.
    """

    if "transformers.generation" in sys.modules:
        return

    class GenerationMixin:
        def generate(self, *args, **kwargs):
            raise RuntimeError("Generation is disabled in query-hidden extraction.")

    class GenerationConfig:
        @classmethod
        def from_model_config(cls, *args, **kwargs):
            return cls()

        @classmethod
        def from_pretrained(cls, *args, return_unused_kwargs=False, **kwargs):
            config = cls()
            if return_unused_kwargs:
                return config, {}
            return config

        def update(self, **kwargs):
            return {}

        def to_dict(self):
            return {}

    class CompileConfig:
        pass

    generation_module = types.ModuleType("transformers.generation")
    generation_module.__path__ = []
    generation_module.GenerationMixin = GenerationMixin
    generation_module.GenerationConfig = GenerationConfig
    generation_module.CompileConfig = CompileConfig

    generation_utils_module = types.ModuleType("transformers.generation.utils")
    generation_utils_module.GenerationMixin = GenerationMixin

    generation_config_module = types.ModuleType("transformers.generation.configuration_utils")
    generation_config_module.GenerationConfig = GenerationConfig
    generation_config_module.CompileConfig = CompileConfig

    sys.modules["transformers.generation"] = generation_module
    sys.modules["transformers.generation.utils"] = generation_utils_module
    sys.modules["transformers.generation.configuration_utils"] = generation_config_module


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
    token_id,
)
from training.constants import DEFAULT_IM_END_TOKEN, VISION_END_TOKEN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract frozen CoVT/Qwen language hidden states for UAV adapter v8."
    )
    parser.add_argument("--index", required=True, help="JSONL file with image, query, and sample_id fields.")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/outputs/query_hidden_cache")
    parser.add_argument("--model-path", default="Wakals/CoVT-7B-seg_depth_dino")
    parser.add_argument("--sam-tokens", type=int, default=8)
    parser.add_argument("--dino-tokens", type=int, default=4)
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--save-dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Storage dtype for cached language hidden states.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--manifest", default=None)
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


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    last_start = len(sequence) - len(pattern)
    for index in range(start, last_start + 1):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def query_token_candidates(tokenizer, query: str) -> list[list[int]]:
    candidates = []
    seen = set()
    for text in (query, query.strip(), " " + query.strip(), "\n" + query.strip()):
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        key = tuple(token_ids)
        if token_ids and key not in seen:
            candidates.append(token_ids)
            seen.add(key)
    return candidates


def marker_token_ids(tokenizer, marker: str) -> list[int]:
    return tokenizer(marker, add_special_tokens=False).input_ids


def locate_query_positions(tokenizer, input_ids, query: str, max_tokens: int) -> tuple[list[int], str]:
    sequence = input_ids.tolist()
    for query_ids in query_token_candidates(tokenizer, query):
        start = find_subsequence(sequence, query_ids)
        if start >= 0:
            end = min(start + len(query_ids), start + max_tokens)
            return list(range(start, end)), "query_subsequence"

    vision_end_ids = marker_token_ids(tokenizer, VISION_END_TOKEN)
    im_end_ids = marker_token_ids(tokenizer, DEFAULT_IM_END_TOKEN)
    start = find_subsequence(sequence, vision_end_ids)
    if start >= 0:
        start += len(vision_end_ids)
        end = find_subsequence(sequence, im_end_ids, start=start)
        if end > start:
            end = min(end, start + max_tokens)
            return list(range(start, end)), "vision_end_to_im_end"

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    positions = [idx for idx, token in enumerate(sequence) if token not in special_ids]
    if positions:
        return positions[:max_tokens], "non_special_fallback"
    return [0], "first_token_fallback"


def cast_for_save(tensor, save_dtype: str, torch_module):
    if save_dtype == "float16":
        return tensor.to(torch_module.float16)
    if save_dtype == "bfloat16":
        return tensor.to(torch_module.bfloat16)
    return tensor.to(torch_module.float32)


def main() -> None:
    args = parse_args()
    patch_transformers_generation_for_extraction()
    index_path = Path(args.index).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as err:
        raise RuntimeError("Please install qwen-vl-utils before running query-hidden extraction.") from err

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

            query = str(row["query"])
            messages = build_messages(image_path, query, args.sam_tokens, args.dino_tokens)
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
                hidden_states = outputs.hidden_states[args.hidden_layer][0]
                positions, strategy = locate_query_positions(
                    processor.tokenizer,
                    inputs["input_ids"][0].detach().cpu(),
                    query,
                    args.max_query_tokens,
                )
                position_tensor = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
                query_hidden = hidden_states.index_select(0, position_tensor).detach().cpu()
                token_ids = inputs["input_ids"][0, position_tensor].detach().cpu().long()

            query_hidden = cast_for_save(query_hidden, args.save_dtype, torch)
            attention_mask = torch.ones(query_hidden.shape[0], dtype=torch.bool)
            summary = {
                "model_path": args.model_path,
                "sample_id": sample_id,
                "image": image_path,
                "query": query,
                "hidden_layer": args.hidden_layer,
                "selection_strategy": strategy,
                "query_token_count": int(query_hidden.shape[0]),
                "query_hidden_shape": list(query_hidden.shape),
                "save_dtype": args.save_dtype,
            }
            torch.save(
                {
                    "metadata": summary,
                    "lm_query": {
                        "hidden_states": query_hidden,
                        "attention_mask": attention_mask,
                        "token_ids": token_ids,
                        "positions": torch.tensor(positions, dtype=torch.long),
                    },
                },
                output_path,
            )
            ok_count += 1
            write_manifest_row(
                manifest_path,
                {"sample_id": sample_id, "status": "ok", "output": str(output_path), **summary},
            )
            print(f"[{idx}/{len(records)}] ok {sample_id} tokens={query_hidden.shape[0]} -> {output_path}")
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
