from __future__ import annotations

import ast
import json
import math
import re
import time
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

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

from .geometry import clamp_box
from .schema import Candidate, Observation


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
CANONICAL_ANCHOR_ORDER = [
    "sam",
    "dino",
    "depth",
    "SD",
    "InternViT",
    "pidinet",
    "siglip",
    "metaclip",
]
DEFAULT_ANCHOR_COUNTS = {
    "sam": 8,
    "dino": 4,
    "depth": 4,
    "SD": 4,
    "InternViT": 4,
    "pidinet": 4,
    "siglip": 4,
    "metaclip": 4,
}


@dataclass
class GrounderSettings:
    model_path: str
    adapter_path: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str = "sdpa"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.9
    prompt_mode: str = "answer_only"
    anchor_model_id: str | list[str] = "[]"
    anchor_prompt_mode: str = "none"
    anchor_token_counts: str | list[int] | None = None
    include_raw_output: bool = False


class GrounderProtocol(Protocol):
    def ground(
        self,
        image: Image.Image,
        query: str,
        candidate_id: str,
        source_agent: str,
        observation: Observation,
        parent_candidate_id: str | None = None,
    ) -> Candidate:
        ...

    def generate_base_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        ...


def parse_anchor_model_ids(value: str | list[str] | None) -> list[str]:
    if value is None or value == "" or value == "[]":
        return []
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    result = [str(item) for item in parsed]
    unsupported = set(result).difference(ANCHOR_TOKEN_BY_ID)
    if unsupported:
        raise ValueError(f"Unsupported anchor model ids: {sorted(unsupported)}")
    return result


def parse_anchor_token_counts(
    anchor_ids: list[str], raw_counts: str | list[int] | None
) -> list[int]:
    if raw_counts is None or raw_counts == "":
        return [DEFAULT_ANCHOR_COUNTS[anchor_id] for anchor_id in anchor_ids]
    if isinstance(raw_counts, str):
        raw_counts = ast.literal_eval(raw_counts)
    counts = [int(value) for value in raw_counts]
    if len(counts) == len(CANONICAL_ANCHOR_ORDER):
        by_id = dict(zip(CANONICAL_ANCHOR_ORDER, counts))
        return [by_id[anchor_id] for anchor_id in anchor_ids]
    if len(counts) == len(anchor_ids):
        return counts
    raise ValueError(
        "anchor_token_counts must contain 8 canonical values or one per selected anchor"
    )


def build_anchor_prompt(anchor_ids: list[str], counts: list[int]) -> str:
    return "".join(
        ANCHOR_START_TOKEN + ANCHOR_TOKEN_BY_ID[anchor_id] * count + ANCHOR_END_TOKEN
        for anchor_id, count in zip(anchor_ids, counts)
    )


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
    assistant_position = text.rfind(assistant_marker)
    search_end = assistant_position if assistant_position >= 0 else len(text)
    user_end = text.rfind(DEFAULT_IM_END_TOKEN, 0, search_end)
    prefix = text[:user_end] if user_end >= 0 else text
    line_prefix = "" if prefix.endswith("\n") else "\n"
    addition = line_prefix + "Visual anchors: " + anchor_prompt + "\n"
    if user_end >= 0:
        return text[:user_end] + addition + text[user_end:]
    return text + addition


def parse_bbox_text(text: str) -> list[float] | None:
    answer = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE
    )
    search_text = answer.group(1) if answer else text
    patterns = (
        r"\{\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
        r"<\s*(-?\d+(?:\.\d+)?)\s*>\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*\}",
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
    )
    for pattern in patterns:
        match = re.search(pattern, search_text)
        if match:
            values = [float(match.group(index)) for index in range(1, 5)]
            if max(abs(value) for value in values) > 1.5:
                values = [value / 1000.0 for value in values]
            return clamp_box(values)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", search_text)
    if len(numbers) < 4:
        return None
    values = [float(value) for value in numbers[:4]]
    if max(abs(value) for value in values) > 1.5:
        values = [value / 1000.0 for value in values]
    return clamp_box(values)


def _resolve_dtype(value: str, torch_module):
    if value == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[value]


def _single_token_id(tokenizer, token: str) -> int:
    ids = tokenizer(token, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} is not a single tokenizer id: {ids}")
    return ids[0]


class CoVTGrounder:
    def __init__(self, model, processor, device, settings: GrounderSettings):
        self.model = model
        self.processor = processor
        self.device = device
        self.settings = settings
        self.anchor_ids = parse_anchor_model_ids(settings.anchor_model_id)
        self.anchor_counts = parse_anchor_token_counts(
            self.anchor_ids, settings.anchor_token_counts
        )
        self.anchor_prompt = build_anchor_prompt(self.anchor_ids, self.anchor_counts)

    @classmethod
    def load(cls, settings: GrounderSettings) -> "CoVTGrounder":
        import torch
        from transformers import AutoProcessor

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from training.covt_qwen2_5_vl import CoVTForConditionalGeneration

        model_class = CoVTForConditionalGeneration
        print(
            json.dumps(
                {
                    "status": "selected_grounder_model_class",
                    "model_class": model_class.__name__,
                    "generic_qwen_fallback": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        device = torch.device(
            "cuda:0"
            if settings.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if settings.device == "auto"
            else settings.device
        )
        processor_candidates = []
        if settings.adapter_path:
            adapter_path = Path(settings.adapter_path)
            if (adapter_path / "tokenizer_config.json").is_file() or (
                adapter_path / "preprocessor_config.json"
            ).is_file():
                processor_candidates.append(str(adapter_path))
        processor_candidates.append(settings.model_path)
        processor_error: Exception | None = None
        processor = None
        for candidate in processor_candidates:
            try:
                processor = AutoProcessor.from_pretrained(candidate)
                break
            except Exception as error:
                processor_error = error
        if processor is None:
            assert processor_error is not None
            raise processor_error

        model = model_class.from_pretrained(
            settings.model_path,
            torch_dtype=_resolve_dtype(settings.torch_dtype, torch),
            attn_implementation=settings.attn_implementation,
        )
        anchor_ids = parse_anchor_model_ids(settings.anchor_model_id)
        if anchor_ids:
            required = [ANCHOR_START_TOKEN, ANCHOR_END_TOKEN]
            required.extend(ANCHOR_TOKEN_BY_ID[anchor_id] for anchor_id in anchor_ids)
            for token in required:
                _single_token_id(processor.tokenizer, token)
            if hasattr(model, "get_anchor_token_idx"):
                canonical_tokens = [
                    SAM_PAD_TOKEN,
                    DINO_PAD_TOKEN,
                    DEPTH_PAD_TOKEN,
                    SD_PAD_TOKEN,
                    INTERN_PAD_TOKEN,
                    PIDINET_PAD_TOKEN,
                    SIGLIP_PAD_TOKEN,
                    METACLIP_PAD_TOKEN,
                ]
                model.get_anchor_token_idx(
                    *[
                        _single_token_id(processor.tokenizer, token)
                        for token in canonical_tokens
                    ]
                )
        embedding_rows = model.get_input_embeddings().weight.shape[0]
        if len(processor.tokenizer) > embedding_rows:
            model.resize_token_embeddings(len(processor.tokenizer))
        if settings.adapter_path:
            from peft import PeftModel

            non_lora_path = Path(settings.adapter_path) / "non_lora_state_dict.bin"
            if non_lora_path.is_file():
                state = torch.load(non_lora_path, map_location="cpu")
                missing, unexpected = model.load_state_dict(state, strict=False)
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
            model = PeftModel.from_pretrained(model, settings.adapter_path)
        model.to(device)
        model.eval()
        return cls(model, processor, device, settings)

    def _grounding_prompt(self, query: str) -> str:
        if self.settings.prompt_mode == "reasoning":
            return (
                f"Locate the region described by: {query}\n"
                "Think briefly and put the final bounding box in "
                "<answer>{<x1><y1><x2><y2>}</answer>."
            )
        return (
            f"Locate the region described by: {query}\n"
            "Output only the bounding box in the format {<x1><y1><x2><y2>}."
        )

    def _prepare_grounding_inputs(self, image: Image.Image, query: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._grounding_prompt(query)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = insert_anchor_prompt(
            text, self.anchor_prompt, self.settings.anchor_prompt_mode
        )
        return self.processor(text=[text], images=[image], return_tensors="pt")

    def _generation_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.settings.temperature > 0,
            "top_p": self.settings.top_p,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.settings.temperature > 0:
            kwargs["temperature"] = self.settings.temperature
        return kwargs

    def _bbox_token_confidence(
        self, scores: list[Any], token_ids: list[int]
    ) -> tuple[float, int]:
        import torch

        if not scores or not token_ids:
            return 0.5, 0
        selected = []
        for index, token_id in enumerate(token_ids):
            piece = self.processor.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            if re.search(r"[0-9<>{}\[\]().,-]", piece):
                selected.append(index)
        if not selected:
            selected = list(range(len(token_ids)))
        probabilities = []
        for index in selected:
            log_probability = torch.log_softmax(scores[index][0].float(), dim=-1)[
                int(token_ids[index])
            ]
            probabilities.append(float(log_probability.exp().item()))
        confidence = math.exp(
            sum(math.log(max(probability, 1e-9)) for probability in probabilities)
            / max(len(probabilities), 1)
        )
        return max(0.0, min(1.0, confidence)), len(selected)

    def ground(
        self,
        image: Image.Image,
        query: str,
        candidate_id: str,
        source_agent: str,
        observation: Observation,
        parent_candidate_id: str | None = None,
    ) -> Candidate:
        import torch

        started = time.perf_counter()
        inputs = self._prepare_grounding_inputs(image, query)
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            result = self.model.generate(
                **inputs,
                **self._generation_kwargs(self.settings.max_new_tokens),
            )
        generated_ids = result.sequences[0, input_length:]
        token_ids = generated_ids.tolist()
        raw_output = self.processor.decode(
            generated_ids, skip_special_tokens=False
        ).strip()
        confidence, token_count = self._bbox_token_confidence(
            list(result.scores), token_ids
        )
        return Candidate(
            candidate_id=candidate_id,
            bbox=parse_bbox_text(raw_output),
            source_agent=source_agent,
            query_used=query,
            observation=observation,
            bbox_token_confidence=confidence,
            bbox_token_count=token_count,
            raw_output=raw_output if self.settings.include_raw_output else "",
            latency_ms=(time.perf_counter() - started) * 1000,
            parent_candidate_id=parent_candidate_id,
        )

    def generate_base_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[1]
        adapter_context = (
            self.model.disable_adapter()
            if hasattr(self.model, "disable_adapter")
            else nullcontext()
        )
        kwargs = self._generation_kwargs(max_new_tokens)
        kwargs["output_scores"] = False
        with adapter_context, torch.no_grad():
            result = self.model.generate(**inputs, **kwargs)
        sequence = result.sequences if hasattr(result, "sequences") else result
        return self.processor.decode(
            sequence[0, input_length:], skip_special_tokens=True
        ).strip()
