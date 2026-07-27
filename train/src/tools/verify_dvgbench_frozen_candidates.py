#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_agentic.frozen_candidate_selection import (
    FrozenBaseVisualVerifier,
    SELECTOR_SCHEMA_VERSION,
    verify_frozen_candidates,
)
from uav_agentic.grounder import GrounderSettings
from uav_agentic.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen-base visual verifier over an existing candidate trace. "
            "No bbox is regenerated and GT/question_e are never read."
        )
    )
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--image-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--anchor-model-id", default="[]")
    parser.add_argument("--anchor-prompt-mode", default="none")
    parser.add_argument("--anchor-token-counts")
    parser.add_argument("--max-hypotheses", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-revision", default="unknown")
    return parser.parse_args()


def resolve_image(
    row: dict,
    trace_path: Path,
    image_root: Path | None,
) -> Path:
    raw = str(row.get("image") or "").strip()
    if not raw:
        raise ValueError(f"Trace sample {row.get('sample_id')} has no image path")
    value = Path(raw).expanduser()
    candidates = [value]
    if image_root is not None:
        candidates.append(image_root / value)
        candidates.append(image_root / value.name)
    candidates.append(trace_path.parent / value)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Image for {row.get('sample_id')} not found; checked: "
        + ", ".join(str(item) for item in candidates)
    )


def main() -> None:
    args = parse_args()
    if args.max_hypotheses < 2:
        raise ValueError("--max-hypotheses must be at least 2")
    trace_path = Path(args.candidate_trace).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    image_root = (
        Path(args.image_root).expanduser().resolve() if args.image_root else None
    )
    traces = read_jsonl(trace_path, args.limit)
    if not traces:
        raise ValueError(f"No candidate traces found in {trace_path}")
    for row in traces:
        inference = row.get("inference", {})
        if inference.get("question_e_used") is not False:
            raise ValueError(
                f"Unsafe trace {row.get('sample_id')}: question_e_used is not false"
            )

    completed: list[dict] = []
    start = 0
    if args.resume and output_path.is_file():
        completed = read_jsonl(output_path)
        if len(completed) > len(traces):
            raise ValueError("Verifier output is longer than candidate trace")
        for position, (sidecar, trace) in enumerate(zip(completed, traces), 1):
            if sidecar.get("sample_id") != trace.get("sample_id"):
                raise ValueError(f"Resume sample mismatch at row {position}")
            if sidecar.get("question_e_used") is not False:
                raise ValueError(f"Unsafe verifier sidecar at row {position}")
        start = len(completed)
        print(f"[resume] validated {start}/{len(traces)} rows", flush=True)

    pending = traces[start:]
    if pending:
        settings = GrounderSettings(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            max_new_tokens=160,
            temperature=0.0,
            prompt_mode="answer_only",
            anchor_model_id=args.anchor_model_id,
            anchor_prompt_mode=args.anchor_prompt_mode,
            anchor_token_counts=args.anchor_token_counts,
            include_raw_output=False,
        )
        verifier = FrozenBaseVisualVerifier.load(settings)
    else:
        verifier = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if start else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for position, row in enumerate(pending, start + 1):
            assert verifier is not None
            image_path = resolve_image(row, trace_path, image_root)
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
            result = verify_frozen_candidates(
                verifier, image, row, maximum=args.max_hypotheses
            )
            result["candidate_trace"] = str(trace_path)
            result["image"] = str(image_path)
            result["code_revision"] = args.code_revision
            result["protocol"] = {
                "bbox_regenerated": False,
                "adapter_disabled_for_verification": True,
                "question_e_used": False,
                "gt_visible": False,
                "coordinate_frame": "global_normalized",
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{position}/{len(traces)}] {result['sample_id']} "
                f"status={result['status']} confidence={result['confidence']:.3f}",
                flush=True,
            )

    manifest = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "samples": len(traces),
        "candidate_trace": str(trace_path),
        "verifier_output": str(output_path),
        "model_path": args.model_path,
        "adapter_path_loaded_for_grounding_compatibility": args.adapter_path,
        "adapter_disabled_for_verification": True,
        "max_hypotheses": args.max_hypotheses,
        "code_revision": args.code_revision,
        "question_e_used": False,
        "gt_visible": False,
        "bbox_regenerated": False,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
