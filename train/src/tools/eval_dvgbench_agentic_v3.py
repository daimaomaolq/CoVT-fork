#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_agentic.evaluation import attach_evaluation, summarize
from uav_agentic.grounder import CoVTGrounder, GrounderSettings
from uav_agentic.io import (
    candidate_from_prediction,
    inference_input_from_row,
    load_cached_predictions,
    read_jsonl,
    resolve_image_path,
)
from uav_agentic.parent_agent import HierarchicalParentAgent
from uav_agentic.schema import AgenticConfig, Method, to_jsonable


class UnavailableGrounder:
    def ground(self, *args, **kwargs):
        raise RuntimeError("A model is required for uncached grounding calls")

    def generate_base_text(self, *args, **kwargs):
        raise RuntimeError("A model is required for base-generated feedback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnosis-driven hierarchical active perception for DVGBench. "
            "The main track forbids question_e and keeps GT outside inference."
        )
    )
    parser.add_argument("--index", required=True, help="DVGBench JSONL index")
    parser.add_argument("--output", required=True, help="Per-sample trace JSONL")
    parser.add_argument("--summary-output", help="Summary JSON; defaults beside output")
    parser.add_argument("--initial-predictions", help="Optional cached one-pass JSONL")
    parser.add_argument("--query-field", default="question")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--method",
        choices=[method.value for method in Method],
        default=Method.HIERARCHICAL.value,
    )

    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--prompt-mode", choices=("answer_only", "reasoning"), default="answer_only"
    )
    parser.add_argument("--anchor-model-id", default="[]")
    parser.add_argument("--anchor-prompt-mode", default="none")
    parser.add_argument("--anchor-token-counts")
    parser.add_argument("--include-raw-output", action="store_true")

    parser.add_argument("--max-child-perception-calls", type=int, default=3)
    parser.add_argument(
        "--disable-agent",
        action="append",
        choices=("target", "context", "relation", "zoom"),
        default=[],
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.42)
    parser.add_argument("--shape-threshold", type=float, default=0.45)
    parser.add_argument("--small-area-threshold", type=float, default=0.003)
    parser.add_argument("--large-area-threshold", type=float, default=0.55)
    parser.add_argument("--agreement-threshold", type=float, default=0.25)
    parser.add_argument("--relation-threshold", type=float, default=0.35)
    parser.add_argument("--global-constraint-threshold", type=float, default=0.40)
    parser.add_argument("--competition-iou-threshold", type=float, default=0.20)
    parser.add_argument("--competition-margin-threshold", type=float, default=0.12)
    parser.add_argument("--final-confidence-threshold", type=float, default=0.48)
    parser.add_argument("--information-gain-threshold", type=float, default=0.02)
    parser.add_argument("--zoom-scales", type=float, nargs="+", default=[1.5, 2.0])
    parser.add_argument("--zoom-identity-iou-threshold", type=float, default=0.05)
    parser.add_argument("--zoom-center-distance-threshold", type=float, default=0.20)
    parser.add_argument("--zoom-relation-drop-tolerance", type=float, default=0.15)
    parser.add_argument("--zoom-global-drop-tolerance", type=float, default=0.10)
    parser.add_argument("--context-union-margin", type=float, default=1.25)
    parser.add_argument(
        "--front-behind-axis", choices=("unknown", "y"), default="unknown"
    )
    parser.add_argument(
        "--feedback-mode", choices=("off", "template", "base"), default="template"
    )
    parser.add_argument("--no-escalation", action="store_true")

    parser.add_argument("--weight-full-confidence", type=float, default=0.30)
    parser.add_argument("--weight-shape", type=float, default=0.10)
    parser.add_argument("--weight-target", type=float, default=0.20)
    parser.add_argument("--weight-relation", type=float, default=0.15)
    parser.add_argument("--weight-global", type=float, default=0.15)
    parser.add_argument("--weight-stability", type=float, default=0.10)
    parser.add_argument("--ambiguity-penalty-weight", type=float, default=0.15)
    return parser.parse_args()


def build_agent_config(args: argparse.Namespace) -> AgenticConfig:
    return AgenticConfig(
        method=Method(args.method),
        max_child_perception_calls=args.max_child_perception_calls,
        confidence_threshold=args.confidence_threshold,
        shape_threshold=args.shape_threshold,
        small_area_threshold=args.small_area_threshold,
        large_area_threshold=args.large_area_threshold,
        agreement_threshold=args.agreement_threshold,
        relation_threshold=args.relation_threshold,
        global_constraint_threshold=args.global_constraint_threshold,
        competition_iou_threshold=args.competition_iou_threshold,
        competition_margin_threshold=args.competition_margin_threshold,
        final_confidence_threshold=args.final_confidence_threshold,
        information_gain_threshold=args.information_gain_threshold,
        zoom_scales=tuple(args.zoom_scales),
        zoom_identity_iou_threshold=args.zoom_identity_iou_threshold,
        zoom_center_distance_threshold=args.zoom_center_distance_threshold,
        zoom_relation_drop_tolerance=args.zoom_relation_drop_tolerance,
        zoom_global_drop_tolerance=args.zoom_global_drop_tolerance,
        context_union_margin=args.context_union_margin,
        disabled_agents=set(args.disable_agent),
        feedback_mode=args.feedback_mode,
        enable_escalation=not args.no_escalation,
        include_raw_output=args.include_raw_output,
        front_behind_axis=args.front_behind_axis,
        weight_full_confidence=args.weight_full_confidence,
        weight_shape=args.weight_shape,
        weight_target=args.weight_target,
        weight_relation=args.weight_relation,
        weight_global=args.weight_global,
        weight_stability=args.weight_stability,
        ambiguity_penalty_weight=args.ambiguity_penalty_weight,
    )


def build_grounder(args: argparse.Namespace, needs_model: bool):
    if not args.model_path:
        if needs_model:
            raise ValueError(
                "--model-path is required because at least one grounding or feedback "
                "model call is not available from cache"
            )
        return UnavailableGrounder()
    settings = GrounderSettings(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_mode=args.prompt_mode,
        anchor_model_id=args.anchor_model_id,
        anchor_prompt_mode=args.anchor_prompt_mode,
        anchor_token_counts=args.anchor_token_counts,
        include_raw_output=args.include_raw_output,
    )
    return CoVTGrounder.load(settings)


def _public_config(args: argparse.Namespace, agent_config: AgenticConfig) -> dict[str, Any]:
    return {
        "query_field": args.query_field,
        "question_e_used": False,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "initial_predictions": args.initial_predictions,
        "agent": to_jsonable(agent_config),
    }


def main() -> None:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )
    rows = read_jsonl(index_path, args.limit)
    if not rows:
        raise ValueError(f"No rows found in {index_path}")
    samples = [inference_input_from_row(row, args.query_field) for row in rows]
    cached_path = (
        Path(args.initial_predictions).expanduser().resolve()
        if args.initial_predictions
        else None
    )
    cached = load_cached_predictions(cached_path)
    method = Method(args.method)
    all_initial_cached = all(sample["sample_id"] in cached for sample in samples)
    needs_model = (
        not all_initial_cached
        or method != Method.ONE_PASS
        or args.feedback_mode == "base"
    )
    grounder = build_grounder(args, needs_model)
    agent_config = build_agent_config(args)
    parent = HierarchicalParentAgent(grounder, agent_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for position, (row, sample) in enumerate(zip(rows, samples), 1):
            image_path = resolve_image_path(sample["image"], index_path)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image for sample {sample['sample_id']} not found: {image_path}"
                )
            with Image.open(image_path) as loaded_image:
                image = loaded_image.convert("RGB")
            cached_initial = None
            if sample["sample_id"] in cached:
                cached_initial = candidate_from_prediction(
                    cached[sample["sample_id"]], query=sample["query"]
                )
            result = parent.run(
                sample_id=sample["sample_id"],
                image=image,
                query=sample["query"],
                sample_class=sample["class"],
                cached_initial=cached_initial,
            )
            result["image"] = sample["image"]
            result["bbox"] = result["inference"]["final_bbox"]
            result["parse_ok"] = result["bbox"] is not None
            gt_bbox = row.get("bbox_norm")
            if not isinstance(gt_bbox, (list, tuple)) or len(gt_bbox) < 4:
                raise ValueError(
                    f"Missing bbox_norm for offline evaluation of {sample['sample_id']}"
                )
            result = attach_evaluation(result, list(gt_bbox))
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{position}/{len(rows)}] {sample['sample_id']} "
                f"initial_iou={result['evaluation']['initial_iou']:.4f} "
                f"final_iou={result['evaluation']['final_iou']:.4f} "
                f"decision={result['inference']['decision']} "
                f"calls={result['cost']['perception_calls']}",
                flush=True,
            )

    summary = summarize(results, _public_config(args, agent_config))
    summary["predictions"] = str(output_path)
    summary["summary_output"] = str(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
