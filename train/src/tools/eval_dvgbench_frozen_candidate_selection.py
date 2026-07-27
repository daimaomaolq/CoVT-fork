#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_agentic.evaluation import attach_evaluation, summarize
from uav_agentic.frozen_candidate_selection import (
    SelectionConfig,
    apply_selection_to_trace,
    posthoc_action_metrics,
    verifier_summary,
)
from uav_agentic.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate selection on a frozen grounding trace. Selection "
            "receives inference fields and verifier evidence only; GT is joined "
            "after the final bbox has been committed."
        )
    )
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--index", required=True, help="GT-bearing evaluation index")
    parser.add_argument("--verifier-evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument(
        "--selector",
        choices=(
            "initial",
            "stored_fusion",
            "visual_only",
            "conservative_visual",
        ),
        default="conservative_visual",
    )
    parser.add_argument("--verifier-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--verifier-margin-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.05)
    parser.add_argument("--visual-weight", type=float, default=0.55)
    parser.add_argument("--token-weight", type=float, default=0.20)
    parser.add_argument("--relation-weight", type=float, default=0.10)
    parser.add_argument("--global-weight", type=float, default=0.10)
    parser.add_argument("--shape-weight", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--code-revision", default="unknown")
    return parser.parse_args()


def load_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"{name} contains a row without sample_id/id")
        if sample_id in result:
            raise ValueError(f"{name} contains duplicate sample_id={sample_id}")
        result[sample_id] = row
    return result


def main() -> None:
    args = parse_args()
    trace_path = Path(args.candidate_trace).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )
    traces = read_jsonl(trace_path, args.limit)
    index = load_unique(read_jsonl(index_path), "index")
    sidecar_rows = (
        read_jsonl(Path(args.verifier_evidence).expanduser().resolve(), args.limit)
        if args.verifier_evidence
        else []
    )
    sidecar = load_unique(sidecar_rows, "verifier evidence")
    config = SelectionConfig(
        selector=args.selector,
        verifier_confidence_threshold=args.verifier_confidence_threshold,
        verifier_margin_threshold=args.verifier_margin_threshold,
        minimum_composite_gain=args.minimum_composite_gain,
        visual_weight=args.visual_weight,
        token_weight=args.token_weight,
        relation_weight=args.relation_weight,
        global_weight=args.global_weight,
        shape_weight=args.shape_weight,
    )
    config.validate()

    results: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for position, trace in enumerate(traces, 1):
            sample_id = str(trace.get("sample_id") or "")
            if sample_id not in index:
                raise KeyError(f"Sample {sample_id} missing from evaluation index")
            inference = trace.get("inference", {})
            if inference.get("question_e_used") is not False:
                raise ValueError(f"Unsafe trace for {sample_id}: question_e_used")
            verifier = sidecar.get(sample_id)
            if verifier is not None:
                if verifier.get("question_e_used") is not False:
                    raise ValueError(f"Unsafe verifier evidence for {sample_id}")
                if verifier.get("gt_visible") is not False:
                    raise ValueError(f"Verifier saw GT for {sample_id}")

            # The selection function is called before bbox_norm is read.
            result = apply_selection_to_trace(trace, verifier, config)
            gt_bbox = index[sample_id].get("bbox_norm")
            if not isinstance(gt_bbox, (list, tuple)) or len(gt_bbox) < 4:
                raise ValueError(f"Missing bbox_norm in index for {sample_id}")
            result["class"] = str(
                index[sample_id].get("class") or result.get("class") or "unknown"
            )
            result = attach_evaluation(result, list(gt_bbox))
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{position}/{len(traces)}] {sample_id} "
                f"guard={result['inference']['candidate_selection']['guard_reason']} "
                f"initial_iou={result['evaluation']['initial_iou']:.4f} "
                f"final_iou={result['evaluation']['final_iou']:.4f}",
                flush=True,
            )

    public_config = {
        "code_revision": args.code_revision,
        "candidate_trace": str(trace_path),
        "verifier_evidence": str(
            Path(args.verifier_evidence).expanduser().resolve()
        )
        if args.verifier_evidence
        else None,
        "selector": {
            "name": config.selector,
            "verifier_confidence_threshold": (
                config.verifier_confidence_threshold
            ),
            "verifier_margin_threshold": config.verifier_margin_threshold,
            "minimum_composite_gain": config.minimum_composite_gain,
            "weights": {
                "visual": config.visual_weight,
                "token": config.token_weight,
                "relation": config.relation_weight,
                "global": config.global_weight,
                "shape": config.shape_weight,
            },
        },
        "protocol": {
            "frozen_candidate_pool": True,
            "bbox_regenerated": False,
            "question_e_used": False,
            "gt_visible_during_selection": False,
            "gt_joined_after_selection": True,
        },
    }
    summary = summarize(results, public_config)
    summary["frozen_candidate_selection"] = {
        "Selector": config.selector,
        "Replacements": sum(
            bool(row["inference"]["candidate_selection"]["replaced_initial"])
            for row in results
        ),
        "Visual-supported Replacements": sum(
            bool(row["inference"]["candidate_selection"]["replaced_initial"])
            and bool(row["inference"]["candidate_selection"]["visual_supported"])
            for row in results
        ),
        "Guard Reason Distribution": dict(
            sorted(Counter(
                    row["inference"]["candidate_selection"]["guard_reason"]
                    for row in results
                ).items())
        ),
    }
    summary["candidate_verifier"] = verifier_summary(sidecar_rows)
    summary["posthoc_action_quality"] = posthoc_action_metrics(results)
    summary["predictions"] = str(output_path)
    summary["summary_output"] = str(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
