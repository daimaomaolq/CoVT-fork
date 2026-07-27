#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_agentic.counterfactual_candidate_selection import (
    CounterfactualConfig,
    apply_counterfactual_selection,
    counterfactual_verifier_summary,
)
from uav_agentic.evaluation import attach_evaluation, summarize
from uav_agentic.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply v4.3 evidence before joining GT for evaluation."
    )
    parser.add_argument("--candidate-trace", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--verifier-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--max-alternatives", type=int, default=1)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.92)
    parser.add_argument("--maximum-initial-overlap", type=float, default=0.85)
    parser.add_argument("--minimum-independent-gain", type=float, default=0.0)
    parser.add_argument("--allow-parent-accepted", action="store_true")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument("--first-crop-scale", type=float, default=3.0)
    parser.add_argument("--second-crop-scale", type=float, default=4.0)
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
    evidence_path = Path(args.verifier_evidence).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )
    traces = read_jsonl(trace_path, args.limit)
    index = load_unique(read_jsonl(index_path), "index")
    evidence_rows = read_jsonl(evidence_path, args.limit)
    evidence = load_unique(evidence_rows, "counterfactual verifier evidence")
    config = CounterfactualConfig(
        max_alternatives=args.max_alternatives,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        maximum_initial_overlap=args.maximum_initial_overlap,
        minimum_independent_gain=args.minimum_independent_gain,
        require_parent_unresolved=not args.allow_parent_accepted,
        require_independent_support=not args.allow_unsupported,
        first_crop_scale=args.first_crop_scale,
        second_crop_scale=args.second_crop_scale,
    )
    config.validate()
    if len(evidence_rows) != len(traces):
        raise ValueError(
            f"Evidence rows ({len(evidence_rows)}) != trace rows ({len(traces)})"
        )

    results: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for position, trace in enumerate(traces, 1):
            sample_id = str(trace.get("sample_id") or "")
            if sample_id not in index:
                raise KeyError(f"Sample {sample_id} missing from evaluation index")
            sidecar = evidence.get(sample_id)
            if sidecar is None:
                raise KeyError(f"Sample {sample_id} missing verifier evidence")
            if sidecar.get("question_e_used") is not False:
                raise ValueError(f"Unsafe verifier evidence for {sample_id}")
            if sidecar.get("gt_visible") is not False:
                raise ValueError(f"Verifier saw GT for {sample_id}")

            result = apply_counterfactual_selection(trace, sidecar, config)
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
                f"guard={result['inference']['counterfactual_selection']['guard_reason']} "
                f"initial_iou={result['evaluation']['initial_iou']:.4f} "
                f"final_iou={result['evaluation']['final_iou']:.4f}",
                flush=True,
            )

    public_config = {
        "code_revision": args.code_revision,
        "candidate_trace": str(trace_path),
        "verifier_evidence": str(evidence_path),
        "counterfactual": asdict(config),
        "protocol": {
            "frozen_candidate_pool": True,
            "final_bbox_regenerated": False,
            "candidate_centric_transformed_views": True,
            "counterbalanced_labels": True,
            "self_reported_confidence_used": False,
            "question_e_used": False,
            "gt_visible_during_selection": False,
            "gt_joined_after_selection": True,
        },
    }
    summary = summarize(results, public_config)
    replaced = [
        row
        for row in results
        if row["inference"]["counterfactual_selection"]["replaced_initial"]
    ]
    recovered = sum(row["evaluation"]["recovered_at_0_5"] for row in replaced)
    regressed = sum(row["evaluation"]["regressed_at_0_5"] for row in replaced)
    summary["counterfactual_verifier"] = counterfactual_verifier_summary(
        evidence_rows
    )
    summary["counterfactual_selection"] = {
        "Replacements": len(replaced),
        "Recovered": int(recovered),
        "Regressed": int(regressed),
        "Net Recovery Count": int(recovered - regressed),
        "Recovery Precision": (
            recovered / (recovered + regressed)
            if recovered + regressed
            else 0.0
        ),
        "Mean IoU Delta on Replacements": (
            sum(
                row["evaluation"]["final_iou"]
                - row["evaluation"]["initial_iou"]
                for row in replaced
            )
            / len(replaced)
            if replaced
            else 0.0
        ),
        "Guard Reason Distribution": dict(
            sorted(
                Counter(
                    row["inference"]["counterfactual_selection"]["guard_reason"]
                    for row in results
                ).items()
            )
        ),
    }
    summary["predictions"] = str(output_path)
    summary["summary_output"] = str(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
