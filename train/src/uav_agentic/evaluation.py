from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .geometry import box_iou, quantile


IOU_THRESHOLD = 0.5


def attach_evaluation(
    inference_result: dict[str, Any],
    gt_bbox: list[float],
) -> dict[str, Any]:
    """Attach GT-dependent values after inference has completely terminated."""
    gt = [float(value) for value in gt_bbox[:4]]
    inference = inference_result["inference"]
    initial_box = inference["initial_candidate"].get("bbox")
    final_box = inference.get("final_bbox")
    initial_iou = box_iou(initial_box, gt)
    final_iou = box_iou(final_box, gt)

    generated_candidates = [
        item
        for item in inference.get("target_candidates", [])
        if item.get("bbox") is not None
    ]
    candidates = {
        item["candidate_id"]: item
        for item in generated_candidates
        if item.get("accepted_by_guard", True)
    }
    ranked_ids = (
        inference.get("verification_evidence", {})
        .get("fusion", {})
        .get("ranked_candidate_ids", [])
    )
    ranked = [candidates[item] for item in ranked_ids if item in candidates]
    if not ranked:
        ranked = list(candidates.values())
    candidate_ious = [box_iou(item.get("bbox"), gt) for item in ranked]
    oracle_iou = max(
        (box_iou(item.get("bbox"), gt) for item in generated_candidates),
        default=0.0,
    )
    initial_id = inference["initial_candidate"].get("candidate_id", "c00")
    alternative_candidates = [
        item
        for item in generated_candidates
        if (
            item.get("hypothesis_id")
            or item.get("parent_candidate_id")
            or item["candidate_id"]
        )
        != initial_id
    ]
    alternative_ious = [
        box_iou(item.get("bbox"), gt) for item in alternative_candidates
    ]
    alternative_oracle_iou = max(alternative_ious, default=0.0)
    diversity_values = [
        1.0 - box_iou(item.get("bbox"), initial_box) for item in alternative_candidates
    ]
    final_item = next(
        (
            item
            for item in generated_candidates
            if item.get("candidate_id") == inference.get("final_candidate_id")
        ),
        {},
    )
    final_hypothesis_id = (
        final_item.get("hypothesis_id")
        or final_item.get("parent_candidate_id")
        or final_item.get("candidate_id")
    )
    candidate_recall = {
        f"CandidateRecall@{k}": float(
            any(iou >= IOU_THRESHOLD for iou in candidate_ious[:k])
        )
        for k in (1, 2, 3)
    }
    inference_result["evaluation"] = {
        "gt_bbox": gt,
        "initial_iou": initial_iou,
        "final_iou": final_iou,
        "initial_correct_at_0_5": initial_iou >= IOU_THRESHOLD,
        "final_correct_at_0_5": final_iou >= IOU_THRESHOLD,
        "recovered_at_0_5": initial_iou < IOU_THRESHOLD <= final_iou,
        "regressed_at_0_5": final_iou < IOU_THRESHOLD <= initial_iou,
        "candidate_ious_ranked": candidate_ious,
        "candidate_oracle_iou": oracle_iou,
        "oracle_gap_iou": max(0.0, oracle_iou - final_iou),
        "alternative_candidate_oracle_iou": alternative_oracle_iou,
        "alternative_candidate_hit_at_0_5": float(
            initial_iou < IOU_THRESHOLD <= alternative_oracle_iou
        ),
        "alternative_selected_at_0_5": float(
            initial_iou < IOU_THRESHOLD <= final_iou
            and final_hypothesis_id != initial_id
        ),
        "search_yield_at_delta_iou_0_1": float(oracle_iou - initial_iou >= 0.10),
        "generated_candidate_count": len(generated_candidates),
        "hypothesis_count": inference.get("verification_evidence", {})
        .get("fusion", {})
        .get("hypothesis_count", len(ranked)),
        "candidate_diversity": _mean(diversity_values),
        **candidate_recall,
    }
    return inference_result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _macro_acc50(
    rows: list[dict[str, Any]], key: str
) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("class") or "unknown")].append(
            float(row["evaluation"][key])
        )
    per_class = {
        name: _mean([float(value >= IOU_THRESHOLD) for value in values])
        for name, values in sorted(grouped.items())
    }
    return _mean(list(per_class.values())), per_class


def _auroc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return 0.0
    favorable = 0.0
    for pos in positive:
        for neg in negative:
            favorable += float(pos > neg) + 0.5 * float(pos == neg)
    return favorable / (len(positive) * len(negative))


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _calibration(
    labels: list[int], confidences: list[float], bins: int = 10
) -> dict[str, float]:
    if not labels:
        return {"ECE": 0.0, "Brier": 0.0}
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not selected:
            continue
        bin_accuracy = _mean([float(labels[index]) for index in selected])
        bin_confidence = _mean([confidences[index] for index in selected])
        ece += len(selected) / len(labels) * abs(bin_accuracy - bin_confidence)
    brier = _mean(
        [(confidence - label) ** 2 for label, confidence in zip(labels, confidences)]
    )
    return {"ECE": ece, "Brier": brier}


def _failure_type_recovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            diagnosis
            for row in rows
            for diagnosis in row["inference"].get("diagnosis", [])
        }
    )
    table: dict[str, Any] = {}
    for name in names:
        selected = [
            row for row in rows if name in row["inference"].get("diagnosis", [])
        ]
        failures = [
            row for row in selected if not row["evaluation"]["initial_correct_at_0_5"]
        ]
        recovered = sum(bool(row["evaluation"]["recovered_at_0_5"]) for row in failures)
        table[name] = {
            "Count": len(selected),
            "Initial Failures": len(failures),
            "Recovered": recovered,
            "Recovery Rate": _safe_ratio(recovered, len(failures)),
        }
    return table


def summarize(
    rows: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty result set")
    if any("evaluation" not in row for row in rows):
        raise ValueError("All rows must be evaluated before summarization")
    if any(
        row.get("inference", {}).get("question_e_used") is not False for row in rows
    ):
        raise ValueError("Unsafe trace: question_e_used must be explicitly false")

    initial_ious = [float(row["evaluation"]["initial_iou"]) for row in rows]
    final_ious = [float(row["evaluation"]["final_iou"]) for row in rows]
    initial_hits = [int(value >= IOU_THRESHOLD) for value in initial_ious]
    final_hits = [int(value >= IOU_THRESHOLD) for value in final_ious]
    failures = [1 - value for value in initial_hits]
    dispatches = [int(bool(row["cost"].get("dispatch"))) for row in rows]
    recovered = sum(bool(row["evaluation"]["recovered_at_0_5"]) for row in rows)
    regressed = sum(bool(row["evaluation"]["regressed_at_0_5"]) for row in rows)
    initial_failure_count = sum(failures)
    initial_hit_count = sum(initial_hits)
    final_macro, final_class = _macro_acc50(rows, "final_iou")
    initial_macro, initial_class = _macro_acc50(rows, "initial_iou")
    class_counts = dict(
        sorted(Counter(str(row.get("class") or "unknown") for row in rows).items())
    )

    initial_uncertainties = []
    final_confidences = []
    for row in rows:
        inference = row["inference"]
        initial_score = float(
            inference.get("verification_evidence", {}).get(
                "initial_fused_confidence",
                inference.get("initial_candidate", {}).get(
                    "bbox_token_confidence", 0.5
                ),
            )
        )
        if inference.get("initial_candidate", {}).get("bbox") is None:
            initial_score = 0.0
        initial_uncertainties.append(1.0 - max(0.0, min(1.0, initial_score)))
        final_confidences.append(
            max(0.0, min(1.0, float(inference.get("confidence", 0.0))))
        )

    true_positive_dispatch = sum(
        failure and dispatch for failure, dispatch in zip(failures, dispatches)
    )
    false_positive_dispatch = sum(
        (not failure) and dispatch for failure, dispatch in zip(failures, dispatches)
    )
    true_negative_dispatch = sum(
        (not failure) and (not dispatch)
        for failure, dispatch in zip(failures, dispatches)
    )
    covered = [row for row in rows if row["inference"].get("decision") != "escalate"]
    covered_hits = [int(row["evaluation"]["final_correct_at_0_5"]) for row in covered]

    perception_calls = [float(row["cost"].get("perception_calls", 0)) for row in rows]
    executed_calls = [
        float(row["cost"].get("executed_perception_calls", 0)) for row in rows
    ]
    unit_calls = [
        float(
            row["cost"].get(
                "specialized_unit_calls", row["cost"].get("child_agent_calls", 0)
            )
        )
        for row in rows
    ]
    initial_latencies = [
        float(row["cost"].get("initial_latency_ms", 0.0)) for row in rows
    ]
    incremental_latencies = [
        float(
            row["cost"].get(
                "incremental_agent_latency_ms", row["cost"].get("latency_ms", 0.0)
            )
        )
        for row in rows
    ]
    latencies = [
        float(
            row["cost"].get("end_to_end_latency_ms", row["cost"].get("latency_ms", 0.0))
        )
        for row in rows
    ]
    latency_availability = [
        float(bool(row["cost"].get("latency_available", False))) for row in rows
    ]
    dispatch_distribution = Counter(
        call["agent"]
        for row in rows
        for call in row["inference"].get(
            "unit_calls", row["inference"].get("child_calls", [])
        )
        if call.get("status") != "skipped"
    )
    unit_effect = {}
    for agent_name in sorted(dispatch_distribution):
        invoked = [
            row
            for row in rows
            if any(
                call.get("agent") == agent_name and call.get("status") != "skipped"
                for call in row["inference"].get(
                    "unit_calls", row["inference"].get("child_calls", [])
                )
            )
        ]
        unit_effect[agent_name] = {
            "Calls": dispatch_distribution[agent_name],
            "Samples Invoked": len(invoked),
            "Invocation Rate": _safe_ratio(len(invoked), len(rows)),
            "Recovered": sum(
                bool(row["evaluation"]["recovered_at_0_5"]) for row in invoked
            ),
            "Mean IoU Gain": _mean(
                [
                    float(row["evaluation"]["final_iou"])
                    - float(row["evaluation"]["initial_iou"])
                    for row in invoked
                ]
            ),
        }

    oracle_hits = [
        int(float(row["evaluation"]["candidate_oracle_iou"]) >= IOU_THRESHOLD)
        for row in rows
    ]
    alternative_hits = sum(
        int(row["evaluation"].get("alternative_candidate_hit_at_0_5", 0))
        for row in rows
    )
    search_yields = sum(
        int(row["evaluation"].get("search_yield_at_delta_iou_0_1", 0))
        for row in rows
        if not row["evaluation"]["initial_correct_at_0_5"]
    )
    verified_hypotheses = sum(
        sum(
            bool(item.get("cross_view_supported"))
            for item in row["inference"]
            .get("verification_evidence", {})
            .get("fusion", {})
            .get("hypotheses", [])
        )
        for row in rows
    )
    total_hypotheses = sum(
        int(row["evaluation"].get("hypothesis_count", 0)) for row in rows
    )
    initial_verified = 0
    verification_advantages = 0
    relocation_attempts = 0
    relocation_recoveries = 0
    for row in rows:
        fusion_evidence = (
            row["inference"].get("verification_evidence", {}).get("fusion", {})
        )
        hypotheses = fusion_evidence.get("hypotheses", [])
        if any(
            item.get("hypothesis_id") == "c00" and item.get("cross_view_supported")
            for item in hypotheses
        ):
            initial_verified += 1
        if fusion_evidence.get("verification_advantage"):
            verification_advantages += 1
        if row["inference"].get(
            "final_candidate_id"
        ) != "c00" and not fusion_evidence.get("same_identity_hypothesis", True):
            relocation_attempts += 1
            relocation_recoveries += int(
                row["evaluation"].get("recovered_at_0_5", False)
            )
    feedback_rows = [
        row for row in rows if row["inference"].get("human_feedback") is not None
    ]
    feedback_actions = Counter(
        row["inference"]["human_feedback"].get("recommended_action", "unknown")
        for row in feedback_rows
    )
    calibration = _calibration(final_hits, final_confidences)
    parse_failed = sum(
        row["inference"].get("initial_candidate", {}).get("bbox") is None
        for row in rows
    )

    return {
        "schema_version": rows[0].get("schema_version", "unknown"),
        "method": rows[0].get("method", "unknown"),
        "samples": len(rows),
        "config": config or {},
        "protocol_guards": {
            "question_e_used": False,
            "gt_visible_during_inference": False,
            "single_image_transformed_observations_only": True,
        },
        "one_pass": {
            "mIoU": _mean(initial_ious),
            "Acc@0.5": _mean(initial_hits),
            "DVGBench_AVG": initial_macro,
            "class_Acc@0.5": initial_class,
            "class_counts": class_counts,
            "parse_failed": parse_failed,
        },
        "agentic_inference": {
            "mIoU": _mean(final_ious),
            "Acc@0.5": _mean(final_hits),
            "DVGBench_AVG": final_macro,
            "class_Acc@0.5": final_class,
            "class_counts": class_counts,
            "Recovery@0.5": _safe_ratio(recovered, initial_failure_count),
            "False Repair Rate": _safe_ratio(regressed, initial_hit_count),
            "Regression@0.5": _safe_ratio(regressed, initial_hit_count),
            "Net Recovery Count": recovered - regressed,
            "Avg Calls": _mean(perception_calls),
            "Avg Executed Calls": _mean(executed_calls),
            "Avg Specialized Unit Calls": _mean(unit_calls),
            "Avg Child Calls": _mean(unit_calls),
            "Initial Latency_ms": _mean(initial_latencies),
            "Incremental Agent Latency_ms": _mean(incremental_latencies),
            "End-to-end Latency_ms": _mean(latencies),
            "Latency Availability Rate": _mean(latency_availability),
            "Latency_ms": _mean(latencies),
            "Latency_P50_ms": quantile(latencies, 0.50),
            "Latency_P95_ms": quantile(latencies, 0.95),
            "Dispatch Rate": _mean(dispatches),
        },
        "failure_detection": {
            "Precision": _safe_ratio(true_positive_dispatch, sum(dispatches)),
            "Recall": _safe_ratio(true_positive_dispatch, initial_failure_count),
            "Specificity": _safe_ratio(true_negative_dispatch, initial_hit_count),
            "False Dispatch Rate": _safe_ratio(
                false_positive_dispatch, initial_hit_count
            ),
            "AUROC": _auroc(failures, initial_uncertainties),
            "AUPRC": _average_precision(failures, initial_uncertainties),
        },
        "candidate_and_selection": {
            "CandidateRecall@1": _mean(
                [float(row["evaluation"]["CandidateRecall@1"]) for row in rows]
            ),
            "CandidateRecall@2": _mean(
                [float(row["evaluation"]["CandidateRecall@2"]) for row in rows]
            ),
            "CandidateRecall@3": _mean(
                [float(row["evaluation"]["CandidateRecall@3"]) for row in rows]
            ),
            "Candidate Oracle Acc@0.5": _mean(oracle_hits),
            "Alternative Candidate Recall@0.5": _safe_ratio(
                alternative_hits, initial_failure_count
            ),
            "Alternative Selection Success": _safe_ratio(recovered, alternative_hits),
            "Search Yield@DeltaIoU0.1": _safe_ratio(
                search_yields, initial_failure_count
            ),
            "Mean Generated Candidate Count": _mean(
                [
                    float(row["evaluation"].get("generated_candidate_count", 0))
                    for row in rows
                ]
            ),
            "Mean Hypothesis Count": _mean(
                [float(row["evaluation"].get("hypothesis_count", 0)) for row in rows]
            ),
            "Mean Candidate Diversity": _mean(
                [
                    float(row["evaluation"].get("candidate_diversity", 0.0))
                    for row in rows
                ]
            ),
            "Root Verification Rate": _safe_ratio(
                verified_hypotheses, total_hypotheses
            ),
            "Initial Hypothesis Verification Rate": _safe_ratio(
                initial_verified, len(rows)
            ),
            "Verification Advantage Rate": _safe_ratio(
                verification_advantages, len(rows)
            ),
            "Relocation Rate": _safe_ratio(relocation_attempts, len(rows)),
            "Relocation Recovery Precision": _safe_ratio(
                relocation_recoveries, relocation_attempts
            ),
            "Selection Success Given Oracle Hit": _safe_ratio(
                sum(final and oracle for final, oracle in zip(final_hits, oracle_hits)),
                sum(oracle_hits),
            ),
            "Mean Oracle IoU": _mean(
                [float(row["evaluation"]["candidate_oracle_iou"]) for row in rows]
            ),
            "Mean Oracle Gap IoU": _mean(
                [float(row["evaluation"]["oracle_gap_iou"]) for row in rows]
            ),
        },
        "selective_prediction": {
            "Coverage": _safe_ratio(len(covered), len(rows)),
            "Selective Acc@0.5": _mean(covered_hits),
            "Escalation Rate": 1.0 - _safe_ratio(len(covered), len(rows)),
        },
        "confidence_calibration": calibration,
        "failure_type_recovery": _failure_type_recovery(rows),
        "dispatch_distribution": dict(dispatch_distribution),
        "specialized_unit_effect": unit_effect,
        "child_agent_effect": unit_effect,
        "human_feedback": {
            "Count": len(feedback_rows),
            "Valid Rate": _safe_ratio(
                sum(
                    bool(row["inference"]["human_feedback"].get("valid"))
                    for row in feedback_rows
                ),
                len(feedback_rows),
            ),
            "Fallback Rate": _safe_ratio(
                sum(
                    bool(row["inference"]["human_feedback"].get("fallback_used"))
                    for row in feedback_rows
                ),
                len(feedback_rows),
            ),
            "Action Distribution": dict(feedback_actions),
        },
        "counts": {
            "initial_failures": initial_failure_count,
            "initial_hits": initial_hit_count,
            "recovered": recovered,
            "regressed": regressed,
        },
    }
