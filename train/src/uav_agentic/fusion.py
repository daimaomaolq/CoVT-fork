from __future__ import annotations

from dataclasses import dataclass

from .agents.relation_reasoner import absolute_position_score, apply_ordinal_scores
from .geometry import box_iou, box_plausibility, center_distance
from .schema import AgenticConfig, Candidate, QueryConstraintGraph, SpatialFrame


@dataclass
class FusionResult:
    ranked: list[Candidate]
    final: Candidate
    evidence: dict


def _candidate_score(candidate: Candidate, config: AgenticConfig) -> float:
    confidence = (
        candidate.bbox_token_confidence if candidate.confidence_available else 0.5
    )
    weight_total = (
        config.weight_full_confidence
        + config.weight_shape
        + config.weight_target
        + config.weight_relation
        + config.weight_global
        + config.weight_stability
    )
    value = (
        config.weight_full_confidence * confidence
        + config.weight_shape * candidate.box_plausibility
        + config.weight_target * candidate.target_consistency
        + config.weight_relation * candidate.relation_consistency
        + config.weight_global * candidate.global_constraint_score
        + config.weight_stability * candidate.observation_agreement
    ) / weight_total - config.ambiguity_penalty_weight * candidate.ambiguity_penalty
    return max(0.0, min(1.0, value))


def _apply_zoom_guards(
    candidates: list[Candidate],
    by_id: dict[str, Candidate],
    config: AgenticConfig,
) -> None:
    if not config.enable_semantic_frame_protection:
        return
    for candidate in candidates:
        if candidate.source_agent != "ZoomAgent" or not candidate.parent_candidate_id:
            continue
        seed = by_id.get(candidate.parent_candidate_id)
        if seed is None or seed.bbox is None or candidate.bbox is None:
            candidate.accepted_by_guard = False
            candidate.rejection_reasons.append("missing_zoom_seed")
            continue
        identity_iou = box_iou(candidate.bbox, seed.bbox)
        identity_distance = center_distance(candidate.bbox, seed.bbox)
        if (
            identity_iou < config.zoom_identity_iou_threshold
            and identity_distance > config.zoom_center_distance_threshold
        ):
            candidate.accepted_by_guard = False
            candidate.rejection_reasons.append("zoom_identity_inconsistent")
        if (
            candidate.relation_consistency
            < seed.relation_consistency - config.zoom_relation_drop_tolerance
        ):
            candidate.accepted_by_guard = False
            candidate.rejection_reasons.append("zoom_relation_degraded")
        if (
            candidate.global_constraint_score
            < seed.global_constraint_score - config.zoom_global_drop_tolerance
        ):
            candidate.accepted_by_guard = False
            candidate.rejection_reasons.append("zoom_global_constraint_degraded")
        candidate.rejection_reasons = list(dict.fromkeys(candidate.rejection_reasons))


def rank_candidates(
    candidates: list[Candidate],
    graph: QueryConstraintGraph,
    config: AgenticConfig,
) -> FusionResult:
    valid = [candidate for candidate in candidates if candidate.bbox is not None]
    if not valid:
        raise ValueError("rank_candidates requires at least one valid candidate")
    target_specialists = [
        candidate for candidate in valid if candidate.source_agent == "TargetAgent"
    ]
    ordinal_scores = apply_ordinal_scores(valid, graph.ordinal_constraint)
    for candidate in valid:
        candidate.box_plausibility = box_plausibility(candidate.bbox)
        other_ious = [
            box_iou(candidate.bbox, other.bbox)
            for other in valid
            if other.candidate_id != candidate.candidate_id
        ]
        candidate.observation_agreement = max(other_ious, default=0.0)
        if target_specialists:
            if candidate.source_agent == "TargetAgent":
                support_candidates = [
                    other
                    for other in valid
                    if other.candidate_id != candidate.candidate_id
                ]
            else:
                support_candidates = target_specialists
            candidate.target_consistency = max(
                (
                    box_iou(candidate.bbox, support.bbox)
                    for support in support_candidates
                ),
                default=0.5,
            )
        else:
            candidate.target_consistency = 0.5
        if candidate.candidate_id in ordinal_scores:
            candidate.global_constraint_score = ordinal_scores[candidate.candidate_id]
        elif SpatialFrame.GLOBAL_ABSOLUTE in graph.spatial_frames:
            candidate.global_constraint_score = absolute_position_score(
                candidate.bbox, graph.global_position
            )
        elif (
            SpatialFrame.GLOBAL_ORDER not in graph.spatial_frames
            and SpatialFrame.GLOBAL_ABSOLUTE not in graph.spatial_frames
        ):
            candidate.global_constraint_score = 0.5
        candidate.ambiguity_penalty = 0.0
        candidate.fused_score = _candidate_score(candidate, config)

    preliminary = sorted(valid, key=lambda item: item.fused_score, reverse=True)
    if len(preliminary) > 1:
        first, second = preliminary[:2]
        margin = first.fused_score - second.fused_score
        separated = box_iou(first.bbox, second.bbox) < config.competition_iou_threshold
        if separated and margin < config.competition_margin_threshold:
            first.ambiguity_penalty = 1.0 - margin
            second.ambiguity_penalty = 1.0 - margin
            first.fused_score = _candidate_score(first, config)
            second.fused_score = _candidate_score(second, config)

    by_id = {candidate.candidate_id: candidate for candidate in valid}
    _apply_zoom_guards(valid, by_id, config)
    ranked = sorted(
        [candidate for candidate in valid if candidate.accepted_by_guard],
        key=lambda item: item.fused_score,
        reverse=True,
    )
    if not ranked:
        # Guard failure must not erase every prediction. Fall back to the best non-zoom
        # valid candidate and preserve the rejection trace.
        ranked = sorted(
            [candidate for candidate in valid if candidate.source_agent != "ZoomAgent"],
            key=lambda item: item.fused_score,
            reverse=True,
        )
    if not ranked:
        ranked = preliminary
    for index, candidate in enumerate(ranked):
        next_score = ranked[index + 1].fused_score if index + 1 < len(ranked) else 0.0
        candidate.competition_margin = candidate.fused_score - next_score
    final = ranked[0]
    evidence = {
        "valid_candidate_count": len(valid),
        "accepted_candidate_count": len(ranked),
        "ranked_candidate_ids": [candidate.candidate_id for candidate in ranked],
        "top_margin": final.competition_margin,
        "top_score": final.fused_score,
        "rejected_candidates": {
            candidate.candidate_id: candidate.rejection_reasons
            for candidate in valid
            if not candidate.accepted_by_guard
        },
    }
    return FusionResult(ranked=ranked, final=final, evidence=evidence)
