from __future__ import annotations

from dataclasses import dataclass

from .agents.relation_reasoner import absolute_position_score, apply_ordinal_scores
from .geometry import box_area, box_iou, box_plausibility, center_distance
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


def _root_id(candidate: Candidate) -> str:
    return (
        candidate.hypothesis_id
        or candidate.parent_candidate_id
        or candidate.candidate_id
    )


def _is_independent_zoom(candidate: Candidate, config: AgenticConfig) -> bool:
    observation = candidate.observation
    return bool(
        candidate.source_agent == "ZoomAgent"
        and observation.view_type == "crop_zoom"
        and observation.crop_region is not None
        and box_area(observation.crop_region) <= config.verification_max_crop_area
    )


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


def _apply_global_constraints(
    valid: list[Candidate],
    graph: QueryConstraintGraph,
) -> None:
    roots = [candidate for candidate in valid if candidate.parent_candidate_id is None]
    ordinal_scores = apply_ordinal_scores(roots, graph.ordinal_constraint)
    root_by_id = {candidate.candidate_id: candidate for candidate in roots}
    for candidate in valid:
        root = root_by_id.get(_root_id(candidate), candidate)
        if root.candidate_id in ordinal_scores:
            candidate.global_constraint_score = ordinal_scores[root.candidate_id]
        elif SpatialFrame.GLOBAL_ABSOLUTE in graph.spatial_frames:
            candidate.global_constraint_score = absolute_position_score(
                candidate.bbox, graph.global_position
            )
        elif (
            SpatialFrame.GLOBAL_ORDER not in graph.spatial_frames
            and SpatialFrame.GLOBAL_ABSOLUTE not in graph.spatial_frames
        ):
            candidate.global_constraint_score = 0.5


def rank_candidates(
    candidates: list[Candidate],
    graph: QueryConstraintGraph,
    config: AgenticConfig,
) -> FusionResult:
    """Rank identity hypotheses, not a flat list of mutually supporting boxes."""
    valid = [candidate for candidate in candidates if candidate.bbox is not None]
    if not valid:
        raise ValueError("rank_candidates requires at least one valid candidate")
    by_id = {candidate.candidate_id: candidate for candidate in valid}
    for candidate in valid:
        candidate.hypothesis_id = _root_id(candidate)
        candidate.box_plausibility = box_plausibility(candidate.bbox)
        candidate.ambiguity_penalty = 0.0
        candidate.accepted_by_guard = not candidate.rejection_reasons
    _apply_global_constraints(valid, graph)
    _apply_zoom_guards(valid, by_id, config)

    roots = [candidate for candidate in valid if candidate.parent_candidate_id is None]
    children_by_root: dict[str, list[Candidate]] = {}
    for candidate in valid:
        if candidate.parent_candidate_id is not None:
            children_by_root.setdefault(_root_id(candidate), []).append(candidate)

    representatives: list[Candidate] = []
    hypotheses: list[dict] = []
    for root in roots:
        members = [root, *children_by_root.get(root.candidate_id, [])]
        accepted_children = [
            candidate
            for candidate in members[1:]
            if candidate.accepted_by_guard and candidate.bbox is not None
        ]
        independent_children = [
            candidate
            for candidate in accepted_children
            if _is_independent_zoom(candidate, config)
        ]
        stability = max(
            (box_iou(root.bbox, candidate.bbox) for candidate in independent_children),
            default=0.0,
        )
        supporting_children = [
            candidate
            for candidate in independent_children
            if box_iou(root.bbox, candidate.bbox)
            >= config.replacement_cross_view_iou_threshold
            and root.bbox_token_confidence >= config.verification_confidence_threshold
            and candidate.bbox_token_confidence
            >= config.verification_confidence_threshold
        ]
        cross_view_supported = bool(supporting_children)
        target_evidence = stability if cross_view_supported else 0.5
        for candidate in members:
            candidate.target_consistency = target_evidence
            candidate.observation_agreement = stability if accepted_children else 0.0
            candidate.fused_score = _candidate_score(candidate, config)
        eligible = [root, *accepted_children]
        representative = max(eligible, key=lambda item: item.fused_score)
        representatives.append(representative)
        hypotheses.append(
            {
                "hypothesis_id": root.candidate_id,
                "root_candidate_id": root.candidate_id,
                "root_source_agent": root.source_agent,
                "member_candidate_ids": [item.candidate_id for item in members],
                "accepted_verification_ids": [
                    item.candidate_id for item in accepted_children
                ],
                "supporting_verification_ids": [
                    item.candidate_id for item in supporting_children
                ],
                "cross_view_iou": stability,
                "cross_view_supported": cross_view_supported,
                "independent_verification_ids": [
                    item.candidate_id for item in independent_children
                ],
                "representative_candidate_id": representative.candidate_id,
                "hypothesis_score": representative.fused_score,
            }
        )

    if not representatives:
        representatives = sorted(valid, key=lambda item: item.fused_score, reverse=True)

    preliminary = sorted(
        representatives, key=lambda item: item.fused_score, reverse=True
    )
    if len(preliminary) > 1:
        first, second = preliminary[:2]
        margin = first.fused_score - second.fused_score
        separated = box_iou(first.bbox, second.bbox) < config.competition_iou_threshold
        if separated and margin < config.competition_margin_threshold:
            first.ambiguity_penalty = 1.0 - margin
            second.ambiguity_penalty = 1.0 - margin
            first.fused_score = _candidate_score(first, config)
            second.fused_score = _candidate_score(second, config)

    ranked = sorted(
        [candidate for candidate in representatives if candidate.accepted_by_guard],
        key=lambda item: item.fused_score,
        reverse=True,
    )
    if not ranked:
        ranked = preliminary
    for index, candidate in enumerate(ranked):
        next_score = ranked[index + 1].fused_score if index + 1 < len(ranked) else 0.0
        candidate.competition_margin = candidate.fused_score - next_score
    final = ranked[0]

    hypothesis_by_id = {item["hypothesis_id"]: item for item in hypotheses}
    for item in hypotheses:
        representative = by_id.get(item["representative_candidate_id"])
        if representative is not None:
            item["hypothesis_score"] = representative.fused_score
    evidence = {
        "valid_candidate_count": len(valid),
        "hypothesis_count": len(hypotheses),
        "accepted_candidate_count": len(ranked),
        "ranked_candidate_ids": [candidate.candidate_id for candidate in ranked],
        "ranked_hypothesis_ids": [_root_id(candidate) for candidate in ranked],
        "top_hypothesis_id": _root_id(final),
        "top_margin": final.competition_margin,
        "top_score": final.fused_score,
        "hypotheses": hypotheses,
        "selected_hypothesis": hypothesis_by_id.get(_root_id(final), {}),
        "rejected_candidates": {
            candidate.candidate_id: candidate.rejection_reasons
            for candidate in valid
            if not candidate.accepted_by_guard
        },
    }
    return FusionResult(ranked=ranked, final=final, evidence=evidence)
