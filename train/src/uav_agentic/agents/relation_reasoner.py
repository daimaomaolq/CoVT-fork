from __future__ import annotations

from ..geometry import (
    bottom_center,
    box_center,
    box_iou,
    center_distance,
    contains_point,
)
from ..schema import AgentCall, Candidate, SpatialFrame
from .base import AgentContext, AgentResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def pair_relation_score(
    target: list[float],
    context: list[float],
    relations: list[str],
    front_behind_axis: str = "unknown",
) -> tuple[float, dict[str, float]]:
    if not relations:
        return 0.5, {}
    target_x, target_y = box_center(target)
    context_x, context_y = box_center(context)
    by_relation: dict[str, float] = {}
    for relation in relations:
        if relation == "left":
            score = _clamp01(0.5 + (context_x - target_x) * 2)
        elif relation == "right":
            score = _clamp01(0.5 + (target_x - context_x) * 2)
        elif relation == "above":
            score = _clamp01(0.5 + (context_y - target_y) * 2)
        elif relation == "below":
            score = _clamp01(0.5 + (target_y - context_y) * 2)
        elif relation == "near":
            score = _clamp01(1.0 - center_distance(target, context) * 3)
        elif relation == "inside":
            score = float(
                contains_point(context, box_center(target))
                and context[0] <= target[0] <= target[2] <= context[2]
                and context[1] <= target[1] <= target[3] <= context[3]
            )
        elif relation == "overlap":
            score = max(
                _clamp01(box_iou(target, context) * 3),
                float(contains_point(context, bottom_center(target))),
            )
        elif relation in {"front", "behind"}:
            if front_behind_axis == "unknown":
                score = 0.5
            elif relation == "front":
                score = _clamp01(0.5 + (context_y - target_y) * 2)
            else:
                score = _clamp01(0.5 + (target_y - context_y) * 2)
        else:
            score = 0.5
        by_relation[relation] = score
    return sum(by_relation.values()) / len(by_relation), by_relation


def absolute_position_score(box: list[float], position: str | None) -> float:
    if not position:
        return 0.5
    x, y = box_center(box)
    if position == "upper_left":
        return _clamp01((1 - x) * 0.5 + (1 - y) * 0.5)
    if position == "upper_right":
        return _clamp01(x * 0.5 + (1 - y) * 0.5)
    if position == "lower_left":
        return _clamp01((1 - x) * 0.5 + y * 0.5)
    if position == "lower_right":
        return _clamp01(x * 0.5 + y * 0.5)
    if position == "left":
        return 1 - x
    if position == "right":
        return x
    if position == "top":
        return 1 - y
    if position == "bottom":
        return y
    if position == "center":
        return _clamp01(1 - (((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5) * 2)
    if position == "edge":
        return _clamp01(max(abs(x - 0.5), abs(y - 0.5)) * 2)
    return 0.5


def apply_ordinal_scores(
    candidates: list[Candidate], ordinal: str | None
) -> dict[str, float]:
    valid = [candidate for candidate in candidates if candidate.bbox is not None]
    if not valid or not ordinal:
        return {}
    if ordinal in {"leftmost", "first", "second", "third"}:
        ordered = sorted(valid, key=lambda item: box_center(item.bbox)[0])
    elif ordinal == "rightmost":
        ordered = sorted(valid, key=lambda item: box_center(item.bbox)[0], reverse=True)
    elif ordinal == "topmost":
        ordered = sorted(valid, key=lambda item: box_center(item.bbox)[1])
    elif ordinal == "bottommost":
        ordered = sorted(valid, key=lambda item: box_center(item.bbox)[1], reverse=True)
    else:
        return {}
    desired_index = {"second": 1, "third": 2}.get(ordinal, 0)
    scores = {}
    for index, candidate in enumerate(ordered):
        score = (
            1.0
            if index == desired_index
            else max(0.0, 0.5 - 0.2 * abs(index - desired_index))
        )
        scores[candidate.candidate_id] = score
    return scores


class RelationAgent:
    name = "RelationAgent"

    def run(self, context: AgentContext) -> AgentResult:
        target_candidates = [
            candidate
            for candidate in context.target_candidates
            if candidate.bbox is not None
        ]
        context_candidates = [
            candidate
            for candidate in context.context_candidates
            if candidate.bbox is not None
        ]
        root_targets = [
            candidate
            for candidate in target_candidates
            if candidate.parent_candidate_id is None
        ]
        ordinal_scores = apply_ordinal_scores(
            root_targets, context.graph.ordinal_constraint
        )
        ranking = []
        unresolved = []
        for target in target_candidates:
            relation_details: dict[str, float] = {}
            if context.graph.has_relation and context_candidates:
                scored_contexts = []
                for reference in context_candidates:
                    score, details = pair_relation_score(
                        target.bbox,
                        reference.bbox,
                        context.graph.relations,
                        context.config.front_behind_axis,
                    )
                    scored_contexts.append((score, reference, details))
                relation_score, best_context, relation_details = max(
                    scored_contexts, key=lambda item: item[0]
                )
                target.context_consistency = best_context.bbox_token_confidence
                target.relation_consistency = relation_score
                best_context_id = best_context.candidate_id
            elif context.graph.has_relation:
                target.relation_consistency = 0.0
                best_context_id = None
                unresolved.append("relation_missing_context_evidence")
            else:
                best_context_id = None

            absolute_score = absolute_position_score(
                target.bbox, context.graph.global_position
            )
            root_id = (
                target.hypothesis_id
                or target.parent_candidate_id
                or target.candidate_id
            )
            ordinal_score = ordinal_scores.get(root_id)
            if ordinal_score is not None:
                target.global_constraint_score = ordinal_score
            elif SpatialFrame.GLOBAL_ABSOLUTE in context.graph.spatial_frames:
                target.global_constraint_score = absolute_score
            else:
                target.global_constraint_score = 0.5
            ranking.append(
                {
                    "target_candidate_id": target.candidate_id,
                    "context_candidate_id": best_context_id,
                    "relation_score": target.relation_consistency,
                    "relation_details": relation_details,
                    "global_constraint_score": target.global_constraint_score,
                }
            )

        ranking.sort(
            key=lambda item: (item["relation_score"] + item["global_constraint_score"]),
            reverse=True,
        )
        if (
            SpatialFrame.ORIENTATION_DEPENDENT in context.graph.spatial_frames
            and context.config.front_behind_axis == "unknown"
        ):
            unresolved.append("orientation_unresolved")
        call = AgentCall(
            call_id="call_relation",
            agent=self.name,
            action="relation_and_global_ranking",
            input={
                "target_candidate_ids": [
                    candidate.candidate_id for candidate in target_candidates
                ],
                "context_candidate_ids": [
                    candidate.candidate_id for candidate in context_candidates
                ],
                "relations": context.graph.relations,
                "global_position": context.graph.global_position,
                "ordinal_constraint": context.graph.ordinal_constraint,
            },
            output={"ranking": ranking},
            evidence={"unresolved": list(dict.fromkeys(unresolved))},
            model_call=False,
            perception_call=False,
        )
        return AgentResult(call=call, candidates=[], evidence=call.evidence)
