from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import box_area, box_plausibility
from .schema import AgenticConfig, Candidate, Method, QueryConstraintGraph, SpatialFrame


@dataclass
class RoutingPlan:
    perception_actions: list[str] = field(default_factory=list)
    run_relation_reasoner: bool = False
    preliminary_suspicions: list[str] = field(default_factory=list)
    action_utilities: dict[str, float] = field(default_factory=dict)
    rationale: dict[str, list[str]] = field(default_factory=dict)


def preliminary_suspicions(
    initial: Candidate,
    graph: QueryConstraintGraph,
    config: AgenticConfig,
) -> list[str]:
    suspicions = []
    if initial.bbox is None:
        return ["parse_risk", "needs_target_verification"]
    area = box_area(initial.bbox)
    shape = box_plausibility(initial.bbox)
    if (
        initial.confidence_available
        and initial.bbox_token_confidence < config.confidence_threshold
    ):
        suspicions.append("low_token_confidence")
    if shape < config.shape_threshold or area > config.large_area_threshold:
        suspicions.append("shape_risk")
    if area < config.small_area_threshold or {"small", "tiny"}.intersection(
        graph.attributes
    ):
        suspicions.append("small_object_risk")
    if graph.has_context:
        suspicions.append("needs_context_verification")
    if graph.has_relation:
        suspicions.append("needs_relation_verification")
    if SpatialFrame.GLOBAL_ABSOLUTE in graph.spatial_frames:
        suspicions.append("global_position_sensitive")
    if SpatialFrame.GLOBAL_ORDER in graph.spatial_frames:
        suspicions.extend(("global_position_sensitive", "candidate_ambiguity"))
    if (
        graph.has_context
        or graph.has_relation
        or SpatialFrame.GLOBAL_ORDER in graph.spatial_frames
    ):
        suspicions.append("needs_target_verification")
    if SpatialFrame.ORIENTATION_DEPENDENT in graph.spatial_frames:
        suspicions.append("orientation_evidence_risk")
    if SpatialFrame.TEMPORAL_EVENT in graph.spatial_frames:
        suspicions.append("temporal_evidence_risk")
    return list(dict.fromkeys(suspicions))


def _utility(need: float, expected_recovery: float, relative_cost: float) -> float:
    return need * expected_recovery / max(relative_cost, 1e-6)


class DependencyAwareRouter:
    def plan(
        self,
        initial: Candidate,
        graph: QueryConstraintGraph,
        config: AgenticConfig,
    ) -> RoutingPlan:
        suspicions = preliminary_suspicions(initial, graph, config)
        plan = RoutingPlan(preliminary_suspicions=suspicions)
        if config.method in {
            Method.ONE_PASS,
            Method.CONFIDENCE_GATED,
            Method.PARENT_ONLY,
        }:
            return plan

        needs_target = bool(
            {
                "parse_risk",
                "low_token_confidence",
                "shape_risk",
                "needs_target_verification",
                "candidate_ambiguity",
            }.intersection(suspicions)
        )
        if (
            config.method == Method.HIERARCHICAL
            and config.competition_probe_mode == "always"
            and config.max_child_perception_calls > 0
        ):
            # A single prediction cannot expose a confident localization error.
            # One target-clause probe supplies the minimum independent competition.
            needs_target = True
        needs_context = graph.has_context and bool(
            {
                "needs_context_verification",
                "needs_relation_verification",
                "candidate_ambiguity",
            }.intersection(suspicions)
        )
        needs_zoom = bool({"small_object_risk", "shape_risk"}.intersection(suspicions))

        if config.method == Method.STATIC_ALL:
            needs_target = True
            needs_context = graph.has_context
            needs_zoom = initial.bbox is not None

        action_candidates: list[tuple[str, float, list[str]]] = []
        if needs_target and "target" not in config.disabled_agents:
            reasons = [
                reason
                for reason in suspicions
                if reason
                in {
                    "parse_risk",
                    "low_token_confidence",
                    "shape_risk",
                    "needs_target_verification",
                    "candidate_ambiguity",
                }
            ]
            action_candidates.append(
                (
                    "target",
                    _utility(1.0, 0.75 if "parse_risk" in reasons else 0.55, 1.0),
                    reasons,
                )
            )
        if needs_context and "context" not in config.disabled_agents:
            reasons = [
                reason
                for reason in suspicions
                if reason
                in {
                    "needs_context_verification",
                    "needs_relation_verification",
                    "candidate_ambiguity",
                }
            ]
            action_candidates.append(
                (
                    "context",
                    _utility(1.0, 0.50, 1.0),
                    reasons,
                )
            )
        if needs_zoom and "zoom" not in config.disabled_agents:
            reasons = [
                reason
                for reason in suspicions
                if reason in {"small_object_risk", "shape_risk"}
            ]
            action_candidates.append(
                (
                    "zoom",
                    _utility(
                        1.0, 0.60 if "small_object_risk" in reasons else 0.40, 1.0
                    ),
                    reasons,
                )
            )

        # Dependencies override a pure utility sort: target/context evidence must precede
        # relation reasoning, and identity selection must precede zoom.
        order = {"target": 0, "context": 1, "zoom": 2}
        action_candidates.sort(key=lambda item: (order[item[0]], -item[1]))
        budget = config.max_child_perception_calls
        plan.perception_actions = [item[0] for item in action_candidates[:budget]]
        plan.action_utilities = {item[0]: item[1] for item in action_candidates}
        plan.rationale = {item[0]: item[2] for item in action_candidates}
        plan.run_relation_reasoner = "relation" not in config.disabled_agents and (
            graph.has_relation
            or SpatialFrame.GLOBAL_ABSOLUTE in graph.spatial_frames
            or SpatialFrame.GLOBAL_ORDER in graph.spatial_frames
        )
        return plan
