from __future__ import annotations

import time
from typing import Any

from PIL import Image, ImageEnhance

from .agents import ContextAgent, RelationAgent, TargetAgent, ZoomAgent
from .agents.base import AgentContext
from .feedback import generate_feedback
from .fusion import FusionResult, rank_candidates
from .geometry import box_area, box_iou, box_plausibility
from .grounder import GrounderProtocol
from .query_constraints import parse_query_constraints
from .routing import DependencyAwareRouter, RoutingPlan
from .schema import (
    AgentCall,
    AgenticConfig,
    Candidate,
    Decision,
    Method,
    QueryConstraintGraph,
    Observation,
    SpatialFrame,
    to_jsonable,
)


SCHEMA_VERSION = "dai-uav-agent-v5.0"


def _safe_fusion(
    candidates: list[Candidate],
    graph,
    config: AgenticConfig,
) -> FusionResult:
    if any(candidate.bbox is not None for candidate in candidates):
        return rank_candidates(candidates, graph, config)
    fallback = candidates[0]
    fallback.fused_score = 0.0
    fallback.competition_margin = 0.0
    return FusionResult(
        ranked=[fallback],
        final=fallback,
        evidence={
            "valid_candidate_count": 0,
            "accepted_candidate_count": 0,
            "ranked_candidate_ids": [fallback.candidate_id],
            "top_margin": 0.0,
            "top_score": 0.0,
            "rejected_candidates": {},
        },
    )


def _apply_false_repair_guard(
    fusion: FusionResult,
    initial: Candidate,
    initial_score: float,
    config: AgenticConfig,
) -> FusionResult:
    selected = fusion.final
    comparable_initial_score = initial.fused_score
    score_improvement = selected.fused_score - comparable_initial_score
    confidence_gain = (
        selected.bbox_token_confidence - initial.bbox_token_confidence
        if selected.confidence_available and initial.confidence_available
        else 0.0
    )
    evidence: list[str] = []
    if initial.bbox is None:
        evidence.append("initial_parse_failed")
    if (
        selected.confidence_available
        and initial.confidence_available
        and selected.bbox_token_confidence >= config.replacement_confidence_threshold
        and confidence_gain >= config.replacement_confidence_gain_threshold
    ):
        evidence.append("strong_token_confidence_gain")

    selected_hypothesis_id = (
        selected.hypothesis_id or selected.parent_candidate_id or selected.candidate_id
    )
    selected_hypothesis = next(
        (
            item
            for item in fusion.evidence.get("hypotheses", [])
            if item.get("hypothesis_id") == selected_hypothesis_id
        ),
        {},
    )
    supporting_verification_ids = selected_hypothesis.get(
        "supporting_verification_ids", []
    )
    if not supporting_verification_ids:
        supporting_verification_ids = [
            candidate.candidate_id
            for candidate in fusion.ranked
            if candidate.source_agent == "ZoomAgent"
            and candidate.observation.view_type == "crop_zoom"
            and candidate.observation.crop_region is not None
            and box_area(candidate.observation.crop_region)
            <= config.verification_max_crop_area
            and (candidate.hypothesis_id or candidate.parent_candidate_id)
            == selected_hypothesis_id
            and candidate.bbox is not None
            and selected.bbox is not None
            and box_iou(candidate.bbox, selected.bbox)
            >= config.replacement_cross_view_iou_threshold
            and candidate.bbox_token_confidence
            >= config.verification_confidence_threshold
        ]
    cross_view_partner_id = (
        supporting_verification_ids[0] if supporting_verification_ids else None
    )
    if selected_hypothesis.get("cross_view_supported") or cross_view_partner_id:
        evidence.append("cross_view_zoom_confirmation")

    relation_gain = selected.relation_consistency - initial.relation_consistency
    if (
        selected.relation_consistency >= config.relation_threshold
        and relation_gain >= config.replacement_constraint_gain_threshold
    ):
        evidence.append("relation_constraint_gain")
    global_gain = selected.global_constraint_score - initial.global_constraint_score
    if (
        selected.global_constraint_score >= config.global_constraint_threshold
        and global_gain >= config.replacement_constraint_gain_threshold
    ):
        evidence.append("global_constraint_gain")

    identity_iou = (
        box_iou(selected.bbox, initial.bbox)
        if selected.bbox is not None and initial.bbox is not None
        else 0.0
    )
    initial_hypothesis = next(
        (
            item
            for item in fusion.evidence.get("hypotheses", [])
            if item.get("hypothesis_id") == initial.candidate_id
        ),
        {},
    )
    selected_verification_strength = float(
        selected_hypothesis.get("verification_strength", 0.0)
    )
    initial_verification_strength = float(
        initial_hypothesis.get("verification_strength", 0.0)
    )
    verification_advantage = bool(
        selected_hypothesis.get("cross_view_supported")
        and selected_verification_strength
        >= initial_verification_strength + config.verification_advantage_margin
    )
    if verification_advantage:
        evidence.append("cross_view_verification_advantage")
    constraint_relocation_supported = bool(
        {"relation_constraint_gain", "global_constraint_gain"}.intersection(evidence)
    )
    same_identity = (
        initial.bbox is not None
        and identity_iou >= config.replacement_identity_iou_threshold
    )
    strong_confidence_support = "strong_token_confidence_gain" in evidence
    if initial.bbox is None:
        semantic_replacement_supported = bool(evidence)
    elif same_identity:
        semantic_replacement_supported = bool(
            verification_advantage
            or constraint_relocation_supported
            or strong_confidence_support
        )
    else:
        semantic_replacement_supported = bool(
            verification_advantage or constraint_relocation_supported
        )
    identity_preserved = bool(
        initial.bbox is None
        or same_identity
        or verification_advantage
        or constraint_relocation_supported
    )
    replacement_supported = (
        semantic_replacement_supported
        and (initial.bbox is None or score_improvement >= config.false_repair_margin)
        and identity_preserved
    )
    fusion.evidence.update(
        {
            "pre_guard_final_candidate_id": selected.candidate_id,
            "pre_guard_final_hypothesis_id": selected_hypothesis_id,
            "pre_perception_initial_score": initial_score,
            "comparable_initial_score": comparable_initial_score,
            "pre_guard_score_improvement": score_improvement,
            "replacement_confidence_gain": confidence_gain,
            "replacement_support_evidence": evidence,
            "cross_view_partner_id": cross_view_partner_id,
            "replacement_identity_iou": identity_iou,
            "same_identity_hypothesis": same_identity,
            "initial_verification_strength": initial_verification_strength,
            "selected_verification_strength": selected_verification_strength,
            "verification_advantage": verification_advantage,
            "replacement_identity_preserved": identity_preserved,
            "constraint_relocation_supported": constraint_relocation_supported,
            "replacement_supported": replacement_supported,
        }
    )
    if (
        config.enable_false_repair_guard
        and selected.candidate_id != initial.candidate_id
        and not replacement_supported
    ):
        ranked = [
            initial,
            *[
                candidate
                for candidate in fusion.ranked
                if candidate.candidate_id != initial.candidate_id
            ],
        ]
        fusion.ranked = ranked
        fusion.final = initial
        next_score = ranked[1].fused_score if len(ranked) > 1 else 0.0
        initial.competition_margin = max(0.0, comparable_initial_score - next_score)
        fusion.evidence["top_score"] = comparable_initial_score
        fusion.evidence["top_hypothesis_id"] = initial.candidate_id
        fusion.evidence["top_margin"] = initial.competition_margin
        fusion.evidence["ranked_candidate_ids"] = [item.candidate_id for item in ranked]
        fusion.evidence["ranked_hypothesis_ids"] = [
            item.hypothesis_id or item.parent_candidate_id or item.candidate_id
            for item in ranked
        ]
        fusion.evidence["selected_hypothesis"] = next(
            (
                item
                for item in fusion.evidence.get("hypotheses", [])
                if item.get("hypothesis_id") == initial.candidate_id
            ),
            {},
        )
        fusion.evidence["false_repair_guard_applied"] = True
        fusion.evidence["false_repair_guard_reason"] = (
            "identity_not_preserved"
            if not identity_preserved
            else (
                "missing_independent_replacement_evidence"
                if score_improvement >= config.false_repair_margin
                else "insufficient_comparable_score_gain"
            )
        )
    else:
        fusion.evidence["false_repair_guard_applied"] = False
        fusion.evidence["false_repair_guard_reason"] = None
    return fusion


class HierarchicalParentAgent:
    def __init__(
        self,
        grounder: GrounderProtocol,
        config: AgenticConfig,
    ):
        config.validate()
        self.grounder = grounder
        self.config = config
        self.router = DependencyAwareRouter()
        self.target_agent = TargetAgent()
        self.context_agent = ContextAgent()
        self.relation_reasoner = RelationAgent()
        self.zoom_agent = ZoomAgent()

    def _base_call(
        self,
        image: Image.Image,
        query: str,
        cached_initial: Candidate | None,
    ) -> tuple[Candidate, AgentCall, bool]:
        if cached_initial is not None:
            cached_initial.candidate_id = "c00"
            cached_initial.source_agent = "BaseGrounder"
            call = AgentCall(
                call_id="call_c00",
                agent="BaseGrounder",
                action="one_pass_grounding_cached",
                input={"query": query, "observation_id": "full"},
                output={"candidate": to_jsonable(cached_initial)},
                evidence={"cached": True},
                model_call=False,
                perception_call=True,
                status="cached",
                latency_ms=cached_initial.latency_ms,
            )
            return cached_initial, call, False
        observation = Observation(
            observation_id="full",
            view_type="full_image",
            preserves_context=True,
        )
        candidate = self.grounder.ground(
            image,
            query,
            "c00",
            "BaseGrounder",
            observation,
        )
        call = AgentCall(
            call_id="call_c00",
            agent="BaseGrounder",
            action="one_pass_grounding",
            input={"query": query, "observation_id": "full"},
            output={"candidate": to_jsonable(candidate)},
            evidence={
                "parse_ok": candidate.parse_ok,
                "bbox_token_confidence": candidate.bbox_token_confidence,
                "bbox_token_count": candidate.bbox_token_count,
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return candidate, call, True

    def _generic_parent_rerun(
        self,
        image: Image.Image,
        query: str,
        candidate_id: str,
        contrast: float,
        source: str,
    ) -> tuple[Candidate, AgentCall]:
        transformed = ImageEnhance.Contrast(image).enhance(contrast)
        observation = Observation(
            observation_id=f"contrast_{contrast:.2f}",
            view_type="transformed_observation",
            transform=f"contrast_{contrast:.2f}",
            preserves_context=True,
        )
        candidate = self.grounder.ground(
            transformed,
            query,
            candidate_id,
            source,
            observation,
        )
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=source,
            action="generic_transformed_view_rerun",
            input={"query": query, "observation": to_jsonable(observation)},
            output={"candidate": to_jsonable(candidate)},
            evidence={
                "parse_ok": candidate.parse_ok,
                "bbox_token_confidence": candidate.bbox_token_confidence,
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return candidate, call

    @staticmethod
    def _next_candidate_id(candidates: list[Candidate]) -> str:
        return f"c{len(candidates):02d}"

    def _relation_call(
        self,
        context: AgentContext,
        calls: list[AgentCall],
    ) -> dict[str, Any]:
        result = self.relation_reasoner.run(context)
        result.call.call_id = (
            f"call_relation_{sum(call.agent == 'RelationAgent' for call in calls)}"
        )
        calls.append(result.call)
        return result.evidence

    def _confirmed_diagnosis(
        self,
        initial: Candidate,
        candidates: list[Candidate],
        context_candidates: list[Candidate],
        graph,
        routing: RoutingPlan,
        fusion: FusionResult,
        relation_evidence: dict[str, Any],
        attempted: set[str],
    ) -> tuple[list[str], list[str]]:
        diagnosis: list[str] = []
        unresolved: list[str] = []
        if initial.bbox is None:
            diagnosis.append("parse_failed")
        if (
            initial.confidence_available
            and initial.bbox_token_confidence < self.config.confidence_threshold
        ):
            diagnosis.append("low_confidence")
        if "shape_risk" in routing.preliminary_suspicions:
            diagnosis.append("bbox_too_coarse")
        if "small_object_risk" in routing.preliminary_suspicions:
            diagnosis.append("small_object_uncertain")

        target_outputs = [
            candidate
            for candidate in candidates
            if candidate.source_agent == "TargetAgent"
        ]
        if "target" in attempted and not any(
            item.bbox is not None for item in target_outputs
        ):
            diagnosis.append("target_missing")
            unresolved.append("target_missing")
        if (
            graph.has_context
            and "context" in attempted
            and not any(candidate.bbox is not None for candidate in context_candidates)
        ):
            diagnosis.append("context_missing")
            unresolved.append("context_missing")

        valid_candidates = [
            candidate for candidate in candidates if candidate.bbox is not None
        ]
        best_relation = max(
            (candidate.relation_consistency for candidate in valid_candidates),
            default=0.0,
        )
        relation_verified = "relation" in attempted
        if graph.has_relation and not relation_verified:
            unresolved.append("relation_not_verified")
        if (
            graph.has_relation
            and relation_verified
            and best_relation < self.config.relation_threshold
        ):
            diagnosis.append("relation_wrong")
            unresolved.append("relation_wrong")
        best_global = max(
            (candidate.global_constraint_score for candidate in valid_candidates),
            default=0.0,
        )
        if (
            SpatialFrame.GLOBAL_ABSOLUTE in graph.spatial_frames
            or SpatialFrame.GLOBAL_ORDER in graph.spatial_frames
        ) and best_global < self.config.global_constraint_threshold:
            diagnosis.append("global_position_unresolved")
            unresolved.append("global_position_unresolved")
        for item in relation_evidence.get("unresolved", []):
            if item == "relation_missing_context_evidence":
                diagnosis.append("context_missing")
                unresolved.append("context_missing")
                continue
            if item == "orientation_unresolved":
                diagnosis.append("orientation_unresolved")
                unresolved.append("orientation_unresolved")
        if SpatialFrame.TEMPORAL_EVENT in graph.spatial_frames:
            diagnosis.append("temporal_event_unresolved")
            unresolved.append("temporal_event_unresolved")

        if len(fusion.ranked) > 1:
            first, second = fusion.ranked[:2]
            separated = (
                first.bbox is not None
                and second.bbox is not None
                and box_iou(first.bbox, second.bbox)
                < self.config.competition_iou_threshold
            )
            if (
                separated
                and first.fused_score - second.fused_score
                < self.config.competition_margin_threshold
            ):
                diagnosis.append("ambiguous_candidates")
                unresolved.append("ambiguous_candidates")

        if fusion.final.bbox is None:
            unresolved.append("observation_insufficient")
        elif fusion.final.fused_score < self.config.final_confidence_threshold:
            unresolved.append("low_final_confidence")
        if (
            "small_object_uncertain" in diagnosis
            and "zoom" not in attempted
            and self.config.max_child_perception_calls > 0
        ):
            unresolved.append("small_object_uncertain")
        if (
            "bbox_too_coarse" in diagnosis
            and "zoom" in attempted
            and not any(
                candidate.source_agent == "ZoomAgent" and candidate.accepted_by_guard
                for candidate in candidates
            )
        ):
            unresolved.append("bbox_too_coarse")
        return list(dict.fromkeys(diagnosis)), list(dict.fromkeys(unresolved))

    def run(
        self,
        sample_id: str,
        image: Image.Image,
        query: str,
        sample_class: str = "unknown",
        cached_initial: Candidate | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        graph = parse_query_constraints(query)
        if not self.config.enable_constraint_graph:
            graph = QueryConstraintGraph(
                original=query,
                target=graph.target,
                attributes=graph.attributes,
                spatial_frames=[SpatialFrame.LOCAL_ATTRIBUTE],
                local_target_query=graph.local_target_query,
                zoom_query=query,
                parser_version="constraint-graph-disabled-ablation",
            )
        calls: list[AgentCall] = []
        candidates: list[Candidate] = []
        context_candidates: list[Candidate] = []
        attempted: set[str] = set()
        relation_evidence: dict[str, Any] = {}

        initial, base_call, base_executed = self._base_call(
            image, query, cached_initial
        )
        calls.append(base_call)
        candidates.append(initial)
        initial_only_fusion = _safe_fusion(candidates, graph, self.config)
        initial_score = initial_only_fusion.final.fused_score

        routing = self.router.plan(initial, graph, self.config)
        if self.config.method == Method.ONE_PASS:
            fusion = initial_only_fusion
        elif self.config.method == Method.CONFIDENCE_GATED:
            gate = bool(
                {"parse_risk", "low_token_confidence"}.intersection(
                    routing.preliminary_suspicions
                )
            )
            if gate and self.config.max_child_perception_calls > 0:
                candidate, call = self._generic_parent_rerun(
                    image,
                    query,
                    self._next_candidate_id(candidates),
                    1.10,
                    "ConfidenceGatedRerun",
                )
                candidates.append(candidate)
                calls.append(call)
            fusion = _safe_fusion(candidates, graph, self.config)
        elif self.config.method == Method.PARENT_ONLY:
            verify = bool(routing.preliminary_suspicions or graph.is_position_sensitive)
            if verify and self.config.max_child_perception_calls > 0:
                candidate, call = self._generic_parent_rerun(
                    image,
                    query,
                    self._next_candidate_id(candidates),
                    1.15,
                    "ParentVerification",
                )
                candidates.append(candidate)
                calls.append(call)
            fusion = _safe_fusion(candidates, graph, self.config)
        else:
            budget = self.config.max_child_perception_calls
            specialized_perception_agents = {
                "TargetAgent",
                "ContextAgent",
                "ZoomAgent",
            }

            def budget_used() -> int:
                return sum(
                    call.model_call
                    and call.perception_call
                    and call.agent in specialized_perception_agents
                    for call in calls
                )

            def make_context() -> AgentContext:
                return AgentContext(
                    image=image,
                    graph=graph,
                    grounder=self.grounder,
                    config=self.config,
                    target_candidates=candidates,
                    context_candidates=context_candidates,
                )

            def append_target_result(result) -> None:
                existing_roots = [
                    candidate
                    for candidate in candidates
                    if candidate.parent_candidate_id is None
                    and candidate.bbox is not None
                ]
                for candidate in result.candidates:
                    if candidate.bbox is not None and any(
                        box_iou(candidate.bbox, existing.bbox) >= 0.995
                        for existing in existing_roots
                    ):
                        candidate.accepted_by_guard = False
                        candidate.rejection_reasons.append("duplicate_hypothesis")
                    candidates.append(candidate)
                calls.append(result.call)
                attempted.add("target")

            def refresh_relation_and_fusion() -> FusionResult:
                nonlocal relation_evidence
                if routing.run_relation_reasoner:
                    relation_evidence = self._relation_call(make_context(), calls)
                    attempted.add("relation")
                return _safe_fusion(candidates, graph, self.config)

            planned_actions = set(routing.perception_actions)
            if (
                "target" in planned_actions
                and budget_used() < budget
                and "target" not in self.config.disabled_agents
            ):
                result = self.target_agent.run(
                    make_context(), self._next_candidate_id(candidates)
                )
                append_target_result(result)

            if (
                "context" in planned_actions
                and budget_used() < budget
                and "context" not in self.config.disabled_agents
            ):
                context_id = f"x{len(context_candidates):02d}"
                result = self.context_agent.run(make_context(), context_id)
                context_candidates.extend(result.candidates)
                calls.append(result.call)
                attempted.add("context")

            preliminary_fusion = refresh_relation_and_fusion()
            search_risk = (
                bool(routing.preliminary_suspicions) or graph.is_position_sensitive
            )

            def is_verified(seed: Candidate) -> bool:
                return any(
                    candidate.source_agent == "ZoomAgent"
                    and candidate.parent_candidate_id == seed.candidate_id
                    and candidate.bbox is not None
                    and candidate.accepted_by_guard
                    for candidate in candidates
                )

            def best_unverified_diverse_target() -> Candidate | None:
                alternatives = [
                    candidate
                    for candidate in candidates
                    if candidate.source_agent == "TargetAgent"
                    and candidate.parent_candidate_id is None
                    and candidate.bbox is not None
                    and candidate.accepted_by_guard
                    and not is_verified(candidate)
                    and (
                        initial.bbox is None
                        or box_iou(candidate.bbox, initial.bbox)
                        < self.config.target_diversity_iou_threshold
                    )
                ]
                return (
                    max(alternatives, key=lambda item: item.fused_score)
                    if alternatives
                    else None
                )

            def verify_hypothesis(seed: Candidate) -> None:
                nonlocal preliminary_fusion
                if (
                    budget_used() >= budget
                    or "zoom" in self.config.disabled_agents
                    or seed.bbox is None
                    or is_verified(seed)
                ):
                    return
                result = self.zoom_agent.run(
                    make_context(),
                    seed,
                    self._next_candidate_id(candidates),
                )
                candidates.extend(result.candidates)
                calls.append(result.call)
                attempted.add("zoom")
                preliminary_fusion = refresh_relation_and_fusion()

            # Verification is symmetric: the initial hypothesis receives the same
            # semantic transformed-view test as any proposed replacement.
            if (
                self.config.method == Method.HIERARCHICAL
                and self.config.enable_symmetric_verification
            ):
                verify_hypothesis(initial)

            alternative = best_unverified_diverse_target()
            if alternative is not None:
                verify_hypothesis(alternative)

            explore_competition = bool(
                search_risk or self.config.competition_probe_mode == "always"
            )
            if (
                explore_competition
                and budget_used() < budget
                and "target" not in self.config.disabled_agents
            ):
                remaining = min(
                    budget - budget_used(), self.config.max_target_tile_calls
                )
                regions = self.target_agent.proposal_regions(
                    make_context(), initial, remaining
                )
                for region in regions:
                    if budget_used() >= budget:
                        break
                    result = self.target_agent.run_transformed_view(
                        make_context(),
                        self._next_candidate_id(candidates),
                        region,
                    )
                    append_target_result(result)
                    preliminary_fusion = refresh_relation_and_fusion()
                    alternative = best_unverified_diverse_target()
                    if alternative is not None and budget_used() < budget:
                        verify_hypothesis(alternative)

            fusion = _safe_fusion(candidates, graph, self.config)

        fusion = _apply_false_repair_guard(fusion, initial, initial_score, self.config)
        diagnosis, unresolved = self._confirmed_diagnosis(
            initial,
            candidates,
            context_candidates,
            graph,
            routing,
            fusion,
            relation_evidence,
            attempted,
        )
        information_gain = fusion.final.fused_score - initial_score
        if self.config.method in {
            Method.ONE_PASS,
            Method.CONFIDENCE_GATED,
            Method.PARENT_ONLY,
        }:
            should_escalate = False
        else:
            should_escalate = bool(unresolved) and self.config.enable_escalation
            if (
                attempted
                and information_gain < self.config.information_gain_threshold
                and fusion.final.fused_score < self.config.final_confidence_threshold
            ):
                should_escalate = self.config.enable_escalation
                if "no_information_gain" not in unresolved:
                    unresolved.append("no_information_gain")

        if should_escalate:
            decision = Decision.ESCALATE
            stop_reason = (
                "perception_budget_exhausted"
                if len(
                    [
                        call
                        for call in calls
                        if call.perception_call
                        and call.agent in {"TargetAgent", "ContextAgent", "ZoomAgent"}
                    ]
                )
                >= self.config.max_child_perception_calls
                else "observation_insufficient"
            )
        elif fusion.final.candidate_id == initial.candidate_id:
            decision = Decision.ACCEPT
            stop_reason = "initial_passed_unsupervised_gate"
        else:
            decision = Decision.REFINE
            stop_reason = "refined_candidate_selected"

        feedback = None
        if decision == Decision.ESCALATE:
            feedback, feedback_call = generate_feedback(
                self.config.feedback_mode,
                self.grounder,
                unresolved or diagnosis,
                graph,
                fusion.final.bbox,
                calls_used=sum(call.perception_call for call in calls),
                max_calls=1 + self.config.max_child_perception_calls,
                top_score=fusion.final.fused_score,
                top_margin=fusion.final.competition_margin,
            )
            if feedback_call:
                calls.append(feedback_call)

        specialized_units = {
            "TargetAgent",
            "ContextAgent",
            "RelationAgent",
            "ZoomAgent",
        }
        unit_calls = [
            call
            for call in calls
            if call.agent in specialized_units and call.status != "skipped"
        ]
        unit_perception_calls = [call for call in unit_calls if call.perception_call]
        if (
            self.config.method in {Method.HIERARCHICAL, Method.STATIC_ALL}
            and len(unit_perception_calls) > self.config.max_child_perception_calls
        ):
            raise RuntimeError(
                "Specialized perception budget exceeded: "
                f"{len(unit_perception_calls)} > {self.config.max_child_perception_calls}"
            )
        perception_calls = [call for call in calls if call.perception_call]
        executed_perception_calls = [
            call for call in perception_calls if call.model_call
        ]
        feedback_calls = [call for call in calls if call.agent == "FeedbackGenerator"]
        wall_latency_ms = (time.perf_counter() - started) * 1000
        initial_latency_ms = max(0.0, float(initial.latency_ms))
        if base_executed:
            end_to_end_latency_ms = wall_latency_ms
            incremental_latency_ms = max(0.0, wall_latency_ms - initial_latency_ms)
        else:
            incremental_latency_ms = wall_latency_ms
            end_to_end_latency_ms = initial_latency_ms + incremental_latency_ms
        latency_available = initial_latency_ms > 0.0
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "class": sample_class or "unknown",
            "method": self.config.method.value,
            "inference": {
                "query": query,
                "constraint_graph": to_jsonable(graph),
                "initial_candidate": to_jsonable(initial),
                "preliminary_suspicions": routing.preliminary_suspicions,
                "routing_plan": {
                    "perception_actions": routing.perception_actions,
                    "run_relation_reasoner": routing.run_relation_reasoner,
                    "action_utilities": routing.action_utilities,
                    "rationale": routing.rationale,
                    "executed_perception_actions": [
                        f"{call.agent}:{call.action}" for call in unit_perception_calls
                    ],
                    "budget_used": len(unit_perception_calls),
                    "budget_limit": self.config.max_child_perception_calls,
                },
                "diagnosis": diagnosis,
                "confirmed_diagnosis": diagnosis,
                "action_trace": [to_jsonable(call) for call in calls],
                "unresolved_constraints": unresolved,
                "verification_evidence": {
                    "initial_parse_validity": float(initial.bbox is not None),
                    "initial_bbox_token_confidence": initial.bbox_token_confidence,
                    "initial_confidence_available": initial.confidence_available,
                    "initial_fused_confidence": initial_score,
                    "initial_box_area": box_area(initial.bbox),
                    "initial_box_plausibility": box_plausibility(initial.bbox),
                    "relation": relation_evidence,
                    "fusion": fusion.evidence,
                    "information_gain": information_gain,
                },
                "agent_calls": [to_jsonable(call) for call in calls],
                "unit_calls": [to_jsonable(call) for call in unit_calls],
                "child_calls": [to_jsonable(call) for call in unit_calls],
                "target_candidates": [
                    to_jsonable(candidate) for candidate in candidates
                ],
                "context_candidates": [
                    to_jsonable(candidate) for candidate in context_candidates
                ],
                "hypothesis_clusters": fusion.evidence.get("hypotheses", []),
                "final_hypothesis_id": fusion.evidence.get("top_hypothesis_id"),
                "final_candidate_id": fusion.final.candidate_id,
                "final_bbox": fusion.final.bbox,
                "confidence": fusion.final.fused_score,
                "decision": decision.value,
                "stop_reason": stop_reason,
                "human_feedback": to_jsonable(feedback) if feedback else None,
                "question_e_used": False,
            },
            "cost": {
                "perception_calls": len(perception_calls),
                "executed_perception_calls": len(executed_perception_calls),
                "specialized_unit_perception_calls": len(unit_perception_calls),
                "specialized_unit_calls": len(unit_calls),
                "specialized_model_calls": len(unit_perception_calls),
                "target_search_calls": sum(
                    call.agent == "TargetAgent" and call.perception_call
                    for call in unit_calls
                ),
                "context_search_calls": sum(
                    call.agent == "ContextAgent" and call.perception_call
                    for call in unit_calls
                ),
                "zoom_verification_calls": sum(
                    call.agent == "ZoomAgent" and call.perception_call
                    for call in unit_calls
                ),
                "child_perception_calls": len(unit_perception_calls),
                "child_agent_calls": len(unit_calls),
                "relation_calls": sum(call.agent == "RelationAgent" for call in calls),
                "feedback_llm_calls": sum(
                    call.agent == "FeedbackGenerator" and call.model_call
                    for call in calls
                ),
                "feedback_calls": len(feedback_calls),
                "dispatch": bool(unit_calls) or len(perception_calls) > 1,
                "cached_initial": not base_executed,
                "initial_latency_ms": initial_latency_ms,
                "incremental_agent_latency_ms": incremental_latency_ms,
                "end_to_end_latency_ms": end_to_end_latency_ms,
                "latency_available": latency_available,
                "latency_ms": end_to_end_latency_ms,
            },
        }
