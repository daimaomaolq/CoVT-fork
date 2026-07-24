from __future__ import annotations

import json
import re
import time
from typing import Any

from .geometry import expand_box
from .grounder import GrounderProtocol
from .schema import AgentCall, EscalationFeedback, QueryConstraintGraph, SpatialFrame, to_jsonable


ALLOWED_ACTIONS = {
    "higher_resolution_centered_view",
    "lower_altitude_context_preserving_view",
    "wider_context_view",
    "lateral_reposition",
    "oblique_context_preserving_view",
    "recenter_region",
    "request_temporal_observation",
    "manual_review",
}


def select_observation_action(
    diagnosis: list[str],
    graph: QueryConstraintGraph,
) -> tuple[str, str, str]:
    diagnosis_set = set(diagnosis)
    if (
        "temporal_event_unresolved" in diagnosis_set
        or SpatialFrame.TEMPORAL_EVENT in graph.spatial_frames
    ):
        return (
            "request_temporal_observation",
            "A single still image cannot verify the queried event or motion state.",
            "short temporal evidence showing target motion or state change",
        )
    if (
        "orientation_unresolved" in diagnosis_set
        or SpatialFrame.ORIENTATION_DEPENDENT in graph.spatial_frames
    ):
        return (
            "oblique_context_preserving_view",
            "The current overhead view does not expose reliable orientation evidence.",
            "target orientation and its relation to the reference context",
        )
    if "ambiguous_candidates" in diagnosis_set:
        return (
            "lateral_reposition",
            "Several spatially separated candidates remain similarly plausible.",
            "viewpoint-dependent evidence that separates the competing candidates",
        )
    if "context_missing" in diagnosis_set or "relation_wrong" in diagnosis_set:
        return (
            "wider_context_view",
            "The target-context relation cannot be verified from the available region.",
            "a joint view containing both the target and its reference context",
        )
    if "small_object_uncertain" in diagnosis_set:
        if SpatialFrame.OBJECT_RELATIVE in graph.spatial_frames:
            return (
                "lower_altitude_context_preserving_view",
                "The target is too small, but the reference context must remain visible.",
                "higher-resolution target appearance together with its context",
            )
        return (
            "higher_resolution_centered_view",
            "The target is too small for reliable localization.",
            "higher-resolution target appearance",
        )
    if "bbox_too_coarse" in diagnosis_set:
        return (
            "higher_resolution_centered_view",
            "The current region is too coarse for a precise target box.",
            "a sharper, centered target boundary",
        )
    if "target_missing" in diagnosis_set or "parse_failed" in diagnosis_set:
        return (
            "wider_context_view",
            "No reliable target candidate was obtained from the current observation.",
            "a wider search view with the expected scene context",
        )
    return (
        "manual_review",
        "The remaining uncertainty is not resolved by the available single-image actions.",
        "operator confirmation of the most plausible candidate",
    )


def template_feedback(
    diagnosis: list[str],
    graph: QueryConstraintGraph,
    final_bbox: list[float] | None,
) -> EscalationFeedback:
    action, reason, evidence = select_observation_action(diagnosis, graph)
    region = expand_box(final_bbox, 1.5) if final_bbox is not None else None
    constraints = ", ".join(diagnosis) if diagnosis else "unresolved reliability"
    return EscalationFeedback(
        status="need_additional_observation",
        uncertainty_summary=f"Unresolved constraints: {constraints}.",
        recommended_action=action,
        region_of_interest=region,
        reason=reason,
        expected_evidence=evidence,
        operator_message=(
            f"Current grounding is not reliable enough. Recommended next observation: "
            f"{action.replace('_', ' ')}. Preserve the query-relevant context."
        ),
        generator="deterministic_template",
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None


def generate_feedback(
    mode: str,
    grounder: GrounderProtocol,
    diagnosis: list[str],
    graph: QueryConstraintGraph,
    final_bbox: list[float] | None,
    calls_used: int,
    max_calls: int,
    top_score: float,
    top_margin: float,
) -> tuple[EscalationFeedback | None, AgentCall | None]:
    if mode == "off":
        return None, None
    fallback = template_feedback(diagnosis, graph, final_bbox)
    if mode == "template":
        call = AgentCall(
            call_id="call_feedback",
            agent="FeedbackGenerator",
            action="template_feedback",
            input={
                "diagnosis": diagnosis,
                "spatial_frames": [frame.value for frame in graph.spatial_frames],
            },
            output=to_jsonable(fallback),
            model_call=False,
            perception_call=False,
        )
        return fallback, call

    allowed_action = fallback.recommended_action
    state = {
        "original_query": graph.original,
        "query_frames": [frame.value for frame in graph.spatial_frames],
        "calls_used": calls_used,
        "max_calls": max_calls,
        "unresolved_constraints": diagnosis,
        "top_score": round(top_score, 4),
        "top_margin": round(top_margin, 4),
        "current_region_of_interest": fallback.region_of_interest,
        "selected_action": allowed_action,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
    }
    prompt = (
        "You are a bounded UAV perception feedback narrator. Convert the provided "
        "structured state into concise operator-facing feedback. Do not decide a new "
        "action, do not claim the current box is certainly wrong, do not invent facts, "
        "and do not provide numeric flight altitude, distance, speed, or control commands. "
        "The recommended_action must equal selected_action. Output JSON only with keys: "
        "status, uncertainty_summary, recommended_action, region_of_interest, reason, "
        "expected_evidence, operator_message.\nSTATE:\n"
        + json.dumps(state, ensure_ascii=False)
    )
    started = time.perf_counter()
    raw = ""
    parsed = None
    error = None
    try:
        raw = grounder.generate_base_text(prompt, max_new_tokens=256)
        parsed = _extract_json(raw)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    valid = (
        parsed is not None
        and parsed.get("recommended_action") == allowed_action
        and parsed.get("recommended_action") in ALLOWED_ACTIONS
        and all(
            key in parsed
            for key in (
                "status", "uncertainty_summary", "reason",
                "expected_evidence", "operator_message",
            )
        )
    )
    if valid:
        feedback = EscalationFeedback(
            status=str(parsed["status"]),
            uncertainty_summary=str(parsed["uncertainty_summary"]),
            recommended_action=str(parsed["recommended_action"]),
            region_of_interest=parsed.get("region_of_interest"),
            reason=str(parsed["reason"]),
            expected_evidence=str(parsed["expected_evidence"]),
            operator_message=str(parsed["operator_message"]),
            generator="base_model_adapter_disabled",
        )
    else:
        feedback = fallback
        feedback.generator = "base_model_fallback_template"
        feedback.fallback_used = True
    call = AgentCall(
        call_id="call_feedback",
        agent="FeedbackGenerator",
        action="base_model_narration",
        input=state,
        output=to_jsonable(feedback),
        evidence={
            "raw_output": raw,
            "schema_valid": bool(valid),
            "error": error,
        },
        model_call=True,
        perception_call=False,
        latency_ms=(time.perf_counter() - started) * 1000,
        status="completed" if valid else "fallback",
    )
    return feedback, call
