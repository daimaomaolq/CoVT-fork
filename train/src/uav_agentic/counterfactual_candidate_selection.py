from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .frozen_candidate_selection import FrozenBaseVisualVerifier, clamp01
from .geometry import box_iou


COUNTERFACTUAL_SCHEMA_VERSION = "dai-uav-agent-v4.3-counterfactual"


@dataclass(frozen=True)
class CounterfactualConfig:
    max_alternatives: int = 1
    duplicate_iou_threshold: float = 0.92
    maximum_initial_overlap: float = 0.85
    minimum_independent_gain: float = 0.0
    require_parent_unresolved: bool = True
    require_independent_support: bool = True
    first_crop_scale: float = 3.0
    second_crop_scale: float = 4.0

    def validate(self) -> None:
        if self.max_alternatives < 1:
            raise ValueError("max_alternatives must be at least one")
        for name in (
            "duplicate_iou_threshold",
            "maximum_initial_overlap",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.first_crop_scale <= 1.0 or self.second_crop_scale <= 1.0:
            raise ValueError("crop scales must be greater than one")


def candidate_independent_score(candidate: dict[str, Any]) -> float:
    """Score from pre-existing non-verifier evidence only."""
    token = (
        clamp01(candidate.get("bbox_token_confidence"), 0.5)
        if candidate.get("confidence_available", True)
        else 0.5
    )
    return (
        0.40 * token
        + 0.20 * clamp01(candidate.get("target_consistency"), 0.5)
        + 0.15 * clamp01(candidate.get("relation_consistency"), 0.5)
        + 0.15 * clamp01(candidate.get("global_constraint_score"), 0.5)
        + 0.10 * clamp01(candidate.get("box_plausibility"), 0.5)
    )


def _cluster_support(inference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cluster in inference.get("hypothesis_clusters", []):
        supported = bool(
            cluster.get("cross_view_supported")
            or cluster.get("independent_verification_ids")
            or cluster.get("supporting_verification_ids")
        )
        for candidate_id in cluster.get("member_candidate_ids", []):
            result[str(candidate_id)] = {
                "supported": supported,
                "cross_view_iou": clamp01(cluster.get("cross_view_iou"), 0.0),
                "hypothesis_id": str(cluster.get("hypothesis_id") or ""),
                "member_count": len(cluster.get("member_candidate_ids", [])),
            }
    return result


def candidate_support(
    candidate: dict[str, Any],
    support_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    cluster = copy.deepcopy(support_map.get(candidate_id, {}))
    zoom_observation = bool(
        candidate.get("source_agent") == "ZoomAgent"
        and candidate.get("parent_candidate_id")
    )
    cluster["zoom_observation"] = zoom_observation
    cluster["supported"] = bool(cluster.get("supported") or zoom_observation)
    return cluster


def _candidate_priority(
    candidate: dict[str, Any],
    support: dict[str, Any],
    initial: dict[str, Any],
) -> tuple[float, float, float, float]:
    return (
        float(bool(support.get("supported"))),
        float(bool(support.get("zoom_observation"))),
        candidate_independent_score(candidate),
        1.0 - box_iou(candidate.get("bbox"), initial.get("bbox")),
    )


def eligible_alternatives(
    inference: dict[str, Any],
    config: CounterfactualConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Route alternatives without accepting verifier or GT information."""
    config.validate()
    initial = inference["initial_candidate"]
    initial_id = str(initial.get("candidate_id") or "")
    initial_score = candidate_independent_score(initial)
    support_map = _cluster_support(inference)
    audit: dict[str, Any] = {
        "parent_decision": str(inference.get("decision") or "unknown"),
        "initial_candidate_id": initial_id,
        "initial_independent_score": initial_score,
        "require_parent_unresolved": config.require_parent_unresolved,
        "require_independent_support": config.require_independent_support,
        "rejected": [],
    }
    if config.require_parent_unresolved and inference.get("decision") == "accept":
        audit["route_reason"] = "parent_already_accepted"
        return [], audit

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in inference.get("target_candidates", []):
        candidate = copy.deepcopy(raw)
        candidate_id = str(candidate.get("candidate_id") or "")
        reasons: list[str] = []
        if not candidate_id or candidate_id == initial_id:
            continue
        if candidate.get("bbox") is None or not candidate.get("parse_ok", True):
            reasons.append("invalid_bbox")
        overlap = box_iou(candidate.get("bbox"), initial.get("bbox"))
        if overlap > config.maximum_initial_overlap:
            reasons.append("insufficient_relocation")
        score = candidate_independent_score(candidate)
        if score - initial_score < config.minimum_independent_gain:
            reasons.append("lower_independent_score")
        support = candidate_support(candidate, support_map)
        if config.require_independent_support and not support.get("supported"):
            reasons.append("no_independent_support")
        if reasons:
            audit["rejected"].append(
                {"candidate_id": candidate_id, "reasons": reasons}
            )
            continue
        candidate["counterfactual_independent_score"] = score
        candidate["counterfactual_support"] = support
        candidate["counterfactual_initial_iou"] = overlap
        candidates.append((candidate, support))

    candidates.sort(
        key=lambda item: _candidate_priority(item[0], item[1], initial),
        reverse=True,
    )
    deduplicated: list[dict[str, Any]] = []
    for candidate, _ in candidates:
        if any(
            box_iou(candidate["bbox"], kept["bbox"])
            >= config.duplicate_iou_threshold
            for kept in deduplicated
        ):
            audit["rejected"].append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reasons": ["duplicate_eligible_alternative"],
                }
            )
            continue
        deduplicated.append(candidate)
        if len(deduplicated) >= config.max_alternatives:
            break
    audit["route_reason"] = (
        "eligible_counterfactual_pairs"
        if deduplicated
        else "no_independently_supported_alternative"
    )
    audit["eligible_candidate_ids"] = [
        str(candidate.get("candidate_id")) for candidate in deduplicated
    ]
    audit["eligible_candidates"] = [
        {
            "candidate_id": str(candidate.get("candidate_id")),
            "source_agent": candidate.get("source_agent"),
            "independent_score": candidate["counterfactual_independent_score"],
            "initial_iou": candidate["counterfactual_initial_iou"],
            "support": candidate["counterfactual_support"],
        }
        for candidate in deduplicated
    ]
    return deduplicated, audit


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _expanded_crop(
    image: Image.Image,
    bbox: list[float],
    scale: float,
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_width = min(1.0, max((x2 - x1) * scale, 0.16))
    crop_height = min(1.0, max((y2 - y1) * scale, 0.16))
    left = max(0.0, min(1.0 - crop_width, center_x - crop_width / 2.0))
    top = max(0.0, min(1.0 - crop_height, center_y - crop_height / 2.0))
    right = left + crop_width
    bottom = top + crop_height
    return image.crop(
        (
            round(left * width),
            round(top * height),
            max(round(right * width), round(left * width) + 1),
            max(round(bottom * height), round(top * height) + 1),
        )
    )


def render_counterfactual_sheet(
    image: Image.Image,
    initial: dict[str, Any],
    alternative: dict[str, Any],
    swap: bool,
    crop_scale: float,
) -> tuple[Image.Image, dict[str, str]]:
    """Render a global frame plus counterbalanced candidate-centric crops."""
    original = image.convert("RGB")
    width = 1008
    top_height = max(378, round(width * original.height / original.width))
    top_height = min(top_height, 630)
    panel_height = 448
    banner_height = 52
    canvas = Image.new("RGB", (width, top_height + panel_height), "white")
    top = original.resize((width, top_height), Image.Resampling.LANCZOS)
    canvas.paste(top, (0, 0))

    assignments = (
        {"A": alternative, "B": initial}
        if swap
        else {"A": initial, "B": alternative}
    )
    colors = {"A": "#00a878", "B": "#ff7a00"}
    mapping = {
        label: str(candidate.get("candidate_id"))
        for label, candidate in assignments.items()
    }
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(34)
    small_font = _load_font(19)
    line_width = 7
    for label, candidate in assignments.items():
        x1, y1, x2, y2 = [float(value) for value in candidate["bbox"][:4]]
        pixels = (
            round(x1 * width),
            round(y1 * top_height),
            round(x2 * width),
            round(y2 * top_height),
        )
        draw.rectangle(pixels, outline=colors[label], width=line_width)
        tag_width = 62
        tag_height = 46
        tag_x = max(0, min(width - tag_width, pixels[0]))
        tag_y = max(0, pixels[1] - tag_height)
        draw.rectangle(
            (tag_x, tag_y, tag_x + tag_width, tag_y + tag_height),
            fill=colors[label],
        )
        draw.text((tag_x + 18, tag_y + 5), label, fill="white", font=label_font)

    panel_width = width // 2
    for index, label in enumerate(("A", "B")):
        candidate = assignments[label]
        crop = _expanded_crop(original, candidate["bbox"], crop_scale)
        fitted = ImageOps.fit(
            crop,
            (panel_width, panel_height - banner_height),
            method=Image.Resampling.LANCZOS,
        )
        left = index * panel_width
        canvas.paste(fitted, (left, top_height + banner_height))
        draw.rectangle(
            (left, top_height, left + panel_width - 1, top_height + panel_height - 1),
            outline=colors[label],
            width=line_width,
        )
        draw.rectangle(
            (left, top_height, left + panel_width, top_height + banner_height),
            fill=colors[label],
        )
        bbox_text = ",".join(
            f"{float(value):.3f}" for value in candidate["bbox"][:4]
        )
        draw.text(
            (left + 14, top_height + 6),
            f"{label}  global bbox [{bbox_text}]",
            fill="white",
            font=small_font,
        )
    return canvas, mapping


def counterfactual_prompt(
    inference: dict[str, Any],
    mapping: dict[str, str],
    candidates: dict[str, dict[str, Any]],
) -> str:
    graph = inference.get("constraint_graph", {})
    candidate_lines = []
    for label in ("A", "B"):
        candidate = candidates[mapping[label]]
        candidate_lines.append(
            f"Candidate {label}: global bbox="
            f"{[round(float(v), 4) for v in candidate['bbox'][:4]]}."
        )
    return (
        "Compare two candidate boxes for one UAV referring-grounding query. "
        "The TOP panel is the original full image and preserves global position, "
        "context, and relations. The LOWER panels are enlarged crops of the same "
        "single image and are only for inspecting target identity and attributes. "
        "They are not new viewpoints. Judge the visible object inside each marked "
        "box, not the crop background. Prefer neither label by default.\n"
        f"Original query: {inference.get('query', '')}\n"
        f"Target: {graph.get('target') or 'unspecified'}\n"
        f"Attributes: {', '.join(graph.get('attributes', [])) or 'none'}\n"
        f"Context: {graph.get('context') or 'none'}\n"
        f"Relations: {', '.join(graph.get('relations', [])) or 'none'}\n"
        f"Global position: {graph.get('global_position') or 'none'}\n"
        f"Ordinal: {graph.get('ordinal_constraint') or 'none'}\n"
        + "\n".join(candidate_lines)
        + "\nRespond with exactly one token: <A> if A is better, <B> if B is "
        "better, or <X> when the image is insufficient or they are tied."
    )


def parse_counterfactual_choice(raw: str) -> str | None:
    original = str(raw or "").strip()
    if not original:
        return None
    try:
        parsed = json.loads(original)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        original = str(parsed.get("selected") or parsed.get("choice") or "")
    text = original.upper()
    exact = re.fullmatch(r"\s*[<\[\(\"']?([ABX])[>\]\)\"']?[.!]?\s*", text)
    if exact:
        return exact.group(1)
    tokens = re.findall(r"<([ABX])>", text)
    return tokens[0] if len(tokens) == 1 else None


def verify_counterfactual_candidates(
    verifier: FrozenBaseVisualVerifier,
    image: Image.Image,
    row: dict[str, Any],
    config: CounterfactualConfig,
) -> dict[str, Any]:
    """Counterbalanced pairwise verification without GT or question_e."""
    inference = copy.deepcopy(row["inference"])
    if inference.get("question_e_used") is not False:
        raise ValueError("Unsafe trace: question_e_used must be explicitly false")
    alternatives, route = eligible_alternatives(inference, config)
    initial = inference["initial_candidate"]
    if not alternatives:
        return {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "sample_id": row["sample_id"],
            "status": "skipped",
            "abstained": True,
            "winner_candidate_id": None,
            "route": route,
            "pairwise_records": [],
            "model_calls": 0,
            "latency_ms": 0.0,
            "confidence_source": "counterbalanced_agreement",
            "question_e_used": False,
            "gt_visible": False,
        }

    by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in [initial, *alternatives]
    }
    pairwise_records: list[dict[str, Any]] = []
    total_latency = 0.0
    model_calls = 0
    for alternative in alternatives:
        passes: list[dict[str, Any]] = []
        for pass_index, (swap, crop_scale) in enumerate(
            (
                (False, config.first_crop_scale),
                (True, config.second_crop_scale),
            ),
            1,
        ):
            sheet, mapping = render_counterfactual_sheet(
                image, initial, alternative, swap=swap, crop_scale=crop_scale
            )
            prompt = counterfactual_prompt(inference, mapping, by_id)
            started = time.perf_counter()
            raw = verifier.generate_visual_text(sheet, prompt, max_new_tokens=12)
            latency_ms = (time.perf_counter() - started) * 1000.0
            total_latency += latency_ms
            model_calls += 1
            choice = parse_counterfactual_choice(raw)
            chosen_candidate_id = mapping.get(choice) if choice in {"A", "B"} else None
            passes.append(
                {
                    "pass_index": pass_index,
                    "swap": swap,
                    "crop_scale": crop_scale,
                    "label_to_candidate": mapping,
                    "choice": choice,
                    "chosen_candidate_id": chosen_candidate_id,
                    "raw_output": raw,
                    "latency_ms": latency_ms,
                }
            )
        choices = [item["chosen_candidate_id"] for item in passes]
        alternative_id = str(alternative.get("candidate_id"))
        initial_id = str(initial.get("candidate_id"))
        if choices == [alternative_id, alternative_id]:
            outcome = "alternative_wins"
        elif choices == [initial_id, initial_id]:
            outcome = "initial_wins"
        elif any(choice is None for choice in choices):
            outcome = "invalid_or_abstained"
        else:
            outcome = "counterbalance_disagreement"
        pairwise_records.append(
            {
                "initial_candidate_id": initial_id,
                "alternative_candidate_id": alternative_id,
                "alternative_source_agent": alternative.get("source_agent"),
                "alternative_independent_score": candidate_independent_score(
                    alternative
                ),
                "outcome": outcome,
                "passes": passes,
            }
        )

    winners = [
        item["alternative_candidate_id"]
        for item in pairwise_records
        if item["outcome"] == "alternative_wins"
    ]
    winner = winners[0] if len(winners) == 1 else None
    if winner is not None:
        status = "completed"
        reason = "single_counterbalanced_winner"
    elif len(winners) > 1:
        status = "abstained"
        reason = "multiple_counterbalanced_winners"
    elif any(item["outcome"] == "initial_wins" for item in pairwise_records):
        status = "completed"
        reason = "initial_counterfactually_confirmed"
    else:
        status = "abstained"
        reason = "no_counterbalanced_consensus"
    return {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "status": status,
        "reason": reason,
        "abstained": status == "abstained",
        "winner_candidate_id": winner,
        "route": route,
        "pairwise_records": pairwise_records,
        "model_calls": model_calls,
        "latency_ms": total_latency,
        "confidence_source": "counterbalanced_agreement",
        "self_reported_confidence_used": False,
        "question_e_used": False,
        "gt_visible": False,
        "config": asdict(config),
    }


def select_counterfactual_candidate(
    inference: dict[str, Any],
    evidence: dict[str, Any] | None,
    config: CounterfactualConfig,
) -> dict[str, Any]:
    """Commit only a uniquely counterbalanced, independently eligible winner."""
    safe = copy.deepcopy(inference)
    if safe.get("question_e_used") is not False:
        raise ValueError("Unsafe inference: question_e_used must be false")
    initial = safe["initial_candidate"]
    alternatives, route = eligible_alternatives(safe, config)
    eligible = {str(item.get("candidate_id")): item for item in alternatives}
    verifier = copy.deepcopy(evidence or {})
    winner_id = str(verifier.get("winner_candidate_id") or "")
    selected = initial
    guard_reason = "keep_initial_no_counterfactual_winner"
    record = next(
        (
            item
            for item in verifier.get("pairwise_records", [])
            if str(item.get("alternative_candidate_id")) == winner_id
        ),
        None,
    )
    pass_choices = (
        [str(item.get("chosen_candidate_id") or "") for item in record.get("passes", [])]
        if record
        else []
    )
    if (
        winner_id in eligible
        and record is not None
        and record.get("outcome") == "alternative_wins"
        and pass_choices == [winner_id, winner_id]
    ):
        selected = eligible[winner_id]
        guard_reason = "counterbalanced_independent_replacement"
    elif verifier.get("reason") == "multiple_counterbalanced_winners":
        guard_reason = "ambiguous_counterfactual_winners_keep_initial"
    elif not alternatives:
        guard_reason = route.get(
            "route_reason", "no_independently_supported_alternative"
        )
    selected_id = str(selected.get("candidate_id") or initial.get("candidate_id"))
    return {
        "selected_candidate": copy.deepcopy(selected),
        "selected_candidate_id": selected_id,
        "selected_hypothesis_id": str(
            selected.get("hypothesis_id")
            or selected.get("parent_candidate_id")
            or selected_id
        ),
        "replaced_initial": selected_id
        != str(initial.get("candidate_id") or ""),
        "guard_reason": guard_reason,
        "independent_score": candidate_independent_score(selected),
        "route": route,
        "gt_visible": False,
        "question_e_used": False,
    }


def apply_counterfactual_selection(
    row: dict[str, Any],
    evidence: dict[str, Any] | None,
    config: CounterfactualConfig,
) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result.pop("evaluation", None)
    inference = result["inference"]
    selection = select_counterfactual_candidate(inference, evidence, config)
    previous_decision = inference.get("decision")
    previous_stop_reason = inference.get("stop_reason")
    selected = selection["selected_candidate"]
    inference["pre_selector_final_candidate_id"] = inference.get("final_candidate_id")
    inference["counterfactual_verification"] = copy.deepcopy(evidence or {})
    inference["counterfactual_selection"] = selection
    inference["final_candidate_id"] = selection["selected_candidate_id"]
    inference["final_hypothesis_id"] = selection["selected_hypothesis_id"]
    inference["final_bbox"] = copy.deepcopy(selected.get("bbox"))
    inference["confidence"] = selection["independent_score"]

    model_calls = int((evidence or {}).get("model_calls", 0))
    latency_ms = float((evidence or {}).get("latency_ms", 0.0))
    if selection["replaced_initial"]:
        inference["decision"] = "refine"
        inference["stop_reason"] = selection["guard_reason"]
    elif model_calls and (evidence or {}).get("abstained"):
        inference["decision"] = "escalate"
        inference["stop_reason"] = "counterfactual_abstained_keep_initial"
    else:
        inference["decision"] = previous_decision
        inference["stop_reason"] = previous_stop_reason
    call = {
        "call_id": "call_counterfactual_candidate_verifier",
        "agent": "CounterfactualCandidateVerifier",
        "action": "counterbalanced_candidate_centric_pairwise_verification",
        "input": {
            "query": inference.get("query", ""),
            "coordinate_frame": "global_normalized",
            "transformed_observations": [
                "full_image_with_candidates",
                "context_preserving_candidate_crops",
            ],
        },
        "output": {
            "winner_candidate_id": (evidence or {}).get("winner_candidate_id"),
            "reason": (evidence or {}).get("reason"),
        },
        "evidence": copy.deepcopy(evidence or {}),
        "model_call": bool(model_calls),
        "perception_call": bool(model_calls),
        "model_calls": model_calls,
        "latency_ms": latency_ms,
        "status": (evidence or {}).get("status", "skipped"),
    }
    if model_calls:
        inference.setdefault("action_trace", []).append(call)
        inference.setdefault("agent_calls", []).append(call)
        inference.setdefault("unit_calls", []).append(call)
    inference["child_calls"] = inference["unit_calls"]
    result["schema_version"] = COUNTERFACTUAL_SCHEMA_VERSION
    result["method"] = "hierarchical+counterfactual_v4_3"
    result["bbox"] = copy.deepcopy(inference["final_bbox"])
    result["parse_ok"] = result["bbox"] is not None
    cost = result.setdefault("cost", {})
    cost["counterfactual_verifier_calls"] = model_calls
    cost["perception_calls"] = float(cost.get("perception_calls", 0)) + model_calls
    cost["executed_perception_calls"] = float(
        cost.get("executed_perception_calls", 0)
    ) + model_calls
    cost["incremental_agent_latency_ms"] = float(
        cost.get("incremental_agent_latency_ms", 0.0)
    ) + latency_ms
    cost["end_to_end_latency_ms"] = float(
        cost.get("end_to_end_latency_ms", cost.get("latency_ms", 0.0))
    ) + latency_ms
    cost["latency_ms"] = cost["end_to_end_latency_ms"]
    cost["dispatch"] = bool(cost.get("dispatch")) or bool(model_calls)
    return result


def counterfactual_verifier_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(row.get("status", "unknown") for row in rows)
    outcomes = Counter(
        pair.get("outcome", "unknown")
        for row in rows
        for pair in row.get("pairwise_records", [])
    )
    calls = sum(int(row.get("model_calls", 0)) for row in rows)
    routed = sum(bool(row.get("model_calls", 0)) for row in rows)
    winners = sum(bool(row.get("winner_candidate_id")) for row in rows)
    return {
        "Samples": len(rows),
        "Routed Samples": routed,
        "Routing Rate": routed / len(rows) if rows else 0.0,
        "Model Calls": calls,
        "Avg Additional Calls": calls / len(rows) if rows else 0.0,
        "Counterbalanced Winners": winners,
        "Winner Rate": winners / routed if routed else 0.0,
        "Status Distribution": dict(sorted(status.items())),
        "Pair Outcome Distribution": dict(sorted(outcomes.items())),
        "Self-reported Confidence Used": False,
    }
