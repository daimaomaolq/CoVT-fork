from __future__ import annotations

import copy
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .geometry import box_iou
from .grounder import CoVTGrounder, insert_anchor_prompt


SELECTOR_SCHEMA_VERSION = "dai-uav-agent-v4.2-frozen-selection"


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def hypothesis_id(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("hypothesis_id")
        or candidate.get("parent_candidate_id")
        or candidate.get("candidate_id")
        or ""
    )


def valid_candidates(inference: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in inference.get("target_candidates", [])
        if item.get("bbox") is not None
        and item.get("accepted_by_guard", True)
        and "duplicate_hypothesis" not in item.get("rejection_reasons", [])
    ]


def hypothesis_representatives(
    inference: dict[str, Any],
    maximum: int = 4,
) -> list[dict[str, Any]]:
    candidates = valid_candidates(inference)
    by_id = {str(item.get("candidate_id")): item for item in candidates}
    clusters = inference.get("hypothesis_clusters") or (
        inference.get("verification_evidence", {})
        .get("fusion", {})
        .get("hypotheses", [])
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cluster in clusters:
        candidate = by_id.get(str(cluster.get("representative_candidate_id", "")))
        if candidate is None:
            candidate = by_id.get(str(cluster.get("root_candidate_id", "")))
        if candidate is None:
            continue
        root = str(cluster.get("hypothesis_id") or hypothesis_id(candidate))
        if not root or root in seen:
            continue
        seen.add(root)
        result.append(candidate)
    if not result:
        ranked_ids = (
            inference.get("verification_evidence", {})
            .get("fusion", {})
            .get("ranked_candidate_ids", [])
        )
        ordered = [by_id[item] for item in ranked_ids if item in by_id]
        ordered.extend(item for item in candidates if item not in ordered)
        for candidate in ordered:
            root = hypothesis_id(candidate)
            if not root or root in seen:
                continue
            seen.add(root)
            result.append(candidate)
    return result[:maximum]


def render_candidate_overlay(
    image: Image.Image,
    representatives: list[dict[str, Any]],
) -> tuple[Image.Image, dict[str, str]]:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size
    palette = ("#ff2d55", "#00a8ff", "#34c759", "#ff9500", "#af52de")
    line_width = max(2, round(min(width, height) / 250))
    mapping: dict[str, str] = {}
    for index, candidate in enumerate(representatives):
        label = f"H{index}"
        mapping[label] = hypothesis_id(candidate)
        x1, y1, x2, y2 = [float(value) for value in candidate["bbox"][:4]]
        pixels = (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
        color = palette[index % len(palette)]
        draw.rectangle(pixels, outline=color, width=line_width)
        left, top, right, bottom = draw.textbbox(
            (pixels[0], pixels[1]), label, font=font
        )
        tag_width = right - left + 6
        tag_height = bottom - top + 4
        tag_y = max(0, pixels[1] - tag_height)
        draw.rectangle(
            (pixels[0], tag_y, pixels[0] + tag_width, tag_y + tag_height),
            fill=color,
        )
        draw.text((pixels[0] + 3, tag_y + 2), label, fill="white", font=font)
    return canvas, mapping


def verifier_prompt(
    inference: dict[str, Any],
    representatives: list[dict[str, Any]],
    label_mapping: dict[str, str],
) -> str:
    graph = inference.get("constraint_graph", {})
    candidate_lines = []
    by_hypothesis = {hypothesis_id(item): item for item in representatives}
    for label, root in label_mapping.items():
        candidate = by_hypothesis[root]
        bbox = [round(float(value), 4) for value in candidate["bbox"][:4]]
        candidate_lines.append(
            f"- {label}: bbox={bbox}; source={candidate.get('source_agent', 'unknown')}"
        )
    return (
        "You are a conservative visual grounding verifier. The full UAV image "
        "contains labelled candidate boxes. Select exactly one box only when its "
        "visible pixels satisfy the referring query; otherwise abstain. Evaluate "
        "target identity and attributes first, then context, spatial relation, "
        "ordinal condition, and original-image global position. Do not prefer a "
        "box merely because it is larger. All coordinates refer to the ORIGINAL "
        "full image, not a crop.\n\n"
        f"Original query: {inference.get('query', '')}\n"
        f"Target: {graph.get('target', '')}\n"
        f"Attributes: {', '.join(graph.get('attributes', [])) or 'none'}\n"
        f"Context: {graph.get('context') or 'none'}\n"
        f"Relations: {', '.join(graph.get('relations', [])) or 'none'}\n"
        f"Global position: {graph.get('global_position') or 'none'}\n"
        f"Ordinal: {graph.get('ordinal_constraint') or 'none'}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\nReturn strict JSON only:\n"
        '{"selected":"H0|H1|...|ABSTAIN","confidence":0.0,'
        '"target_match":0.0,"context_match":0.0,"relation_match":0.0,'
        '"global_match":0.0,"reason":"brief visible evidence"}'
    )


def parse_verifier_json(raw: str) -> dict[str, Any] | None:
    options = [raw.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        options.insert(0, fenced.group(1))
    embedded = re.search(r"\{.*\}", raw, re.DOTALL)
    if embedded:
        options.append(embedded.group(0))
    for option in options:
        try:
            parsed = json.loads(option)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class FrozenBaseVisualVerifier(CoVTGrounder):
    """CoVT loader with a frozen-base image-conditioned text path."""

    def generate_visual_text(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 160,
    ) -> str:
        import torch

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = insert_anchor_prompt(
            text, self.anchor_prompt, self.settings.anchor_prompt_mode
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[1]
        adapter_context = (
            self.model.disable_adapter()
            if hasattr(self.model, "disable_adapter")
            else _NullContext()
        )
        kwargs = self._generation_kwargs(max_new_tokens)
        kwargs["output_scores"] = False
        with adapter_context, torch.no_grad(), self._autocast_context(torch):
            result = self.model.generate(**inputs, **kwargs)
        sequence = result.sequences if hasattr(result, "sequences") else result
        raw = self.processor.decode(
            sequence[0, input_length:], skip_special_tokens=True
        ).strip()
        del result, sequence, inputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return raw


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def verify_frozen_candidates(
    verifier: FrozenBaseVisualVerifier,
    image: Image.Image,
    row: dict[str, Any],
    maximum: int = 4,
) -> dict[str, Any]:
    """Verify candidates without reading row['evaluation'] or any GT field."""
    inference = copy.deepcopy(row["inference"])
    if inference.get("question_e_used") is not False:
        raise ValueError("Unsafe trace: question_e_used must be explicitly false")
    representatives = hypothesis_representatives(inference, maximum)
    if len(representatives) < 2:
        return {
            "schema_version": SELECTOR_SCHEMA_VERSION,
            "sample_id": row["sample_id"],
            "status": "skipped",
            "reason": "fewer_than_two_hypotheses",
            "abstained": True,
            "selected_hypothesis_id": None,
            "confidence": 0.0,
            "hypothesis_scores": {},
            "question_e_used": False,
            "gt_visible": False,
            "model_calls": 0,
            "latency_ms": 0.0,
        }
    overlay, mapping = render_candidate_overlay(image, representatives)
    prompt = verifier_prompt(inference, representatives, mapping)
    started = time.perf_counter()
    raw = verifier.generate_visual_text(overlay, prompt)
    latency_ms = (time.perf_counter() - started) * 1000
    parsed = parse_verifier_json(raw)
    selected_label = (
        str(parsed.get("selected", "ABSTAIN")).strip().upper()
        if parsed
        else "ABSTAIN"
    )
    confidence = clamp01(parsed.get("confidence"), 0.0) if parsed else 0.0
    selected_hypothesis_id = mapping.get(selected_label)
    abstained = selected_hypothesis_id is None
    residual = (1.0 - confidence) / max(len(mapping) - 1, 1)
    scores = {
        root: confidence if root == selected_hypothesis_id else residual
        for root in mapping.values()
    }
    if abstained:
        scores = {root: 0.5 for root in mapping.values()}
    return {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "status": "abstained" if abstained else "completed",
        "abstained": abstained,
        "selected_label": selected_label,
        "selected_hypothesis_id": selected_hypothesis_id,
        "confidence": confidence,
        "hypothesis_scores": scores,
        "label_to_hypothesis": mapping,
        "target_match": clamp01(parsed.get("target_match"), 0.5)
        if parsed
        else 0.5,
        "context_match": clamp01(parsed.get("context_match"), 0.5)
        if parsed
        else 0.5,
        "relation_match": clamp01(parsed.get("relation_match"), 0.5)
        if parsed
        else 0.5,
        "global_match": clamp01(parsed.get("global_match"), 0.5)
        if parsed
        else 0.5,
        "reason": str(parsed.get("reason", "")) if parsed else "invalid_json",
        "raw_output": raw,
        "adapter_disabled": True,
        "full_image_coordinates_preserved": True,
        "question_e_used": False,
        "gt_visible": False,
        "model_calls": 1,
        "latency_ms": latency_ms,
    }


@dataclass(frozen=True)
class SelectionConfig:
    selector: str = "conservative_visual"
    verifier_confidence_threshold: float = 0.70
    verifier_margin_threshold: float = 0.20
    minimum_composite_gain: float = 0.05
    visual_weight: float = 0.55
    token_weight: float = 0.20
    relation_weight: float = 0.10
    global_weight: float = 0.10
    shape_weight: float = 0.05

    def validate(self) -> None:
        if self.selector not in {
            "initial",
            "stored_fusion",
            "visual_only",
            "conservative_visual",
        }:
            raise ValueError(f"Unknown selector: {self.selector}")
        for name in (
            "verifier_confidence_threshold",
            "verifier_margin_threshold",
            "minimum_composite_gain",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def _candidate_score(
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    config: SelectionConfig,
) -> float:
    root = hypothesis_id(candidate)
    visual_available = not evidence.get("abstained", True)
    visual = clamp01(evidence.get("hypothesis_scores", {}).get(root), 0.5)
    token = (
        clamp01(candidate.get("bbox_token_confidence"), 0.5)
        if candidate.get("confidence_available", True)
        else 0.5
    )
    relation = clamp01(candidate.get("relation_consistency"), 0.5)
    global_score = clamp01(candidate.get("global_constraint_score"), 0.5)
    shape = clamp01(candidate.get("box_plausibility"), 0.5)
    weighted = (
        config.token_weight * token
        + config.relation_weight * relation
        + config.global_weight * global_score
        + config.shape_weight * shape
    )
    denominator = (
        config.token_weight
        + config.relation_weight
        + config.global_weight
        + config.shape_weight
    )
    if visual_available:
        weighted += config.visual_weight * visual
        denominator += config.visual_weight
    return weighted / max(denominator, 1e-9)


def select_frozen_candidate(
    inference: dict[str, Any],
    verifier_evidence: dict[str, Any] | None,
    config: SelectionConfig,
) -> dict[str, Any]:
    """Select using inference fields only. GT is intentionally not accepted."""
    config.validate()
    safe_inference = copy.deepcopy(inference)
    if safe_inference.get("question_e_used") is not False:
        raise ValueError("Unsafe inference: question_e_used must be false")
    candidates = valid_candidates(safe_inference)
    initial = safe_inference["initial_candidate"]
    by_id = {str(item.get("candidate_id")): item for item in candidates}
    if str(initial.get("candidate_id")) not in by_id and initial.get("bbox") is not None:
        candidates.insert(0, initial)
        by_id[str(initial.get("candidate_id"))] = initial
    evidence = copy.deepcopy(verifier_evidence or {})

    stored_id = str(safe_inference.get("final_candidate_id") or "")
    selected = initial
    guard_reason = "selector_initial"
    visual_supported = False
    scores: dict[str, float] = {}

    if config.selector == "stored_fusion" and stored_id in by_id:
        selected = by_id[stored_id]
        guard_reason = "stored_fusion"
    elif config.selector in {"visual_only", "conservative_visual"}:
        representatives = hypothesis_representatives(safe_inference, maximum=32)
        representative_by_root = {
            hypothesis_id(item): item for item in representatives
        }
        for root, candidate in representative_by_root.items():
            scores[root] = _candidate_score(candidate, evidence, config)
        selected_root = str(evidence.get("selected_hypothesis_id") or "")
        proposed = representative_by_root.get(selected_root)
        confidence = clamp01(evidence.get("confidence"), 0.0)
        visual_scores = sorted(
            (
                clamp01(value, 0.0)
                for value in evidence.get("hypothesis_scores", {}).values()
            ),
            reverse=True,
        )
        margin = (
            visual_scores[0] - visual_scores[1]
            if len(visual_scores) >= 2
            else visual_scores[0]
            if visual_scores
            else 0.0
        )
        initial_root = hypothesis_id(initial)
        initial_score = scores.get(
            initial_root, _candidate_score(initial, evidence, config)
        )
        proposed_score = scores.get(selected_root, 0.0)
        visual_supported = bool(
            proposed is not None
            and not evidence.get("abstained", True)
            and confidence >= config.verifier_confidence_threshold
            and margin >= config.verifier_margin_threshold
        )
        if config.selector == "visual_only" and proposed is not None:
            selected = proposed
            guard_reason = "visual_selected"
        elif (
            visual_supported
            and proposed is not None
            and (
                selected_root == initial_root
                or proposed_score - initial_score >= config.minimum_composite_gain
            )
        ):
            selected = proposed
            guard_reason = (
                "visual_confirmed_initial"
                if selected_root == initial_root
                else "visual_supported_replacement"
            )
        elif evidence.get("abstained", True):
            guard_reason = "verifier_abstained_keep_initial"
        elif proposed is None:
            guard_reason = "invalid_verifier_hypothesis_keep_initial"
        elif confidence < config.verifier_confidence_threshold:
            guard_reason = "low_verifier_confidence_keep_initial"
        elif margin < config.verifier_margin_threshold:
            guard_reason = "low_verifier_margin_keep_initial"
        else:
            guard_reason = "insufficient_composite_gain_keep_initial"

    selected_id = str(selected.get("candidate_id") or initial.get("candidate_id"))
    return {
        "selected_candidate": copy.deepcopy(selected),
        "selected_candidate_id": selected_id,
        "selected_hypothesis_id": hypothesis_id(selected),
        "replaced_initial": selected_id != str(initial.get("candidate_id")),
        "guard_reason": guard_reason,
        "visual_supported": visual_supported,
        "candidate_scores": scores,
        "gt_visible": False,
        "question_e_used": False,
    }


def apply_selection_to_trace(
    row: dict[str, Any],
    verifier_evidence: dict[str, Any] | None,
    config: SelectionConfig,
) -> dict[str, Any]:
    """Rewrite inference output before GT-dependent evaluation is attached."""
    result = copy.deepcopy(row)
    result.pop("evaluation", None)
    has_verifier_record = verifier_evidence is not None
    inference = result["inference"]
    selection = select_frozen_candidate(inference, verifier_evidence, config)
    selected = selection["selected_candidate"]
    previous_final = inference.get("final_candidate_id")
    inference["pre_selector_final_candidate_id"] = previous_final
    inference["candidate_verification"] = copy.deepcopy(verifier_evidence or {})
    inference["candidate_selection"] = selection
    inference["final_candidate_id"] = selection["selected_candidate_id"]
    inference["final_hypothesis_id"] = selection["selected_hypothesis_id"]
    inference["final_bbox"] = copy.deepcopy(selected.get("bbox"))
    inference["confidence"] = clamp01(
        selection["candidate_scores"].get(
            selection["selected_hypothesis_id"],
            selected.get("fused_score", selected.get("bbox_token_confidence", 0.5)),
        ),
        0.5,
    )
    if selection["replaced_initial"]:
        inference["decision"] = "refine"
        inference["stop_reason"] = "visual_supported_candidate_replacement"
    elif verifier_evidence and verifier_evidence.get("abstained"):
        inference["decision"] = "escalate"
        inference["stop_reason"] = "candidate_verifier_abstained"
    else:
        inference["decision"] = "accept"
        inference["stop_reason"] = selection["guard_reason"]
    verifier_call = {
        "call_id": "call_candidate_verifier",
        "agent": "CandidateVerifier",
        "action": "full_image_pairwise_candidate_verification",
        "input": {
            "query": inference.get("query", ""),
            "coordinate_frame": "global_normalized",
        },
        "output": {
            "selected_hypothesis_id": (
                verifier_evidence or {}
            ).get("selected_hypothesis_id"),
            "confidence": (verifier_evidence or {}).get("confidence", 0.0),
            "abstained": (verifier_evidence or {}).get("abstained", True),
        },
        "evidence": copy.deepcopy(verifier_evidence or {}),
        "model_call": bool((verifier_evidence or {}).get("model_calls", 0)),
        "perception_call": bool((verifier_evidence or {}).get("model_calls", 0)),
        "latency_ms": float((verifier_evidence or {}).get("latency_ms", 0.0)),
        "status": (verifier_evidence or {}).get("status", "skipped"),
    }
    if has_verifier_record:
        inference.setdefault("action_trace", []).append(verifier_call)
        inference.setdefault("agent_calls", []).append(verifier_call)
        inference.setdefault("unit_calls", []).append(verifier_call)
    inference["child_calls"] = inference["unit_calls"]
    result["schema_version"] = SELECTOR_SCHEMA_VERSION
    result["method"] = f"{row.get('method', 'hierarchical')}+{config.selector}"
    result["bbox"] = copy.deepcopy(inference["final_bbox"])
    result["parse_ok"] = result["bbox"] is not None
    cost = result.setdefault("cost", {})
    verifier_model_calls = int((verifier_evidence or {}).get("model_calls", 0))
    verifier_latency = float((verifier_evidence or {}).get("latency_ms", 0.0))
    cost["candidate_verifier_calls"] = verifier_model_calls
    cost["perception_calls"] = float(cost.get("perception_calls", 0)) + (
        verifier_model_calls
    )
    cost["executed_perception_calls"] = float(
        cost.get("executed_perception_calls", 0)
    ) + verifier_model_calls
    cost["incremental_agent_latency_ms"] = float(
        cost.get("incremental_agent_latency_ms", 0.0)
    ) + verifier_latency
    cost["end_to_end_latency_ms"] = float(
        cost.get("end_to_end_latency_ms", cost.get("latency_ms", 0.0))
    ) + verifier_latency
    cost["latency_ms"] = cost["end_to_end_latency_ms"]
    cost["dispatch"] = bool(cost.get("dispatch")) or bool(verifier_model_calls)
    return result


def posthoc_action_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """GT-based diagnostics computed only after all selector decisions."""
    useful_calls = 0
    total_calls = 0
    selected_useful = 0
    regrets: list[float] = []
    by_unit: dict[str, dict[str, float]] = defaultdict(
        lambda: {"Calls": 0.0, "Useful": 0.0, "Mean Best Gain": 0.0}
    )
    unit_gains: dict[str, list[float]] = defaultdict(list)
    source_to_unit = {
        "TargetAgent": "TargetAgent",
        "ZoomAgent": "ZoomAgent",
    }
    for row in rows:
        gt = row["evaluation"]["gt_bbox"]
        initial_iou = float(row["evaluation"]["initial_iou"])
        candidates = valid_candidates(row["inference"])
        gains: dict[str, float] = {}
        for source, unit in source_to_unit.items():
            boxes = [
                item["bbox"] for item in candidates if item.get("source_agent") == source
            ]
            if not boxes:
                continue
            best_gain = max(box_iou(box, gt) for box in boxes) - initial_iou
            gains[unit] = best_gain
            total_calls += 1
            by_unit[unit]["Calls"] += 1
            unit_gains[unit].append(best_gain)
            if best_gain >= 0.10:
                useful_calls += 1
                by_unit[unit]["Useful"] += 1
        selected_id = row["inference"].get("final_candidate_id")
        selected = next(
            (
                item
                for item in candidates
                if str(item.get("candidate_id")) == str(selected_id)
            ),
            row["inference"].get("initial_candidate", {}),
        )
        selected_unit = source_to_unit.get(str(selected.get("source_agent")))
        selected_gain = gains.get(selected_unit, 0.0)
        best_gain = max([0.0, *gains.values()])
        regrets.append(max(0.0, best_gain - max(0.0, selected_gain)))
        if selected_unit and selected_gain >= 0.10:
            selected_useful += 1
    for unit, values in unit_gains.items():
        by_unit[unit]["Mean Best Gain"] = (
            sum(values) / len(values) if values else 0.0
        )
        by_unit[unit]["Useful Rate"] = (
            by_unit[unit]["Useful"] / by_unit[unit]["Calls"]
            if by_unit[unit]["Calls"]
            else 0.0
        )
    return {
        "Useful Call Rate@DeltaIoU0.1": useful_calls / total_calls
        if total_calls
        else 0.0,
        "Wasted Call Rate@DeltaIoU0.1": 1.0 - useful_calls / total_calls
        if total_calls
        else 0.0,
        "Useful Selected Action Count": selected_useful,
        "Mean Action Regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "Per Unit": dict(sorted(by_unit.items())),
    }


def verifier_summary(
    sidecar_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(row.get("status", "unknown") for row in sidecar_rows)
    calls = sum(int(row.get("model_calls", 0)) for row in sidecar_rows)
    completed = [row for row in sidecar_rows if row.get("status") == "completed"]
    return {
        "Samples": len(sidecar_rows),
        "Model Calls": calls,
        "Coverage": len(completed) / len(sidecar_rows) if sidecar_rows else 0.0,
        "Abstain Rate": (
            sum(bool(row.get("abstained", True)) for row in sidecar_rows)
            / len(sidecar_rows)
            if sidecar_rows
            else 0.0
        ),
        "Mean Confidence": (
            sum(float(row.get("confidence", 0.0)) for row in completed)
            / len(completed)
            if completed
            else 0.0
        ),
        "Mean Latency_ms": (
            sum(float(row.get("latency_ms", 0.0)) for row in sidecar_rows)
            / len(sidecar_rows)
            if sidecar_rows
            else 0.0
        ),
        "Status Distribution": dict(counts),
        "GT Visible": False,
        "question_e Used": False,
    }
