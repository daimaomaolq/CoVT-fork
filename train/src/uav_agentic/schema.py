from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


Box = list[float]


class SpatialFrame(str, Enum):
    LOCAL_ATTRIBUTE = "local_attribute"
    GLOBAL_ABSOLUTE = "global_absolute"
    GLOBAL_ORDER = "global_order"
    OBJECT_RELATIVE = "object_relative"
    ORIENTATION_DEPENDENT = "orientation_dependent"
    TEMPORAL_EVENT = "temporal_event"


class Decision(str, Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    ESCALATE = "escalate"


class Method(str, Enum):
    ONE_PASS = "one_pass"
    CONFIDENCE_GATED = "confidence_gated"
    PARENT_ONLY = "parent_only"
    STATIC_ALL = "static_all"
    HIERARCHICAL = "hierarchical"


@dataclass
class QueryConstraintGraph:
    original: str
    target: str
    attributes: list[str] = field(default_factory=list)
    context: str = ""
    relations: list[str] = field(default_factory=list)
    spatial_frames: list[SpatialFrame] = field(
        default_factory=lambda: [SpatialFrame.LOCAL_ATTRIBUTE]
    )
    global_position: str | None = None
    ordinal_constraint: str | None = None
    local_target_query: str = ""
    zoom_query: str = ""
    parser_version: str = "constraint-parser-v3"

    @property
    def has_context(self) -> bool:
        return bool(self.context)

    @property
    def has_relation(self) -> bool:
        return bool(self.relations)

    @property
    def is_position_sensitive(self) -> bool:
        sensitive = {
            SpatialFrame.GLOBAL_ABSOLUTE,
            SpatialFrame.GLOBAL_ORDER,
            SpatialFrame.OBJECT_RELATIVE,
            SpatialFrame.ORIENTATION_DEPENDENT,
        }
        return bool(sensitive.intersection(self.spatial_frames))


@dataclass
class Observation:
    observation_id: str
    view_type: str
    coordinate_frame: str = "global_normalized"
    crop_region: Box | None = None
    transform: str | None = None
    preserves_context: bool = True


@dataclass
class Candidate:
    candidate_id: str
    bbox: Box | None
    source_agent: str
    query_used: str
    observation: Observation
    bbox_token_confidence: float = 0.5
    confidence_available: bool = True
    bbox_token_count: int = 0
    raw_output: str = ""
    latency_ms: float = 0.0
    parent_candidate_id: str | None = None
    parse_ok: bool = True
    box_plausibility: float = 0.0
    target_consistency: float = 0.5
    context_consistency: float = 0.5
    relation_consistency: float = 0.5
    global_constraint_score: float = 0.5
    observation_agreement: float = 0.0
    ambiguity_penalty: float = 0.0
    competition_margin: float = 0.0
    fused_score: float = 0.0
    accepted_by_guard: bool = True
    rejection_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parse_ok = self.bbox is not None


@dataclass
class AgentCall:
    call_id: str
    agent: str
    action: str
    input: dict[str, Any]
    output: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    model_call: bool = False
    perception_call: bool = False
    latency_ms: float = 0.0
    status: str = "completed"


@dataclass
class EscalationFeedback:
    status: str
    uncertainty_summary: str
    recommended_action: str
    region_of_interest: Box | None
    reason: str
    expected_evidence: str
    operator_message: str
    generator: str
    valid: bool = True
    fallback_used: bool = False


@dataclass
class AgenticConfig:
    method: Method = Method.HIERARCHICAL
    max_child_perception_calls: int = 2
    confidence_threshold: float = 0.42
    shape_threshold: float = 0.45
    small_area_threshold: float = 0.003
    large_area_threshold: float = 0.55
    agreement_threshold: float = 0.25
    relation_threshold: float = 0.35
    global_constraint_threshold: float = 0.40
    competition_iou_threshold: float = 0.20
    competition_margin_threshold: float = 0.12
    final_confidence_threshold: float = 0.48
    information_gain_threshold: float = 0.02
    zoom_scales: tuple[float, ...] = (1.5, 2.0)
    zoom_identity_iou_threshold: float = 0.05
    zoom_center_distance_threshold: float = 0.20
    zoom_relation_drop_tolerance: float = 0.15
    zoom_global_drop_tolerance: float = 0.10
    context_union_margin: float = 1.25
    disabled_agents: set[str] = field(default_factory=set)
    feedback_mode: str = "template"
    enable_escalation: bool = True
    include_raw_output: bool = False
    front_behind_axis: str = "unknown"
    weight_full_confidence: float = 0.30
    weight_shape: float = 0.10
    weight_target: float = 0.20
    weight_relation: float = 0.15
    weight_global: float = 0.15
    weight_stability: float = 0.10
    ambiguity_penalty_weight: float = 0.15

    def validate(self) -> None:
        if self.max_child_perception_calls < 0:
            raise ValueError("max_child_perception_calls must be non-negative")
        bounded = {
            "confidence_threshold": self.confidence_threshold,
            "shape_threshold": self.shape_threshold,
            "small_area_threshold": self.small_area_threshold,
            "large_area_threshold": self.large_area_threshold,
            "agreement_threshold": self.agreement_threshold,
            "relation_threshold": self.relation_threshold,
            "global_constraint_threshold": self.global_constraint_threshold,
            "competition_iou_threshold": self.competition_iou_threshold,
            "competition_margin_threshold": self.competition_margin_threshold,
            "final_confidence_threshold": self.final_confidence_threshold,
            "information_gain_threshold": self.information_gain_threshold,
            "zoom_identity_iou_threshold": self.zoom_identity_iou_threshold,
            "zoom_center_distance_threshold": self.zoom_center_distance_threshold,
            "zoom_relation_drop_tolerance": self.zoom_relation_drop_tolerance,
            "zoom_global_drop_tolerance": self.zoom_global_drop_tolerance,
        }
        invalid = {name: value for name, value in bounded.items() if not 0 <= value <= 1}
        if invalid:
            raise ValueError(f"Thresholds must be in [0, 1]: {invalid}")
        weights = {
            "full_confidence": self.weight_full_confidence,
            "shape": self.weight_shape,
            "target": self.weight_target,
            "relation": self.weight_relation,
            "global": self.weight_global,
            "stability": self.weight_stability,
        }
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"Fusion weights must be non-negative with positive sum: {weights}")
        if self.ambiguity_penalty_weight < 0:
            raise ValueError("ambiguity_penalty_weight must be non-negative")
        if not self.zoom_scales or any(scale <= 1.0 for scale in self.zoom_scales):
            raise ValueError("zoom_scales must contain values greater than 1")
        if self.feedback_mode not in {"off", "template", "base"}:
            raise ValueError("feedback_mode must be off, template, or base")
        if self.front_behind_axis not in {"unknown", "y"}:
            raise ValueError("front_behind_axis must be unknown or y")
        unknown = self.disabled_agents.difference({"target", "context", "relation", "zoom"})
        if unknown:
            raise ValueError(f"Unknown disabled agents: {sorted(unknown)}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
