from __future__ import annotations

from PIL import Image

from ..geometry import (
    box_area,
    center_distance,
    expand_box,
    map_from_crop,
)
from ..schema import AgentCall, Candidate, Observation, to_jsonable
from .base import AgentContext, AgentResult


def _ensure_minimum_crop(
    region: list[float], minimum_size: float = 0.08
) -> list[float]:
    x1, y1, x2, y2 = region
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width = max(width, minimum_size)
    height = max(height, minimum_size)
    x1, x2 = center_x - width / 2, center_x + width / 2
    y1, y2 = center_y - height / 2, center_y + height / 2
    if x1 < 0.0:
        x2 -= x1
        x1 = 0.0
    if x2 > 1.0:
        x1 -= x2 - 1.0
        x2 = 1.0
    if y1 < 0.0:
        y2 -= y1
        y1 = 0.0
    if y2 > 1.0:
        y1 -= y2 - 1.0
        y2 = 1.0
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]


def _crop_image(image: Image.Image, region: list[float]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = region
    return image.crop(
        (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
    )


class ZoomAgent:
    name = "ZoomAgent"

    def _select_scale(self, seed: Candidate, context: AgentContext) -> float:
        scales = sorted(context.config.zoom_scales)
        return (
            scales[-1]
            if box_area(seed.bbox) < context.config.small_area_threshold
            else scales[0]
        )

    def _build_crop(
        self,
        seed: Candidate,
        context: AgentContext,
        scale: float,
    ) -> tuple[list[float], bool]:
        assert seed.bbox is not None
        if not context.config.enable_semantic_frame_protection:
            region = _ensure_minimum_crop(
                expand_box(seed.bbox, scale), context.config.zoom_min_crop_size
            )
            return region, False
        # Zoom verifies one target hypothesis. Including a large context region
        # (for example a road) can turn the crop back into the full image and
        # falsely count an identical rerun as independent support. The parent
        # re-applies context, relation and global constraints after remapping.
        region = expand_box(seed.bbox, scale)
        return (
            _ensure_minimum_crop(region, context.config.zoom_min_crop_size),
            False,
        )

    def run(
        self,
        context: AgentContext,
        seed: Candidate,
        candidate_id: str,
    ) -> AgentResult:
        if seed.bbox is None:
            call = AgentCall(
                call_id=f"call_{candidate_id}",
                agent=self.name,
                action="crop_zoom",
                input={"seed_candidate_id": seed.candidate_id},
                output={"candidate": None},
                evidence={"skipped_reason": "seed_has_no_bbox"},
                status="skipped",
            )
            return AgentResult(call=call, candidates=[], evidence=call.evidence)
        scale = self._select_scale(seed, context)
        crop_region, preserves_context = self._build_crop(seed, context, scale)
        crop = _crop_image(context.image, crop_region)
        observation = Observation(
            observation_id=f"{candidate_id}_zoom",
            view_type="crop_zoom",
            crop_region=crop_region,
            transform=f"crop_scale_{scale:g}",
            preserves_context=preserves_context,
        )
        # Candidate verification retains implicit target semantics and
        # object-relative relations. Only global image-coordinate and ordinal phrases
        # are removed, because those constraints are re-applied after remapping.
        zoom_query = (
            context.graph.zoom_query
            if context.config.enable_semantic_frame_protection
            else context.graph.original
        )
        candidate = context.grounder.ground(
            crop,
            zoom_query,
            candidate_id,
            self.name,
            observation,
            parent_candidate_id=seed.candidate_id,
        )
        local_bbox = candidate.bbox
        candidate.bbox = map_from_crop(candidate.bbox, crop_region)
        candidate.parse_ok = candidate.bbox is not None
        candidate.hypothesis_id = seed.hypothesis_id or seed.candidate_id
        rejection_reasons = []
        if candidate.bbox is not None:
            identity_distance = center_distance(candidate.bbox, seed.bbox)
            if (
                context.config.enable_semantic_frame_protection
                and identity_distance > context.config.zoom_center_distance_threshold
            ):
                rejection_reasons.append("zoom_identity_shift")
        else:
            identity_distance = 1.0
            rejection_reasons.append("zoom_parse_failed")

        candidate.rejection_reasons.extend(rejection_reasons)
        candidate.accepted_by_guard = not rejection_reasons
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=self.name,
            action=(
                "semantic_query_zoom_verification"
                if context.config.enable_semantic_frame_protection
                else "naive_crop_zoom"
            ),
            input={
                "seed_candidate_id": seed.candidate_id,
                "query": zoom_query,
                "query_scope": (
                    "semantic_query_without_global_coordinates"
                    if context.config.enable_semantic_frame_protection
                    else "full_query_ablation"
                ),
                "global_constraints_reapplied_after_mapping": True,
                "spatial_frames": [
                    frame.value for frame in context.graph.spatial_frames
                ],
                "crop_region": crop_region,
                "scale": scale,
                "preserves_context": preserves_context,
                "semantic_frame_protection": context.config.enable_semantic_frame_protection,
                "relation_checked_in_global_frame": True,
            },
            output={
                "local_bbox": local_bbox,
                "global_bbox": candidate.bbox,
                "candidate": to_jsonable(candidate),
            },
            evidence={
                "identity_center_distance": identity_distance,
                "guard_passed": candidate.accepted_by_guard,
                "rejection_reasons": rejection_reasons,
                "hypothesis_id": candidate.hypothesis_id,
                "verification_of": seed.candidate_id,
                "independent_transformed_view": (
                    box_area(crop_region) <= context.config.verification_max_crop_area
                ),
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return AgentResult(call=call, candidates=[candidate], evidence=call.evidence)
