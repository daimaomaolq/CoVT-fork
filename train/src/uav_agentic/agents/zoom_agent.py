from __future__ import annotations

from PIL import Image

from ..geometry import (
    box_area,
    center_distance,
    expand_box,
    map_from_crop,
    union_box,
)
from ..schema import AgentCall, Candidate, Observation, SpatialFrame, to_jsonable
from .base import AgentContext, AgentResult


def _ensure_minimum_crop(
    region: list[float], minimum_size: float = 0.08
) -> list[float]:
    x1, y1, x2, y2 = region
    width, height = x2 - x1, y2 - y1
    if width >= minimum_size and height >= minimum_size:
        return region
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width = max(width, minimum_size)
    height = max(height, minimum_size)
    return [
        max(0.0, center_x - width / 2),
        max(0.0, center_y - height / 2),
        min(1.0, center_x + width / 2),
        min(1.0, center_y + height / 2),
    ]


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
            region = _ensure_minimum_crop(expand_box(seed.bbox, scale))
            return region, False
        object_relative = SpatialFrame.OBJECT_RELATIVE in context.graph.spatial_frames
        valid_context = [
            candidate
            for candidate in context.context_candidates
            if candidate.bbox is not None
        ]
        if object_relative and valid_context:
            best_context = max(
                valid_context, key=lambda item: item.bbox_token_confidence
            )
            region = union_box([seed.bbox, best_context.bbox])
            region = expand_box(region, context.config.context_union_margin)
            preserves_context = True
        else:
            region = expand_box(seed.bbox, scale)
            preserves_context = not object_relative
        return _ensure_minimum_crop(region), preserves_context

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
        protected_query = (
            f"{context.graph.original} "
            f"[This is a crop from whole-image normalized region {crop_region}. "
            "Refine the already selected target identity. Interpret top/bottom/left/"
            "right and ordinal terms in the original whole-image frame, not relative "
            "to this crop, and do not re-rank identities using crop-local position.]"
        )
        zoom_query = (
            protected_query
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
        if (
            context.config.enable_semantic_frame_protection
            and SpatialFrame.OBJECT_RELATIVE in context.graph.spatial_frames
            and not preserves_context
        ):
            rejection_reasons.append("context_not_preserved")
        candidate.rejection_reasons.extend(rejection_reasons)
        candidate.accepted_by_guard = not rejection_reasons
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=self.name,
            action=(
                "semantic_frame_preserving_zoom"
                if context.config.enable_semantic_frame_protection
                else "naive_crop_zoom"
            ),
            input={
                "seed_candidate_id": seed.candidate_id,
                "query": zoom_query,
                "spatial_frames": [
                    frame.value for frame in context.graph.spatial_frames
                ],
                "crop_region": crop_region,
                "scale": scale,
                "preserves_context": preserves_context,
                "semantic_frame_protection": context.config.enable_semantic_frame_protection,
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
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return AgentResult(call=call, candidates=[candidate], evidence=call.evidence)
