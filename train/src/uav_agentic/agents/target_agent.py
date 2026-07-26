from __future__ import annotations

from PIL import Image

from ..geometry import box_center, box_iou, map_from_crop
from ..schema import AgentCall, Observation, SpatialFrame, to_jsonable
from .base import AgentContext, AgentResult
from .relation_reasoner import absolute_position_score


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


def build_overlapping_tiles(grid: int, overlap: float) -> list[list[float]]:
    """Return deterministic global-normalized transformed observation regions."""
    cell = 1.0 / grid
    padding = overlap / 2.0
    regions = []
    for row in range(grid):
        for column in range(grid):
            regions.append(
                [
                    max(0.0, column * cell - padding),
                    max(0.0, row * cell - padding),
                    min(1.0, (column + 1) * cell + padding),
                    min(1.0, (row + 1) * cell + padding),
                ]
            )
    return regions


class TargetAgent:
    name = "TargetAgent"

    def proposal_regions(
        self,
        context: AgentContext,
        seed,
        limit: int,
    ) -> list[list[float]]:
        if limit <= 0:
            return []
        regions = build_overlapping_tiles(
            context.config.target_tile_grid,
            context.config.target_tile_overlap,
        )
        seed_center = box_center(seed.bbox) if seed.bbox is not None else None

        def priority(region: list[float]) -> tuple[float, float, float]:
            contains_seed = 0.0
            if seed_center is not None:
                x, y = seed_center
                contains_seed = float(
                    region[0] <= x <= region[2] and region[1] <= y <= region[3]
                )
            global_score = 0.5
            if SpatialFrame.GLOBAL_ABSOLUTE in context.graph.spatial_frames:
                global_score = absolute_position_score(
                    region, context.graph.global_position
                )
            overlap = box_iou(region, seed.bbox) if seed.bbox is not None else 0.0
            # Global constraints are applied in the original image frame. Otherwise,
            # inspect regions not containing the current hypothesis first.
            return (
                -global_score
                if SpatialFrame.GLOBAL_ABSOLUTE in context.graph.spatial_frames
                else contains_seed,
                overlap,
                region[1] + region[0] * 0.01,
            )

        return sorted(regions, key=priority)[:limit]

    def run(
        self,
        context: AgentContext,
        candidate_id: str,
        mode: str = "global_competition_probe",
    ) -> AgentResult:
        if mode not in {"global_competition_probe", "global_search", "local_verify"}:
            raise ValueError(f"Unsupported TargetAgent mode: {mode}")
        query = context.graph.local_target_query
        observation = Observation(
            observation_id=f"{candidate_id}_target_full",
            view_type="full_image_target_probe",
            preserves_context=True,
        )
        candidate = context.grounder.ground(
            context.image,
            query,
            candidate_id,
            self.name,
            observation,
        )
        candidate.hypothesis_id = candidate.candidate_id
        candidate.target_consistency = 0.5 if candidate.bbox is not None else 0.0
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=self.name,
            action=mode,
            input={
                "query": query,
                "query_scope": "target_clause",
                "target": context.graph.target,
                "attributes": context.graph.attributes,
                "observation_id": observation.observation_id,
            },
            output={"candidate": to_jsonable(candidate)},
            evidence={
                "parse_ok": candidate.parse_ok,
                "bbox_token_confidence": candidate.bbox_token_confidence,
                "hypothesis_id": candidate.hypothesis_id,
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return AgentResult(call=call, candidates=[candidate], evidence=call.evidence)

    def run_transformed_view(
        self,
        context: AgentContext,
        candidate_id: str,
        region: list[float],
    ) -> AgentResult:
        query = context.graph.local_target_query
        observation = Observation(
            observation_id=f"{candidate_id}_target_tile",
            view_type="target_search_tile",
            crop_region=region,
            transform=(
                f"overlapping_grid_{context.config.target_tile_grid}x"
                f"{context.config.target_tile_grid}"
            ),
            preserves_context=False,
        )
        crop = _crop_image(context.image, region)
        candidate = context.grounder.ground(
            crop,
            query,
            candidate_id,
            self.name,
            observation,
        )
        local_bbox = candidate.bbox
        candidate.bbox = map_from_crop(candidate.bbox, region)
        candidate.parse_ok = candidate.bbox is not None
        candidate.hypothesis_id = candidate.candidate_id
        candidate.target_consistency = 0.5 if candidate.bbox is not None else 0.0
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=self.name,
            action="transformed_view_target_search",
            input={
                "query": query,
                "query_scope": "target_clause_without_global_coordinates",
                "crop_region": region,
                "global_constraints_reapplied_after_mapping": True,
            },
            output={
                "local_bbox": local_bbox,
                "global_bbox": candidate.bbox,
                "candidate": to_jsonable(candidate),
            },
            evidence={
                "parse_ok": candidate.parse_ok,
                "bbox_token_confidence": candidate.bbox_token_confidence,
                "hypothesis_id": candidate.hypothesis_id,
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return AgentResult(call=call, candidates=[candidate], evidence=call.evidence)
