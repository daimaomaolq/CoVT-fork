from __future__ import annotations

from ..schema import AgentCall, Observation, to_jsonable
from .base import AgentContext, AgentResult


class TargetAgent:
    name = "TargetAgent"

    def run(
        self,
        context: AgentContext,
        candidate_id: str,
        mode: str = "global_search",
    ) -> AgentResult:
        if mode not in {"global_search", "local_verify"}:
            raise ValueError(f"Unsupported TargetAgent mode: {mode}")
        query = context.graph.local_target_query
        observation = Observation(
            observation_id=f"{candidate_id}_target_full",
            view_type="full_image",
            preserves_context=True,
        )
        candidate = context.grounder.ground(
            context.image,
            query,
            candidate_id,
            self.name,
            observation,
        )
        # A specialist cannot self-certify semantic correctness; fusion derives
        # target consistency from agreement with independent observations.
        candidate.target_consistency = 0.5 if candidate.bbox is not None else 0.0
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=self.name,
            action=mode,
            input={
                "query": query,
                "target": context.graph.target,
                "attributes": context.graph.attributes,
                "observation_id": observation.observation_id,
            },
            output={"candidate": to_jsonable(candidate)},
            evidence={
                "parse_ok": candidate.parse_ok,
                "bbox_token_confidence": candidate.bbox_token_confidence,
            },
            model_call=True,
            perception_call=True,
            latency_ms=candidate.latency_ms,
        )
        return AgentResult(call=call, candidates=[candidate], evidence=call.evidence)
