from __future__ import annotations

from ..schema import AgentCall, Observation, to_jsonable
from .base import AgentContext, AgentResult


class ContextAgent:
    name = "ContextAgent"

    def run(self, context: AgentContext, candidate_id: str) -> AgentResult:
        if not context.graph.context:
            call = AgentCall(
                call_id=f"call_{candidate_id}",
                agent=self.name,
                action="context_search",
                input={"context": ""},
                output={"candidate": None},
                evidence={"skipped_reason": "query_has_no_context"},
                status="skipped",
            )
            return AgentResult(call=call, candidates=[], evidence=call.evidence)
        observation = Observation(
            observation_id=f"{candidate_id}_context_full",
            view_type="full_image",
            preserves_context=True,
        )
        candidate = context.grounder.ground(
            context.image,
            context.graph.context,
            candidate_id,
            self.name,
            observation,
        )
        candidate.context_consistency = (
            candidate.bbox_token_confidence if candidate.bbox is not None else 0.0
        )
        call = AgentCall(
            call_id=f"call_{candidate_id}",
            agent=self.name,
            action="context_search",
            input={
                "query": context.graph.context,
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
