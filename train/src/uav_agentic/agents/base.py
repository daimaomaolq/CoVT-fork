from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..grounder import GrounderProtocol
from ..schema import AgentCall, AgenticConfig, Candidate, QueryConstraintGraph


@dataclass
class AgentContext:
    image: Image.Image
    graph: QueryConstraintGraph
    grounder: GrounderProtocol
    config: AgenticConfig
    target_candidates: list[Candidate]
    context_candidates: list[Candidate]


@dataclass
class AgentResult:
    call: AgentCall
    candidates: list[Candidate]
    evidence: dict
