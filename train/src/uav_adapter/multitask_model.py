from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_adapter.model import UAVPerceptionAdapter


class UAVMultiTaskAdapter(nn.Module):
    """Shared CoVT visual-token adapter with grounding, answer, and caption heads.

    The grounding branch reuses the existing UAVPerceptionAdapter. The extra heads
    are intentionally lightweight so the project can train one adapter family for
    UAVIT-1M tasks while keeping CoVT frozen.
    """

    def __init__(
        self,
        *,
        answer_vocab_size: int = 4096,
        caption_embedding_dim: int = 256,
        **grounding_kwargs,
    ) -> None:
        super().__init__()
        hidden_dim = int(grounding_kwargs.get("hidden_dim", 256))
        dropout = float(grounding_kwargs.get("dropout", 0.1))
        self.grounding = UAVPerceptionAdapter(**grounding_kwargs)
        self.answer_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, answer_vocab_size),
        )
        self.caption_embedding_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, caption_embedding_dim),
        )

    def forward(
        self,
        sam_tokens: torch.Tensor,
        dino_tokens: torch.Tensor,
        query_tokens: torch.Tensor | None = None,
        lm_query_hidden: torch.Tensor | None = None,
        lm_query_mask: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
        scale_labels: torch.Tensor | None = None,
        region_ids: torch.Tensor | None = None,
        rule_ids: torch.Tensor | None = None,
        task: str = "grounding",
    ) -> dict[str, torch.Tensor]:
        output = self.grounding(
            sam_tokens,
            dino_tokens,
            query_tokens=query_tokens,
            lm_query_hidden=lm_query_hidden,
            lm_query_mask=lm_query_mask,
            category_ids=category_ids,
            scale_labels=scale_labels,
            region_ids=region_ids,
            rule_ids=rule_ids,
        )
        shared = output["shared_feature"]
        if task in {"answer", "multitask"}:
            output["answer_logits"] = self.answer_head(shared)
        if task in {"caption", "multitask"}:
            output["caption_embedding"] = F.normalize(self.caption_embedding_head(shared), dim=-1)
        return output
