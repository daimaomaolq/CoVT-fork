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
        self.choice_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.choice_scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
        choice_tokens: torch.Tensor | None = None,
        choice_mask: torch.Tensor | None = None,
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
            if choice_tokens is not None:
                output["choice_logits"] = self.score_choices(shared, choice_tokens, choice_mask)
        if task in {"caption", "multitask"}:
            output["caption_embedding"] = F.normalize(self.caption_embedding_head(shared), dim=-1)
        return output

    def score_choices(
        self,
        shared: torch.Tensor,
        choice_tokens: torch.Tensor,
        choice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if choice_tokens.ndim != 3:
            raise ValueError(f"Expected choice_tokens [batch, choices, tokens], got {list(choice_tokens.shape)}")
        batch_size, choice_count, token_count = choice_tokens.shape
        flat_tokens = choice_tokens.reshape(batch_size * choice_count, token_count).to(shared.device)
        token_mask = flat_tokens != 0
        if not bool(token_mask.any(dim=1).all()):
            token_mask = token_mask.clone()
            token_mask[~token_mask.any(dim=1), 0] = True
        choice_emb = self.grounding.query_embedding(flat_tokens)
        lengths = token_mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
        choice_context = (choice_emb * token_mask.unsqueeze(-1)).sum(dim=1) / lengths
        choice_context = self.choice_encoder(choice_context).reshape(batch_size, choice_count, -1)
        shared_expanded = shared.unsqueeze(1).expand(-1, choice_count, -1)
        features = torch.cat(
            [
                shared_expanded,
                choice_context,
                shared_expanded * choice_context,
            ],
            dim=-1,
        )
        logits = self.choice_scorer(features).squeeze(-1)
        if choice_mask is not None:
            logits = logits.masked_fill(~choice_mask.to(shared.device, dtype=torch.bool), -1e4)
        return logits
