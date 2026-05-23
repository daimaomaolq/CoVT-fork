from __future__ import annotations

import torch
import torch.nn as nn


class UAVPerceptionAdapter(nn.Module):
    """Query-conditioned Seg-DINO grounding adapter for UAV perception.

    The adapter turns CoVT Seg-DINO visual thought tokens into ranked explicit
    region candidates. It keeps CoVT frozen and learns only a lightweight
    grounding module on top of cached tokens.
    """

    def __init__(
        self,
        sam_dim: int = 256,
        dino_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_region_queries: int = 8,
        num_heads: int = 8,
        query_vocab_size: int = 8192,
        num_scale_bins: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_region_queries = num_region_queries
        self.num_scale_bins = num_scale_bins

        self.sam_proj = nn.Sequential(
            nn.LayerNorm(sam_dim),
            nn.Linear(sam_dim, hidden_dim),
            nn.GELU(),
        )
        self.dino_proj = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, hidden_dim),
            nn.GELU(),
        )

        self.query_embedding = nn.Embedding(query_vocab_size, hidden_dim, padding_idx=0)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.sam_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dino_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.region_queries = nn.Parameter(torch.randn(num_region_queries, hidden_dim) * 0.02)

        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.query_to_visual = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.region_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.region_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, num_scale_bins)

    def _encode_query(self, query_tokens: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if query_tokens is None:
            query_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            query_tokens = query_tokens.to(device)

        query_emb = self.query_embedding(query_tokens)
        mask = (query_tokens != 0).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = (query_emb * mask).sum(dim=1) / lengths
        return self.query_encoder(pooled)

    def forward(
        self,
        sam_tokens: torch.Tensor,
        dino_tokens: torch.Tensor,
        query_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = sam_tokens.shape[0]
        device = sam_tokens.device

        sam_visual = self.sam_proj(sam_tokens) + self.sam_type
        dino_visual = self.dino_proj(dino_tokens) + self.dino_type
        visual_tokens = self.visual_norm(torch.cat([sam_visual, dino_visual], dim=1))

        query_context = self._encode_query(query_tokens, batch_size, device)
        region_queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)
        region_queries = region_queries + query_context.unsqueeze(1)

        attended, attn_weights = self.query_to_visual(
            query=region_queries,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=True,
        )
        self_attended, _ = self.region_self_attention(
            query=attended,
            key=attended,
            value=attended,
            need_weights=False,
        )
        region_features = attended + self_attended
        region_features = region_features + self.region_ffn(region_features)

        candidate_bboxes = self.bbox_head(region_features).sigmoid()
        candidate_scores = self.score_head(region_features).squeeze(-1)
        scale_logits = self.scale_head(region_features)

        top_scores, top_indices = torch.topk(
            candidate_scores,
            k=min(3, candidate_scores.shape[1]),
            dim=1,
        )
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, 4)
        top_bboxes = candidate_bboxes.gather(dim=1, index=gather_index)
        best_index = top_indices[:, 0]
        best_bbox = candidate_bboxes[torch.arange(batch_size, device=device), best_index]
        best_score = candidate_scores[torch.arange(batch_size, device=device), best_index]
        best_scale_logits = scale_logits[torch.arange(batch_size, device=device), best_index]

        return {
            "bbox": best_bbox,
            "score": best_score,
            "candidate_bboxes": candidate_bboxes,
            "candidate_scores": candidate_scores,
            "topk_bboxes": top_bboxes,
            "topk_scores": top_scores,
            "scale_logits": best_scale_logits,
            "candidate_scale_logits": scale_logits,
            "visual_attention": attn_weights,
        }
