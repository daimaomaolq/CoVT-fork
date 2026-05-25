from __future__ import annotations

import math

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
        num_region_queries: int = 32,
        num_heads: int = 8,
        query_vocab_size: int = 8192,
        num_scale_bins: int = 3,
        max_sam_tokens: int = 64,
        max_dino_tokens: int = 2048,
        anchor_delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_region_queries = num_region_queries
        self.num_scale_bins = num_scale_bins
        self.max_sam_tokens = max_sam_tokens
        self.max_dino_tokens = max_dino_tokens
        self.anchor_delta_scale = anchor_delta_scale
        self.register_buffer("anchor_boxes", self._make_anchor_boxes(num_region_queries), persistent=False)

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
        self.sam_position = nn.Parameter(torch.randn(max_sam_tokens, hidden_dim) * 0.01)
        self.dino_position = nn.Parameter(torch.randn(max_dino_tokens, hidden_dim) * 0.01)
        self.region_queries = nn.Parameter(torch.randn(num_region_queries, hidden_dim) * 0.02)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
        nn.init.zeros_(self.bbox_head.weight)
        nn.init.zeros_(self.bbox_head.bias)

    @staticmethod
    def _make_anchor_boxes(num_boxes: int) -> torch.Tensor:
        grid = max(1, math.ceil(math.sqrt(num_boxes)))
        if num_boxes == 1:
            selected_cells = [(grid // 2, grid // 2)]
        else:
            cells = [(x_idx, y_idx) for y_idx in range(grid) for x_idx in range(grid)]
            selected_cells = [
                cells[round(index * (len(cells) - 1) / (num_boxes - 1))]
                for index in range(num_boxes)
            ]

        boxes = []
        base = 0.82 / grid
        scales = (0.45, 0.70, 1.00, 1.35)
        aspect_ratios = (0.60, 1.00, 1.70)
        for index, (x_idx, y_idx) in enumerate(selected_cells):
            cx = (x_idx + 0.5) / grid
            cy = (y_idx + 0.5) / grid
            scale = scales[index % len(scales)]
            aspect = aspect_ratios[(index // len(scales)) % len(aspect_ratios)]
            side = base * scale
            width = min(side * math.sqrt(aspect), 0.80)
            height = min(side / math.sqrt(aspect), 0.80)
            boxes.append(
                [
                    max(cx - width * 0.5, 0.0),
                    max(cy - height * 0.5, 0.0),
                    min(cx + width * 0.5, 1.0),
                    min(cy + height * 0.5, 1.0),
                ]
            )
        return torch.tensor(boxes, dtype=torch.float32)

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

    def _position_encoding(self, table: torch.Tensor, token_count: int) -> torch.Tensor:
        if token_count <= table.shape[0]:
            return table[:token_count].unsqueeze(0)
        source = table.transpose(0, 1).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            source,
            size=token_count,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0).transpose(0, 1).unsqueeze(0)

    def _decode_anchor_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        anchors = self.anchor_boxes.to(device=deltas.device, dtype=deltas.dtype)
        anchor_logits = torch.logit(anchors.clamp(min=1e-4, max=1.0 - 1e-4))
        refined_logits = anchor_logits.unsqueeze(0) + deltas * self.anchor_delta_scale
        return refined_logits.sigmoid()

    def forward(
        self,
        sam_tokens: torch.Tensor,
        dino_tokens: torch.Tensor,
        query_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = sam_tokens.shape[0]
        device = sam_tokens.device

        sam_pos = self._position_encoding(self.sam_position, sam_tokens.shape[1]).to(sam_tokens.dtype)
        dino_pos = self._position_encoding(self.dino_position, dino_tokens.shape[1]).to(dino_tokens.dtype)
        sam_visual = self.sam_proj(sam_tokens) + self.sam_type + sam_pos
        dino_visual = self.dino_proj(dino_tokens) + self.dino_type + dino_pos
        visual_tokens = self.visual_norm(torch.cat([sam_visual, dino_visual], dim=1))

        query_context = self._encode_query(query_tokens, batch_size, device)
        region_queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)
        anchor_context = self.anchor_encoder(self.anchor_boxes.to(device=device, dtype=region_queries.dtype))
        region_queries = region_queries + query_context.unsqueeze(1) + anchor_context.unsqueeze(0)

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

        candidate_bbox_deltas = self.bbox_head(region_features)
        candidate_bboxes = self._decode_anchor_deltas(candidate_bbox_deltas)
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
            "anchor_boxes": self.anchor_boxes.to(device=device, dtype=candidate_bboxes.dtype),
            "candidate_bbox_deltas": candidate_bbox_deltas,
            "candidate_bboxes": candidate_bboxes,
            "candidate_scores": candidate_scores,
            "topk_bboxes": top_bboxes,
            "topk_scores": top_scores,
            "scale_logits": best_scale_logits,
            "candidate_scale_logits": scale_logits,
            "visual_attention": attn_weights,
        }
