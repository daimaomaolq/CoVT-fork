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
        query_encoder_type: str = "transformer",
        query_layers: int = 2,
        num_scale_bins: int = 3,
        max_sam_tokens: int = 64,
        max_dino_tokens: int = 2048,
        max_query_tokens: int = 64,
        lm_query_dim: int = 0,
        max_lm_query_tokens: int = 64,
        category_vocab_size: int = 32,
        region_vocab_size: int = 64,
        rule_vocab_size: int = 256,
        use_query_metadata: bool = True,
        use_output_query_proj: bool = True,
        anchor_delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if query_encoder_type not in {"mean", "transformer"}:
            raise ValueError("query_encoder_type must be 'mean' or 'transformer'")
        self.num_region_queries = num_region_queries
        self.num_scale_bins = num_scale_bins
        self.max_sam_tokens = max_sam_tokens
        self.max_dino_tokens = max_dino_tokens
        self.max_query_tokens = max_query_tokens
        self.lm_query_dim = lm_query_dim
        self.max_lm_query_tokens = max_lm_query_tokens
        self.query_encoder_type = query_encoder_type
        self.use_query_metadata = use_query_metadata
        self.use_output_query_proj = use_output_query_proj
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
        self.query_position = nn.Parameter(torch.randn(max_query_tokens, hidden_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_transformer = nn.TransformerEncoder(encoder_layer, num_layers=query_layers)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.category_embedding = nn.Embedding(category_vocab_size, hidden_dim, padding_idx=0)
        self.scale_embedding = nn.Embedding(num_scale_bins + 1, hidden_dim, padding_idx=0)
        self.region_embedding = nn.Embedding(region_vocab_size, hidden_dim, padding_idx=0)
        self.rule_embedding = nn.Embedding(rule_vocab_size, hidden_dim, padding_idx=0)
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if lm_query_dim > 0:
            self.lm_query_position = nn.Parameter(torch.randn(max_lm_query_tokens, hidden_dim) * 0.01)
            self.lm_query_proj = nn.Sequential(
                nn.LayerNorm(lm_query_dim),
                nn.Linear(lm_query_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.lm_query_pool = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.region_to_lm = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.lm_region_ffn = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
        else:
            self.lm_query_position = None
            self.lm_query_proj = None
            self.lm_query_pool = None
            self.region_to_lm = None
            self.lm_region_ffn = None

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
        self.output_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
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

    def _query_position_encoding(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self._position_encoding(self.query_position, token_count).to(device=device, dtype=dtype)

    def _lm_query_position_encoding(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.lm_query_position is None:
            raise RuntimeError("Language query position encoding requested but lm_query_dim is disabled.")
        return self._position_encoding(self.lm_query_position, token_count).to(device=device, dtype=dtype)

    def _encode_query(
        self,
        query_tokens: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        category_ids: torch.Tensor | None = None,
        scale_labels: torch.Tensor | None = None,
        region_ids: torch.Tensor | None = None,
        rule_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_tokens is None:
            query_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            query_tokens = query_tokens.to(device)

        query_emb = self.query_embedding(query_tokens)
        mask = query_tokens != 0
        if not bool(mask.any(dim=1).all()):
            mask = mask.clone()
            mask[~mask.any(dim=1), 0] = True
        if self.query_encoder_type == "transformer":
            query_emb = query_emb + self._query_position_encoding(query_tokens.shape[1], query_emb.dtype, device)
            encoded = self.query_transformer(query_emb, src_key_padding_mask=~mask)
        else:
            encoded = query_emb

        lengths = mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / lengths
        text_context = self.query_encoder(pooled)

        if not self.use_query_metadata:
            return text_context

        if category_ids is None:
            category_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            category_ids = category_ids.to(device).clamp(min=0, max=self.category_embedding.num_embeddings - 1)
        if scale_labels is None:
            scale_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            scale_ids = (scale_labels.to(device) + 1).clamp(min=0, max=self.scale_embedding.num_embeddings - 1)
        if region_ids is None:
            region_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            region_ids = region_ids.to(device).clamp(min=0, max=self.region_embedding.num_embeddings - 1)
        if rule_ids is None:
            rule_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            rule_ids = rule_ids.to(device).clamp(min=0, max=self.rule_embedding.num_embeddings - 1)

        metadata_context = (
            self.category_embedding(category_ids)
            + self.scale_embedding(scale_ids)
            + self.region_embedding(region_ids)
            + self.rule_embedding(rule_ids)
        )
        return text_context + self.metadata_encoder(metadata_context)

    def _encode_lm_query(
        self,
        lm_query_hidden: torch.Tensor | None,
        lm_query_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.lm_query_proj is None or self.lm_query_pool is None or lm_query_hidden is None:
            return None, None, None

        dtype = next(self.lm_query_proj.parameters()).dtype
        lm_query_hidden = lm_query_hidden.to(device=device, dtype=dtype)
        if lm_query_hidden.ndim != 3:
            raise ValueError(f"Expected lm_query_hidden shape [batch, tokens, dim], got {list(lm_query_hidden.shape)}")
        if lm_query_hidden.shape[0] != batch_size:
            raise ValueError(
                f"lm_query_hidden batch size {lm_query_hidden.shape[0]} does not match visual batch size {batch_size}"
            )

        if lm_query_mask is None:
            lm_query_mask = torch.ones(
                lm_query_hidden.shape[:2],
                dtype=torch.bool,
                device=device,
            )
        else:
            lm_query_mask = lm_query_mask.to(device=device, dtype=torch.bool)
        if lm_query_mask.shape != lm_query_hidden.shape[:2]:
            raise ValueError(
                "lm_query_mask shape must match lm_query_hidden[:2], "
                f"got {list(lm_query_mask.shape)} vs {list(lm_query_hidden.shape[:2])}"
            )
        if not bool(lm_query_mask.any(dim=1).all()):
            lm_query_mask = lm_query_mask.clone()
            lm_query_mask[~lm_query_mask.any(dim=1), 0] = True

        lm_tokens = self.lm_query_proj(lm_query_hidden)
        lm_tokens = lm_tokens + self._lm_query_position_encoding(lm_tokens.shape[1], lm_tokens.dtype, device)
        lengths = lm_query_mask.unsqueeze(-1).sum(dim=1).clamp(min=1)
        pooled = (lm_tokens * lm_query_mask.unsqueeze(-1)).sum(dim=1) / lengths
        lm_context = self.lm_query_pool(pooled)
        return lm_tokens, lm_query_mask, lm_context

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
        lm_query_hidden: torch.Tensor | None = None,
        lm_query_mask: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
        scale_labels: torch.Tensor | None = None,
        region_ids: torch.Tensor | None = None,
        rule_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = sam_tokens.shape[0]
        device = sam_tokens.device

        sam_pos = self._position_encoding(self.sam_position, sam_tokens.shape[1]).to(sam_tokens.dtype)
        dino_pos = self._position_encoding(self.dino_position, dino_tokens.shape[1]).to(dino_tokens.dtype)
        sam_visual = self.sam_proj(sam_tokens) + self.sam_type + sam_pos
        dino_visual = self.dino_proj(dino_tokens) + self.dino_type + dino_pos
        visual_tokens = self.visual_norm(torch.cat([sam_visual, dino_visual], dim=1))

        query_context = self._encode_query(
            query_tokens,
            batch_size,
            device,
            category_ids=category_ids,
            scale_labels=scale_labels,
            region_ids=region_ids,
            rule_ids=rule_ids,
        )
        lm_tokens, lm_mask, lm_context = self._encode_lm_query(
            lm_query_hidden,
            lm_query_mask,
            batch_size,
            device,
        )
        if lm_context is not None:
            query_context = query_context + lm_context
        region_queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)
        anchor_context = self.anchor_encoder(self.anchor_boxes.to(device=device, dtype=region_queries.dtype))
        region_queries = region_queries + query_context.unsqueeze(1) + anchor_context.unsqueeze(0)
        if lm_tokens is not None and lm_mask is not None and self.region_to_lm is not None and self.lm_region_ffn is not None:
            lm_attended, _ = self.region_to_lm(
                query=region_queries,
                key=lm_tokens,
                value=lm_tokens,
                key_padding_mask=~lm_mask,
                need_weights=False,
            )
            region_queries = region_queries + lm_attended
            region_queries = region_queries + self.lm_region_ffn(region_queries)

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
        if self.use_output_query_proj:
            region_features = region_features + self.output_query_proj(query_context).unsqueeze(1)

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
            "shared_feature": region_features.mean(dim=1),
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
