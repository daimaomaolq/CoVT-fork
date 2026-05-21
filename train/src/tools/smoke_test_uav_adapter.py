from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a minimal UAV perception adapter with cached CoVT Seg-DINO tokens."
    )
    parser.add_argument(
        "--token-cache",
        required=True,
        help="Path to a .pt file saved by extract_seg_dino_tokens.py.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string such as cuda:0 or cpu. Use auto to prefer CUDA when available.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str, torch_module):
    if device_arg == "auto":
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_arg)


def first_tensor(value: Any):
    if hasattr(value, "shape") and hasattr(value, "detach"):
        return value
    if isinstance(value, list):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def get_token_tensor(cache: dict[str, Any], anchor_name: str, preferred_key: str):
    anchor = cache["tokens"][anchor_name]
    tensor = anchor.get(preferred_key)
    if tensor is None:
        tensor = first_tensor(anchor)
    if tensor is None:
        raise ValueError(f"No tensor found for {anchor_name}.{preferred_key}")
    return tensor


def main() -> None:
    args = parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = resolve_device(args.device, torch)
    token_cache = Path(args.token_cache).resolve()
    if not token_cache.exists():
        raise FileNotFoundError(token_cache)

    cache = torch.load(token_cache, map_location="cpu")
    sam_tokens = get_token_tensor(cache, "sam", "attended").float().to(device)
    dino_tokens = get_token_tensor(cache, "dino", "attended").float().to(device)

    class MinimalUAVPerceptionAdapter(nn.Module):
        def __init__(self, sam_dim: int, dino_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.sam_pool = nn.Linear(sam_dim, hidden_dim)
            self.dino_pool = nn.Linear(dino_dim, hidden_dim)
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.bbox_head = nn.Linear(hidden_dim, 4)
            self.score_head = nn.Linear(hidden_dim, 1)

        def forward(self, sam_tokens, dino_tokens):
            sam_feat = self.sam_pool(sam_tokens).mean(dim=1)
            dino_feat = self.dino_pool(dino_tokens).mean(dim=1)
            fused = self.fusion(torch.cat([sam_feat, dino_feat], dim=-1))
            bbox = self.bbox_head(fused).sigmoid()
            score = self.score_head(fused).squeeze(-1)
            return bbox, score

    model = MinimalUAVPerceptionAdapter(
        sam_dim=sam_tokens.shape[-1],
        dino_dim=dino_tokens.shape[-1],
    ).to(device)

    bbox_pred, score_pred = model(sam_tokens, dino_tokens)
    bbox_target = torch.tensor([[0.15, 0.20, 0.75, 0.80]], device=device, dtype=bbox_pred.dtype)
    score_target = torch.ones_like(score_pred)

    bbox_loss = F.l1_loss(bbox_pred, bbox_target.expand_as(bbox_pred))
    score_loss = F.binary_cross_entropy_with_logits(score_pred, score_target)
    loss = bbox_loss + score_loss
    loss.backward()

    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm().cpu())

    summary = {
        "token_cache": str(token_cache),
        "device": str(device),
        "sam_tokens_shape": list(sam_tokens.shape),
        "dino_tokens_shape": list(dino_tokens.shape),
        "bbox_pred_shape": list(bbox_pred.shape),
        "score_pred_shape": list(score_pred.shape),
        "loss": float(loss.detach().cpu()),
        "grad_norm": grad_norm,
        "status": "ok" if grad_norm > 0 else "no_grad",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
