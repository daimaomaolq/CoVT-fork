from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uav_adapter.dataset import TokenGroundingDataset
from uav_adapter.model import UAVPerceptionAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the query-conditioned UAVPerceptionAdapter.")
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/checkpoints/uav_adapter")
    parser.add_argument("--bbox-key", default="bbox_norm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-region-queries", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--query-max-len", type=int, default=32)
    parser.add_argument("--query-vocab-size", type=int, default=8192)
    parser.add_argument("--rank-loss-weight", type=float, default=0.2)
    parser.add_argument("--scale-loss-weight", type=float, default=0.1)
    parser.add_argument("--aux-bbox-loss-weight", type=float, default=0.2)
    parser.add_argument("--center-size-loss-weight", type=float, default=0.5)
    parser.add_argument("--score-loss-weight", type=float, default=0.1)
    parser.add_argument("--bbox-logit-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def collate(batch):
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "sam_tokens": torch.stack([item["sam_tokens"] for item in batch]),
        "dino_tokens": torch.stack([item["dino_tokens"] for item in batch]),
        "query_tokens": torch.stack([item["query_tokens"] for item in batch]),
        "scale_label": torch.stack([item["scale_label"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
    }


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def normalize_box_order(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def candidate_iou_xyxy(candidate_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    target_boxes = target_boxes.unsqueeze(1)
    lt = torch.maximum(candidate_boxes[..., :2], target_boxes[..., :2])
    rb = torch.minimum(candidate_boxes[..., 2:], target_boxes[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (candidate_boxes[..., 2] - candidate_boxes[..., 0]).clamp(min=0) * (
        candidate_boxes[..., 3] - candidate_boxes[..., 1]
    ).clamp(min=0)
    area2 = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=0) * (
        target_boxes[..., 3] - target_boxes[..., 1]
    ).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def box_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return torch.cat([center, size], dim=-1)


def candidate_l1_distance(candidate_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    return torch.abs(candidate_boxes - target_boxes.unsqueeze(1)).mean(dim=-1)


def soft_score_targets(candidate_dist: torch.Tensor) -> torch.Tensor:
    return torch.exp(-candidate_dist.detach() * 12.0).clamp(min=0.0, max=1.0)


@torch.no_grad()
def evaluate(model: UAVPerceptionAdapter, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    iou_sum = 0.0
    acc50 = 0.0
    recall3 = 0.0
    oracle_iou_sum = 0.0
    oracle_acc50 = 0.0
    for batch in loader:
        sam_tokens = batch["sam_tokens"].to(device)
        dino_tokens = batch["dino_tokens"].to(device)
        query_tokens = batch["query_tokens"].to(device)
        bbox = batch["bbox"].to(device)
        pred = model(sam_tokens, dino_tokens, query_tokens=query_tokens)
        pred_bbox = normalize_box_order(pred["bbox"])
        candidates = normalize_box_order(pred["candidate_bboxes"])
        loss = F.l1_loss(pred_bbox, bbox, reduction="sum")
        iou = box_iou_xyxy(pred_bbox, bbox)
        candidate_iou = candidate_iou_xyxy(candidates, bbox)
        oracle_iou = candidate_iou.max(dim=1).values
        topk_count = min(3, candidate_iou.shape[1])
        topk_indices = torch.topk(pred["candidate_scores"], k=topk_count, dim=1).indices
        topk_iou = candidate_iou.gather(dim=1, index=topk_indices)
        batch_size = bbox.shape[0]
        total += batch_size
        loss_sum += float(loss.detach().cpu())
        iou_sum += float(iou.sum().detach().cpu())
        acc50 += float((iou >= 0.5).sum().detach().cpu())
        recall3 += float((topk_iou >= 0.5).any(dim=1).sum().detach().cpu())
        oracle_iou_sum += float(oracle_iou.sum().detach().cpu())
        oracle_acc50 += float((oracle_iou >= 0.5).sum().detach().cpu())
    return {
        "l1": loss_sum / max(total, 1),
        "miou": iou_sum / max(total, 1),
        "acc50": acc50 / max(total, 1),
        "recall3": recall3 / max(total, 1),
        "oracle_miou": oracle_iou_sum / max(total, 1),
        "oracle_acc50": oracle_acc50 / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TokenGroundingDataset(
        args.train_index,
        args.token_dir,
        bbox_key=args.bbox_key,
        query_max_len=args.query_max_len,
        query_vocab_size=args.query_vocab_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = None
    if args.val_index:
        val_dataset = TokenGroundingDataset(
            args.val_index,
            args.token_dir,
            bbox_key=args.bbox_key,
            query_max_len=args.query_max_len,
            query_vocab_size=args.query_vocab_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )

    first_batch = next(iter(train_loader))
    model = UAVPerceptionAdapter(
        sam_dim=first_batch["sam_tokens"].shape[-1],
        dino_dim=first_batch["dino_tokens"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_region_queries=args.num_region_queries,
        num_heads=args.num_heads,
        query_vocab_size=args.query_vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        for batch in train_loader:
            sam_tokens = batch["sam_tokens"].to(device)
            dino_tokens = batch["dino_tokens"].to(device)
            query_tokens = batch["query_tokens"].to(device)
            bbox = batch["bbox"].to(device)
            scale_label = batch["scale_label"].to(device)
            pred = model(sam_tokens, dino_tokens, query_tokens=query_tokens)
            candidates = normalize_box_order(pred["candidate_bboxes"])
            candidate_logits = pred["candidate_bbox_logits"]
            candidate_dist = candidate_l1_distance(candidates, bbox)
            best_idx = candidate_dist.detach().argmin(dim=1)
            batch_indices = torch.arange(bbox.shape[0], device=device)
            matched_bbox = candidates[batch_indices, best_idx]
            matched_logits = candidate_logits[batch_indices, best_idx]
            bbox_loss = F.smooth_l1_loss(matched_bbox, bbox)
            bbox_logit_loss = F.binary_cross_entropy_with_logits(matched_logits, bbox)
            aux_bbox_logit_loss = F.binary_cross_entropy_with_logits(
                candidate_logits,
                bbox.unsqueeze(1).expand_as(candidate_logits),
            )
            aux_bbox_loss = F.smooth_l1_loss(candidates, bbox.unsqueeze(1).expand_as(candidates))
            center_size_loss = F.smooth_l1_loss(box_cxcywh(matched_bbox), box_cxcywh(bbox))
            rank_loss = F.cross_entropy(pred["candidate_scores"], best_idx)
            matched_scale_logits = pred["candidate_scale_logits"][batch_indices, best_idx]
            scale_loss = F.cross_entropy(matched_scale_logits, scale_label)
            score_loss = F.binary_cross_entropy_with_logits(
                pred["candidate_scores"],
                soft_score_targets(candidate_dist),
            )
            loss = (
                bbox_loss
                + args.bbox_logit_loss_weight * bbox_logit_loss
                + args.aux_bbox_loss_weight * aux_bbox_loss
                + args.aux_bbox_loss_weight * aux_bbox_logit_loss
                + args.center_size_loss_weight * center_size_loss
                + args.score_loss_weight * score_loss
                + args.rank_loss_weight * rank_loss
                + args.scale_loss_weight * scale_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = bbox.shape[0]
            total += batch_size
            loss_sum += float(loss.detach().cpu()) * batch_size

        row = {"epoch": epoch, "train_loss": loss_sum / max(total, 1)}
        if val_loader is not None:
            row.update({f"val_{key}": value for key, value in evaluate(model, val_loader, device).items()})
        history.append(row)
        print(json.dumps(row, indent=2))

    checkpoint = {
        "model": model.state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "sam_dim": first_batch["sam_tokens"].shape[-1],
            "dino_dim": first_batch["dino_tokens"].shape[-1],
            "num_region_queries": args.num_region_queries,
            "num_heads": args.num_heads,
            "query_vocab_size": args.query_vocab_size,
            "query_max_len": args.query_max_len,
        },
        "history": history,
        "args": vars(args),
    }
    ckpt_path = output_dir / "uav_adapter.pt"
    torch.save(checkpoint, ckpt_path)
    print(json.dumps({"status": "ok", "checkpoint": str(ckpt_path)}, indent=2))


if __name__ == "__main__":
    main()
