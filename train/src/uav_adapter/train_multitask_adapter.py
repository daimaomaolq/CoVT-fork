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

from uav_adapter.multitask_dataset import UAVITMultiTaskTokenDataset, multitask_collate
from uav_adapter.multitask_model import UAVMultiTaskAdapter
from uav_adapter.train_adapter import (
    anchor_score_targets,
    box_iou_xyxy,
    candidate_iou_xyxy,
    candidate_l1_distance,
    generalized_box_iou_xyxy,
    normalize_box_order,
    resolve_device,
    weighted_aux_bbox_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UAVMultiTaskAdapter on UAVIT-1M multitask JSONLs.")
    parser.add_argument("--train-index", action="append", required=True)
    parser.add_argument("--val-index", action="append", default=None)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--lm-query-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-region-queries", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--query-max-len", type=int, default=64)
    parser.add_argument("--query-vocab-size", type=int, default=8192)
    parser.add_argument("--max-lm-query-tokens", type=int, default=64)
    parser.add_argument("--query-encoder-type", default="transformer", choices=("mean", "transformer"))
    parser.add_argument("--query-layers", type=int, default=2)
    parser.add_argument("--category-vocab-size", type=int, default=32)
    parser.add_argument("--region-vocab-size", type=int, default=64)
    parser.add_argument("--rule-vocab-size", type=int, default=256)
    parser.add_argument("--answer-vocab-size", type=int, default=4096)
    parser.add_argument("--caption-embedding-dim", type=int, default=256)
    parser.add_argument("--caption-temperature", type=float, default=0.07)
    parser.add_argument("--disable-query-metadata", action="store_true")
    parser.add_argument("--max-sam-tokens", type=int, default=64)
    parser.add_argument("--max-dino-tokens", type=int, default=2048)
    parser.add_argument("--anchor-delta-scale", type=float, default=1.0)
    parser.add_argument("--grounding-loss-weight", type=float, default=1.0)
    parser.add_argument("--answer-loss-weight", type=float, default=0.5)
    parser.add_argument("--caption-loss-weight", type=float, default=0.2)
    parser.add_argument("--rank-loss-weight", type=float, default=0.3)
    parser.add_argument("--scale-loss-weight", type=float, default=0.05)
    parser.add_argument("--aux-bbox-loss-weight", type=float, default=0.05)
    parser.add_argument("--score-loss-weight", type=float, default=0.2)
    parser.add_argument("--giou-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-loss-weight", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_loader(args: argparse.Namespace, indexes: list[str] | None, shuffle: bool) -> DataLoader | None:
    if not indexes:
        return None
    dataset = UAVITMultiTaskTokenDataset(
        indexes,
        args.token_dir,
        lm_query_dir=args.lm_query_dir,
        query_max_len=args.query_max_len,
        query_vocab_size=args.query_vocab_size,
        answer_vocab_size=args.answer_vocab_size,
        caption_embedding_dim=args.caption_embedding_dim,
        max_lm_query_tokens=args.max_lm_query_tokens,
        region_vocab_size=args.region_vocab_size,
        rule_vocab_size=args.rule_vocab_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=multitask_collate,
    )


def grounding_loss(pred: dict[str, torch.Tensor], bbox: torch.Tensor, scale_label: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    candidates = normalize_box_order(pred["candidate_bboxes"])
    anchors = pred["anchor_boxes"].unsqueeze(0).expand_as(candidates)
    anchor_dist = candidate_l1_distance(anchors, bbox)
    anchor_iou = candidate_iou_xyxy(anchors, bbox)
    anchor_cost = anchor_dist - anchor_iou * 0.25
    best_idx = anchor_cost.detach().argmin(dim=1)
    batch_indices = torch.arange(bbox.shape[0], device=bbox.device)
    matched_bbox = candidates[batch_indices, best_idx]
    bbox_loss = F.smooth_l1_loss(matched_bbox, bbox)
    giou_loss = 1.0 - generalized_box_iou_xyxy(matched_bbox, bbox).mean()
    aux_bbox_loss = weighted_aux_bbox_loss(candidates, bbox, anchor_dist)
    rank_loss = F.cross_entropy(pred["candidate_scores"], best_idx)
    scale_logits = pred["candidate_scale_logits"][batch_indices, best_idx]
    scale_loss = F.cross_entropy(scale_logits, scale_label)
    score_loss = F.binary_cross_entropy_with_logits(
        pred["candidate_scores"],
        anchor_score_targets(anchor_dist, best_idx),
    )
    delta_loss = pred["candidate_bbox_deltas"].pow(2).mean()
    loss = (
        bbox_loss
        + args.giou_loss_weight * giou_loss
        + args.aux_bbox_loss_weight * aux_bbox_loss
        + args.rank_loss_weight * rank_loss
        + args.scale_loss_weight * scale_loss
        + args.score_loss_weight * score_loss
        + args.delta_loss_weight * delta_loss
    )
    return loss, {
        "ground_bbox": bbox_loss,
        "ground_giou": giou_loss,
        "ground_aux_bbox": aux_bbox_loss,
        "ground_rank": rank_loss,
        "ground_scale": scale_loss,
        "ground_score": score_loss,
        "ground_delta": delta_loss,
    }


def caption_loss(pred_embedding: torch.Tensor, target_embedding: torch.Tensor, temperature: float) -> torch.Tensor:
    target_embedding = F.normalize(target_embedding, dim=-1)
    if pred_embedding.shape[0] == 1:
        return 1.0 - (pred_embedding * target_embedding).sum(dim=-1).mean()
    logits = pred_embedding @ target_embedding.T / max(temperature, 1e-4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def subset_batch(batch: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    keys = [
        "sam_tokens",
        "dino_tokens",
        "query_tokens",
        "category_id",
        "region_id",
        "query_rule_id",
        "bbox",
        "scale_label",
        "answer_id",
        "caption_target",
    ]
    out = {key: batch[key][mask] for key in keys}
    if "lm_query_hidden" in batch:
        out["lm_query_hidden"] = batch["lm_query_hidden"][mask]
        out["lm_query_mask"] = batch["lm_query_mask"][mask]
    return out


def forward_task(model: UAVMultiTaskAdapter, batch: dict[str, torch.Tensor], task: str) -> dict[str, torch.Tensor]:
    return model(
        batch["sam_tokens"],
        batch["dino_tokens"],
        query_tokens=batch["query_tokens"],
        lm_query_hidden=batch.get("lm_query_hidden"),
        lm_query_mask=batch.get("lm_query_mask"),
        category_ids=batch["category_id"],
        scale_labels=batch["scale_label"],
        region_ids=batch["region_id"],
        rule_ids=batch["query_rule_id"],
        task=task,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
    return moved


@torch.no_grad()
def evaluate(model: UAVMultiTaskAdapter, loader: DataLoader, device: torch.device, args) -> dict[str, float]:
    model.eval()
    totals = {"grounding": 0, "answer": 0, "caption": 0}
    sums = {"ground_iou": 0.0, "ground_acc50": 0.0, "answer_acc": 0.0, "caption_cos": 0.0}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        ground_mask = batch["has_grounding"].bool()
        answer_mask = batch["has_answer"].bool()
        caption_mask = batch["has_caption"].bool()
        if bool(ground_mask.any()):
            sub = subset_batch(batch, ground_mask)
            pred = forward_task(model, sub, task="grounding")
            pred_bbox = normalize_box_order(pred["bbox"])
            iou = box_iou_xyxy(pred_bbox, sub["bbox"])
            totals["grounding"] += iou.shape[0]
            sums["ground_iou"] += float(iou.sum().cpu())
            sums["ground_acc50"] += float((iou >= 0.5).sum().cpu())
        if bool(answer_mask.any()):
            sub = subset_batch(batch, answer_mask)
            pred = forward_task(model, sub, task="answer")
            answer_pred = pred["answer_logits"].argmax(dim=1)
            totals["answer"] += answer_pred.shape[0]
            sums["answer_acc"] += float((answer_pred == sub["answer_id"]).sum().cpu())
        if bool(caption_mask.any()):
            sub = subset_batch(batch, caption_mask)
            pred = forward_task(model, sub, task="caption")
            target = F.normalize(sub["caption_target"], dim=-1)
            cos = (pred["caption_embedding"] * target).sum(dim=-1)
            totals["caption"] += cos.shape[0]
            sums["caption_cos"] += float(cos.sum().cpu())
    return {
        "ground_miou": sums["ground_iou"] / max(totals["grounding"], 1),
        "ground_acc50": sums["ground_acc50"] / max(totals["grounding"], 1),
        "answer_acc_hash": sums["answer_acc"] / max(totals["answer"], 1),
        "caption_cos": sums["caption_cos"] / max(totals["caption"], 1),
        "ground_samples": totals["grounding"],
        "answer_samples": totals["answer"],
        "caption_samples": totals["caption"],
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args, args.train_index, shuffle=True)
    val_loader = make_loader(args, args.val_index, shuffle=False)
    first_batch = next(iter(train_loader))
    model = UAVMultiTaskAdapter(
        sam_dim=first_batch["sam_tokens"].shape[-1],
        dino_dim=first_batch["dino_tokens"].shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_region_queries=args.num_region_queries,
        num_heads=args.num_heads,
        query_vocab_size=args.query_vocab_size,
        query_encoder_type=args.query_encoder_type,
        query_layers=args.query_layers,
        max_query_tokens=args.query_max_len,
        lm_query_dim=first_batch["lm_query_hidden"].shape[-1] if "lm_query_hidden" in first_batch else 0,
        max_lm_query_tokens=args.max_lm_query_tokens,
        category_vocab_size=args.category_vocab_size,
        region_vocab_size=args.region_vocab_size,
        rule_vocab_size=args.rule_vocab_size,
        use_query_metadata=not args.disable_query_metadata,
        use_output_query_proj=True,
        max_sam_tokens=args.max_sam_tokens,
        max_dino_tokens=args.max_dino_tokens,
        anchor_delta_scale=args.anchor_delta_scale,
        answer_vocab_size=args.answer_vocab_size,
        caption_embedding_dim=args.caption_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "model_type": "UAVMultiTaskAdapter",
        "sam_dim": first_batch["sam_tokens"].shape[-1],
        "dino_dim": first_batch["dino_tokens"].shape[-1],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_region_queries": args.num_region_queries,
        "num_heads": args.num_heads,
        "query_vocab_size": args.query_vocab_size,
        "query_max_len": args.query_max_len,
        "lm_query_dim": first_batch["lm_query_hidden"].shape[-1] if "lm_query_hidden" in first_batch else 0,
        "max_lm_query_tokens": args.max_lm_query_tokens,
        "query_encoder_type": args.query_encoder_type,
        "query_layers": args.query_layers,
        "category_vocab_size": args.category_vocab_size,
        "region_vocab_size": args.region_vocab_size,
        "rule_vocab_size": args.rule_vocab_size,
        "use_query_metadata": not args.disable_query_metadata,
        "use_output_query_proj": True,
        "max_sam_tokens": args.max_sam_tokens,
        "max_dino_tokens": args.max_dino_tokens,
        "anchor_delta_scale": args.anchor_delta_scale,
        "answer_vocab_size": args.answer_vocab_size,
        "caption_embedding_dim": args.caption_embedding_dim,
    }
    history = []

    def save_checkpoint(name: str) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "history": history,
                "args": vars(args),
            },
            output_dir / name,
        )

    best_score = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        component_sums: dict[str, float] = {}
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            loss = torch.zeros((), device=device)
            batch_items = batch["sam_tokens"].shape[0]
            ground_mask = batch["has_grounding"].bool()
            answer_mask = batch["has_answer"].bool()
            caption_mask = batch["has_caption"].bool()
            if bool(ground_mask.any()):
                sub = subset_batch(batch, ground_mask)
                pred = forward_task(model, sub, task="grounding")
                item_loss, comps = grounding_loss(pred, sub["bbox"], sub["scale_label"], args)
                loss = loss + args.grounding_loss_weight * item_loss
                for key, value in comps.items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().cpu()) * sub["bbox"].shape[0]
            if bool(answer_mask.any()):
                sub = subset_batch(batch, answer_mask)
                pred = forward_task(model, sub, task="answer")
                item_loss = F.cross_entropy(pred["answer_logits"], sub["answer_id"])
                loss = loss + args.answer_loss_weight * item_loss
                component_sums["answer_ce"] = component_sums.get("answer_ce", 0.0) + float(item_loss.detach().cpu()) * sub["answer_id"].shape[0]
            if bool(caption_mask.any()):
                sub = subset_batch(batch, caption_mask)
                pred = forward_task(model, sub, task="caption")
                item_loss = caption_loss(pred["caption_embedding"], sub["caption_target"], args.caption_temperature)
                loss = loss + args.caption_loss_weight * item_loss
                component_sums["caption_nce"] = component_sums.get("caption_nce", 0.0) + float(item_loss.detach().cpu()) * sub["caption_target"].shape[0]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_items
            total_items += batch_items

        row = {"epoch": epoch, "train_loss": total_loss / max(total_items, 1)}
        row.update({f"train_{key}": value / max(total_items, 1) for key, value in component_sums.items()})
        if val_loader is not None:
            row.update({f"val_{key}": value for key, value in evaluate(model, val_loader, device, args).items()})
            score = row.get("val_ground_acc50", 0.0) + row.get("val_answer_acc_hash", 0.0) + row.get("val_caption_cos", 0.0)
            if score > best_score:
                best_score = score
                save_checkpoint("best_multitask.pt")
                row["best_checkpoint"] = str(output_dir / "best_multitask.pt")
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    save_checkpoint("uav_multitask_adapter.pt")
    print(json.dumps({"status": "ok", "checkpoint": str(output_dir / "uav_multitask_adapter.pt")}, indent=2))


if __name__ == "__main__":
    main()
