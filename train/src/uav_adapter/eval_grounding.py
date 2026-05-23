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
from torch.utils.data import DataLoader

from uav_adapter.dataset import TokenGroundingDataset
from uav_adapter.model import UAVPerceptionAdapter
from uav_adapter.train_adapter import (
    box_iou_xyxy,
    candidate_iou_xyxy,
    collate,
    normalize_box_order,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the query-conditioned UAVPerceptionAdapter.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bbox-key", default="bbox_norm")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predictions", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    dataset = TokenGroundingDataset(
        args.index,
        args.token_dir,
        bbox_key=args.bbox_key,
        query_max_len=config.get("query_max_len", 32),
        query_vocab_size=config.get("query_vocab_size", 8192),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = UAVPerceptionAdapter(
        sam_dim=config["sam_dim"],
        dino_dim=config["dino_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_region_queries=config.get("num_region_queries", 8),
        num_heads=config.get("num_heads", 8),
        query_vocab_size=config.get("query_vocab_size", 8192),
        max_sam_tokens=config.get("max_sam_tokens", 64),
        max_dino_tokens=config.get("max_dino_tokens", 2048),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pred_path = Path(args.predictions).expanduser().resolve() if args.predictions else None
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    iou_sum = 0.0
    acc50 = 0.0
    acc75 = 0.0
    recall3 = 0.0
    oracle_iou_sum = 0.0
    oracle_acc50 = 0.0
    with torch.no_grad():
        pred_handle = pred_path.open("w", encoding="utf-8") if pred_path is not None else None
        try:
            for batch in loader:
                sam_tokens = batch["sam_tokens"].to(device)
                dino_tokens = batch["dino_tokens"].to(device)
                query_tokens = batch["query_tokens"].to(device)
                target = batch["bbox"].to(device)
                output = model(sam_tokens, dino_tokens, query_tokens=query_tokens)
                bbox = normalize_box_order(output["bbox"])
                candidates = normalize_box_order(output["candidate_bboxes"])
                topk_bboxes = normalize_box_order(output["topk_bboxes"])
                iou = box_iou_xyxy(bbox, target)
                candidate_iou = candidate_iou_xyxy(candidates, target)
                oracle_iou, oracle_idx = candidate_iou.max(dim=1)
                oracle_bbox = candidates[torch.arange(candidates.shape[0], device=device), oracle_idx]
                topk_count = min(3, candidate_iou.shape[1])
                topk_indices = torch.topk(output["candidate_scores"], k=topk_count, dim=1).indices
                topk_iou = candidate_iou.gather(dim=1, index=topk_indices)
                total += target.shape[0]
                iou_sum += float(iou.sum().cpu())
                acc50 += float((iou >= 0.5).sum().cpu())
                acc75 += float((iou >= 0.75).sum().cpu())
                recall3 += float((topk_iou >= 0.5).any(dim=1).sum().cpu())
                oracle_iou_sum += float(oracle_iou.sum().cpu())
                oracle_acc50 += float((oracle_iou >= 0.5).sum().cpu())
                if pred_handle is not None:
                    for sample_id, pred_bbox, top_bboxes, top_scores, best_oracle_bbox, score, item_iou, item_oracle_iou in zip(
                        batch["sample_id"],
                        bbox.cpu().tolist(),
                        topk_bboxes.cpu().tolist(),
                        output["topk_scores"].cpu().tolist(),
                        oracle_bbox.cpu().tolist(),
                        output["score"].cpu().tolist(),
                        iou.cpu().tolist(),
                        oracle_iou.cpu().tolist(),
                    ):
                        pred_handle.write(
                            json.dumps(
                                {
                                    "sample_id": sample_id,
                                    "bbox": pred_bbox,
                                    "topk_bboxes": top_bboxes,
                                    "topk_scores": top_scores,
                                    "oracle_bbox": best_oracle_bbox,
                                    "score": score,
                                    "iou": item_iou,
                                    "oracle_iou": item_oracle_iou,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        finally:
            if pred_handle is not None:
                pred_handle.close()

    print(
        json.dumps(
            {
                "samples": total,
                "mIoU": iou_sum / max(total, 1),
                "Acc@0.5": acc50 / max(total, 1),
                "Acc@0.75": acc75 / max(total, 1),
                "Recall@3": recall3 / max(total, 1),
                "Oracle_mIoU": oracle_iou_sum / max(total, 1),
                "Oracle_Acc@0.5": oracle_acc50 / max(total, 1),
                "predictions": str(pred_path) if pred_path is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
