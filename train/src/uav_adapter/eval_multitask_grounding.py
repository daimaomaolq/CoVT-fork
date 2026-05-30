from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader

from uav_adapter.dataset import TokenGroundingDataset
from uav_adapter.multitask_model import UAVMultiTaskAdapter
from uav_adapter.train_adapter import box_iou_xyxy, collate, normalize_box_order, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UAVMultiTaskAdapter grounding head.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--lm-query-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bbox-key", default="bbox_norm")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predictions", default=None)
    return parser.parse_args()


def read_meta(index_path: str | Path) -> dict[str, dict]:
    meta = {}
    with Path(index_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta[str(row["sample_id"])] = row
    return meta


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    dataset = TokenGroundingDataset(
        args.index,
        args.token_dir,
        lm_query_dir=args.lm_query_dir,
        bbox_key=args.bbox_key,
        query_max_len=config.get("query_max_len", 64),
        query_vocab_size=config.get("query_vocab_size", 8192),
        max_lm_query_tokens=config.get("max_lm_query_tokens", 64),
        region_vocab_size=config.get("region_vocab_size", 64),
        rule_vocab_size=config.get("rule_vocab_size", 256),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = UAVMultiTaskAdapter(
        sam_dim=config["sam_dim"],
        dino_dim=config["dino_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_region_queries=config.get("num_region_queries", 64),
        num_heads=config.get("num_heads", 8),
        query_vocab_size=config.get("query_vocab_size", 8192),
        query_encoder_type=config.get("query_encoder_type", "transformer"),
        query_layers=config.get("query_layers", 2),
        max_query_tokens=config.get("query_max_len", 64),
        lm_query_dim=config.get("lm_query_dim", 0),
        max_lm_query_tokens=config.get("max_lm_query_tokens", 64),
        category_vocab_size=config.get("category_vocab_size", 32),
        region_vocab_size=config.get("region_vocab_size", 64),
        rule_vocab_size=config.get("rule_vocab_size", 256),
        use_query_metadata=config.get("use_query_metadata", True),
        use_output_query_proj=config.get("use_output_query_proj", True),
        max_sam_tokens=config.get("max_sam_tokens", 64),
        max_dino_tokens=config.get("max_dino_tokens", 2048),
        anchor_delta_scale=config.get("anchor_delta_scale", 1.0),
        answer_vocab_size=config.get("answer_vocab_size", 4096),
        caption_embedding_dim=config.get("caption_embedding_dim", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    meta = read_meta(args.index)
    pred_path = Path(args.predictions).expanduser().resolve() if args.predictions else None
    pred_handle = None
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_handle = pred_path.open("w", encoding="utf-8")

    total = 0
    iou_sum = 0.0
    acc50 = 0.0
    per_class_total = defaultdict(int)
    per_class_acc = defaultdict(float)
    with torch.no_grad():
        try:
            for batch in loader:
                sam_tokens = batch["sam_tokens"].to(device)
                dino_tokens = batch["dino_tokens"].to(device)
                query_tokens = batch["query_tokens"].to(device)
                lm_query_hidden = batch.get("lm_query_hidden")
                lm_query_mask = batch.get("lm_query_mask")
                if lm_query_hidden is not None:
                    lm_query_hidden = lm_query_hidden.to(device)
                    lm_query_mask = lm_query_mask.to(device)
                target = batch["bbox"].to(device)
                output = model(
                    sam_tokens,
                    dino_tokens,
                    query_tokens=query_tokens,
                    lm_query_hidden=lm_query_hidden,
                    lm_query_mask=lm_query_mask,
                    category_ids=batch["category_id"].to(device),
                    scale_labels=batch["scale_label"].to(device),
                    region_ids=batch["region_id"].to(device),
                    rule_ids=batch["query_rule_id"].to(device),
                    task="grounding",
                )
                bbox = normalize_box_order(output["bbox"])
                iou = box_iou_xyxy(bbox, target)
                total += iou.shape[0]
                iou_sum += float(iou.sum().cpu())
                acc50 += float((iou >= 0.5).sum().cpu())
                for sample_id, pred_bbox, score, item_iou in zip(
                    batch["sample_id"],
                    bbox.cpu().tolist(),
                    output["score"].cpu().tolist(),
                    iou.cpu().tolist(),
                ):
                    row_meta = meta.get(str(sample_id), {})
                    cls = str(row_meta.get("category") or row_meta.get("class") or "unknown")
                    per_class_total[cls] += 1
                    per_class_acc[cls] += 1.0 if item_iou >= 0.5 else 0.0
                    if pred_handle is not None:
                        pred_handle.write(
                            json.dumps(
                                {
                                    "sample_id": sample_id,
                                    "bbox": pred_bbox,
                                    "score": score,
                                    "iou": item_iou,
                                    "class": cls,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        finally:
            if pred_handle is not None:
                pred_handle.close()

    class_acc = {
        key: per_class_acc[key] / max(per_class_total[key], 1)
        for key in sorted(per_class_total)
    }
    result = {
        "samples": total,
        "mIoU": iou_sum / max(total, 1),
        "Acc@0.5": acc50 / max(total, 1),
        "DVGBench_AVG": sum(class_acc.values()) / max(len(class_acc), 1),
        "class_Acc@0.5": class_acc,
        "class_counts": dict(sorted(per_class_total.items())),
        "predictions": str(pred_path) if pred_path is not None else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
