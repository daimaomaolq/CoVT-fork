from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from uav_adapter.multitask_model import UAVMultiTaskAdapter
from uav_adapter.train_multitask_adapter import (
    TRAIN_COMPONENT_KEYS,
    caption_loss,
    distributed_sum,
    evaluate,
    forward_task,
    grounding_loss,
    is_rank0,
    make_loader,
    maybe_barrier,
    move_batch,
    subset_batch,
    subset_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UAVMultiTaskAdapter with manual distributed gradient all-reduce."
    )
    parser.add_argument("--train-index", action="append", required=True)
    parser.add_argument("--val-index", action="append", default=None)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--lm-query-dir", default=None)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--max-choices", type=int, default=8)
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
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--distributed-eval", action="store_true")
    parser.add_argument("--debug-max-batches", type=int, default=0)
    parser.add_argument("--trace-batches", action="store_true")
    return parser.parse_args()


def setup_manual_distributed() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Manual distributed training requires CUDA devices.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    return True, local_rank, device


def trace(enabled: bool, message: str, device: torch.device) -> None:
    if not enabled:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(
        json.dumps(
            {"trace": message, "rank": rank, "local_rank": local_rank, "device": str(device)},
            ensure_ascii=False,
        ),
        flush=True,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zero_touch_trainable_parameters(model: torch.nn.Module) -> torch.Tensor | None:
    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        term = param.sum() * 0.0
        total = term if total is None else total + term
    return total


def average_gradients(model: torch.nn.Module, distributed: bool) -> None:
    if not distributed:
        return
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def build_model(args: argparse.Namespace, first_batch: dict) -> tuple[UAVMultiTaskAdapter, dict]:
    lm_query_dim = first_batch["lm_query_hidden"].shape[-1] if "lm_query_hidden" in first_batch else 0
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
        lm_query_dim=lm_query_dim,
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
    )
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
        "lm_query_dim": lm_query_dim,
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
        "max_choices": args.max_choices,
        "caption_embedding_dim": args.caption_embedding_dim,
        "manual_grad_all_reduce": True,
    }
    return model, config


def main() -> None:
    args = parse_args()
    distributed, _local_rank, device = setup_manual_distributed()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
    maybe_barrier()

    train_loader, train_sampler = make_loader(args, args.train_index, shuffle=True, distributed=distributed)
    if train_loader is None:
        raise ValueError("At least one --train-index is required.")
    trace(args.trace_batches, "train_loader_ready", device)
    first_batch = next(iter(train_loader))
    trace(args.trace_batches, "first_batch_loaded_cpu", device)
    model, config = build_model(args, first_batch)
    model = model.to(device)
    trace(args.trace_batches, "model_on_device", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace(args.trace_batches, "optimizer_built", device)

    val_loader = None
    if is_rank0() and args.val_index and (not distributed or args.distributed_eval):
        val_loader, _ = make_loader(args, args.val_index, shuffle=False, distributed=False)

    if is_rank0():
        print(
            json.dumps(
                {"status": "using_manual_grad_all_reduce", "distributed": distributed, "world_size": dist.get_world_size() if distributed else 1},
                ensure_ascii=False,
            ),
            flush=True,
        )

    history: list[dict] = []
    best_score = float("-inf")

    def save_checkpoint(name: str) -> None:
        if not is_rank0():
            return
        torch.save(
            {"model": model.state_dict(), "config": config, "history": history, "args": vars(args)},
            output_dir / name,
        )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_items = 0.0
        component_sums: dict[str, float] = {}
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_raw_loaded", device)
            batch = move_batch(raw_batch, device)
            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_batch_on_device", device)
            loss = torch.zeros((), device=device)
            batch_items = batch["sam_tokens"].shape[0]
            ground_mask = batch["has_grounding"].bool()
            answer_mask = batch["has_answer"].bool()
            caption_mask = batch["has_caption"].bool()

            pred_all = forward_task(model, batch, task="multitask")
            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_forward_done", device)
            param_touch = zero_touch_trainable_parameters(model)
            if param_touch is not None:
                loss = loss + param_touch

            if bool(ground_mask.any()):
                sub = subset_batch(batch, ground_mask)
                pred = subset_output(pred_all, ground_mask)
                item_loss, comps = grounding_loss(pred, sub["bbox"], sub["scale_label"], args)
                loss = loss + args.grounding_loss_weight * item_loss
                for key, value in comps.items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(value.detach().cpu()) * sub["bbox"].shape[0]

            if bool(answer_mask.any()):
                sub = subset_batch(batch, answer_mask)
                pred = subset_output(pred_all, answer_mask)
                choice_rows = sub["choice_label"] >= 0
                hash_rows = ~choice_rows
                answer_losses = []
                if bool(choice_rows.any()) and "choice_logits" in pred:
                    item_loss = F.cross_entropy(pred["choice_logits"][choice_rows], sub["choice_label"][choice_rows])
                    answer_losses.append(item_loss)
                    component_sums["answer_choice_ce"] = component_sums.get("answer_choice_ce", 0.0) + float(item_loss.detach().cpu()) * sub["choice_label"][choice_rows].shape[0]
                if bool(hash_rows.any()):
                    item_loss = F.cross_entropy(pred["answer_logits"][hash_rows], sub["answer_id"][hash_rows])
                    answer_losses.append(item_loss)
                    component_sums["answer_hash_ce"] = component_sums.get("answer_hash_ce", 0.0) + float(item_loss.detach().cpu()) * sub["answer_id"][hash_rows].shape[0]
                if answer_losses:
                    loss = loss + args.answer_loss_weight * torch.stack(answer_losses).mean()

            if bool(caption_mask.any()):
                sub = subset_batch(batch, caption_mask)
                pred = subset_output(pred_all, caption_mask)
                item_loss = caption_loss(pred["caption_embedding"], sub["caption_target"], args.caption_temperature)
                loss = loss + args.caption_loss_weight * item_loss
                component_sums["caption_nce"] = component_sums.get("caption_nce", 0.0) + float(item_loss.detach().cpu()) * sub["caption_target"].shape[0]

            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_loss_done", device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_backward_done", device)
            average_gradients(model, distributed)
            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_grad_all_reduce_done", device)
            optimizer.step()
            trace(args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_step_done", device)

            total_loss += float(loss.detach().cpu()) * batch_items
            total_items += float(batch_items)
            if args.debug_max_batches > 0 and batch_index >= args.debug_max_batches:
                break

        if distributed:
            total_loss = distributed_sum(total_loss, device)
            total_items = distributed_sum(total_items, device)
            component_sums = {
                key: distributed_sum(component_sums.get(key, 0.0), device)
                for key in TRAIN_COMPONENT_KEYS
            }

        row = {"epoch": epoch, "train_loss": total_loss / max(total_items, 1.0)}
        row.update({f"train_{key}": value / max(total_items, 1.0) for key, value in component_sums.items()})
        if val_loader is not None and is_rank0():
            row.update({f"val_{key}": value for key, value in evaluate(model, val_loader, device, args).items()})
            score = row.get("val_ground_acc50", 0.0) + row.get("val_answer_acc", 0.0) + row.get("val_caption_cos", 0.0)
            if score > best_score:
                best_score = score
                save_checkpoint("best_multitask.pt")
                row["best_checkpoint"] = str(output_dir / "best_multitask.pt")
        if is_rank0():
            history.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        maybe_barrier()

    save_checkpoint("uav_multitask_adapter.pt")
    if is_rank0() and not (output_dir / "best_multitask.pt").exists():
        save_checkpoint("best_multitask.pt")
    if is_rank0():
        print(
            json.dumps(
                {
                    "status": "ok",
                    "checkpoint": str(output_dir / "uav_multitask_adapter.pt"),
                    "best_checkpoint": str(output_dir / "best_multitask.pt"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
