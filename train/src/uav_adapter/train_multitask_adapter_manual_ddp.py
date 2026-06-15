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
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume model/history from a previous multitask adapter checkpoint. --epochs is treated as the target total epoch.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights/config from a checkpoint, but start fresh optimizer/history at epoch 1.",
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=("nccl", "gloo"),
        help="Process group backend. Use gloo with --grad-sync-device cpu to avoid CUDA/NCCL gradient sync failures.",
    )
    parser.add_argument(
        "--grad-sync-device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Where gradients are all-reduced. cpu uses one flattened Gloo all-reduce then copies grads back to CUDA.",
    )
    return parser.parse_args()


ARCH_CONFIG_TO_ARG = {
    "hidden_dim": "hidden_dim",
    "dropout": "dropout",
    "num_region_queries": "num_region_queries",
    "num_heads": "num_heads",
    "query_vocab_size": "query_vocab_size",
    "query_max_len": "query_max_len",
    "max_lm_query_tokens": "max_lm_query_tokens",
    "query_encoder_type": "query_encoder_type",
    "query_layers": "query_layers",
    "category_vocab_size": "category_vocab_size",
    "region_vocab_size": "region_vocab_size",
    "rule_vocab_size": "rule_vocab_size",
    "max_sam_tokens": "max_sam_tokens",
    "max_dino_tokens": "max_dino_tokens",
    "anchor_delta_scale": "anchor_delta_scale",
    "answer_vocab_size": "answer_vocab_size",
    "max_choices": "max_choices",
    "caption_embedding_dim": "caption_embedding_dim",
}


def apply_resume_arch_config(args: argparse.Namespace, checkpoint: dict) -> None:
    config = checkpoint.get("config") or {}
    for config_key, arg_key in ARCH_CONFIG_TO_ARG.items():
        if config_key in config:
            setattr(args, arg_key, config[config_key])
    if "use_query_metadata" in config:
        args.disable_query_metadata = not bool(config["use_query_metadata"])


def history_best_score(history: list[dict]) -> float:
    best = float("-inf")
    for row in history:
        score = (
            float(row.get("val_ground_acc50", 0.0))
            + float(row.get("val_answer_acc", 0.0))
            + float(row.get("val_caption_cos", 0.0))
        )
        best = max(best, score)
    return best


def last_history_epoch(history: list[dict]) -> int:
    epochs = []
    for row in history:
        try:
            epochs.append(int(row.get("epoch", 0)))
        except (TypeError, ValueError):
            pass
    return max(epochs, default=0)


def setup_manual_distributed(backend: str) -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Manual distributed training requires CUDA devices.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if backend == "nccl":
        dist.init_process_group(backend=backend, device_id=device)
    else:
        dist.init_process_group(backend=backend)
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


def average_gradients(
    model: torch.nn.Module,
    distributed: bool,
    sync_device: str,
) -> None:
    if not distributed:
        return
    world_size = float(dist.get_world_size())
    if sync_device == "cpu":
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        if not grads:
            return
        flat = torch.cat([grad.detach().reshape(-1).cpu() for grad in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for grad in grads:
            length = grad.numel()
            grad.copy_(flat[offset : offset + length].view_as(grad).to(grad.device))
            offset += length
        return
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
    if args.grad_sync_device == "cpu" and args.dist_backend != "gloo":
        raise ValueError("--grad-sync-device cpu requires --dist-backend gloo.")
    distributed, _local_rank, device = setup_manual_distributed(args.dist_backend)
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)
    maybe_barrier()

    resume_checkpoint = None
    init_checkpoint = None
    if args.resume_checkpoint and args.init_checkpoint:
        raise ValueError("Use only one of --resume-checkpoint or --init-checkpoint.")
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser().resolve()
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        apply_resume_arch_config(args, resume_checkpoint)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).expanduser().resolve()
        init_checkpoint = torch.load(init_path, map_location="cpu")
        apply_resume_arch_config(args, init_checkpoint)

    train_loader, train_sampler = make_loader(args, args.train_index, shuffle=True, distributed=distributed)
    if train_loader is None:
        raise ValueError("At least one --train-index is required.")
    trace(args.trace_batches, "train_loader_ready", device)
    first_batch = next(iter(train_loader))
    trace(args.trace_batches, "first_batch_loaded_cpu", device)

    model, config = build_model(args, first_batch)
    model = model.to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        trace(args.trace_batches, "resume_model_loaded", device)
    if init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model"], strict=True)
        trace(args.trace_batches, "init_model_loaded", device)
    trace(args.trace_batches, "model_on_device", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_checkpoint is not None and resume_checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        trace(args.trace_batches, "resume_optimizer_loaded", device)
    trace(args.trace_batches, "optimizer_built", device)

    val_loader = None
    if is_rank0() and args.val_index and (not distributed or args.distributed_eval):
        val_loader, _ = make_loader(args, args.val_index, shuffle=False, distributed=False)

    if is_rank0():
        print(
            json.dumps(
                {
                    "status": "using_manual_grad_all_reduce",
                    "distributed": distributed,
                    "world_size": dist.get_world_size() if distributed else 1,
                    "dist_backend": args.dist_backend,
                    "grad_sync_device": args.grad_sync_device,
                    "resume_checkpoint": str(Path(args.resume_checkpoint).expanduser().resolve()) if args.resume_checkpoint else None,
                    "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if args.init_checkpoint else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    history: list[dict] = list((resume_checkpoint or {}).get("history") or [])
    best_score = history_best_score(history)
    start_epoch = last_history_epoch(history) + 1
    if is_rank0() and resume_checkpoint is not None:
        print(
            json.dumps(
                {
                    "status": "resumed",
                    "start_epoch": start_epoch,
                    "target_epoch": args.epochs,
                    "history_rows": len(history),
                    "best_score": best_score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if is_rank0() and init_checkpoint is not None:
        print(
            json.dumps(
                {
                    "status": "initialized",
                    "start_epoch": start_epoch,
                    "target_epoch": args.epochs,
                    "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def save_checkpoint(name: str) -> None:
        if not is_rank0():
            return
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "history": history,
                "args": vars(args),
            },
            output_dir / name,
        )

    if resume_checkpoint is not None and is_rank0() and not (output_dir / "best_multitask.pt").exists():
        save_checkpoint("best_multitask.pt")

    if start_epoch > args.epochs:
        if is_rank0():
            print(
                json.dumps(
                    {
                        "status": "nothing_to_train",
                        "start_epoch": start_epoch,
                        "target_epoch": args.epochs,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    for epoch in range(start_epoch, args.epochs + 1):
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
            average_gradients(model, distributed, args.grad_sync_device)
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
