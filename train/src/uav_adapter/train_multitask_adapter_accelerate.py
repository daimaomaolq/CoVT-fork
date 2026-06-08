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
import torch.nn.functional as F

try:
    from accelerate import Accelerator, DistributedDataParallelKwargs
except ImportError as exc:  # pragma: no cover - exercised on machines without training deps.
    raise SystemExit(
        "Missing dependency: accelerate. Install train/requirements.txt or run "
        "`pip install accelerate` in the CoVT training environment."
    ) from exc

from uav_adapter.multitask_model import UAVMultiTaskAdapter
from uav_adapter.train_multitask_adapter import (
    TRAIN_COMPONENT_KEYS,
    caption_loss,
    evaluate,
    forward_task,
    grounding_loss,
    make_loader,
    move_batch,
    subset_batch,
    subset_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UAVMultiTaskAdapter with HuggingFace Accelerate."
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
    parser.add_argument(
        "--distributed-eval",
        action="store_true",
        help="Run validation on the main process during multi-GPU training.",
    )
    parser.add_argument(
        "--debug-max-batches",
        type=int,
        default=0,
        help="Stop each epoch after this many train batches. 0 means full epoch.",
    )
    parser.add_argument(
        "--trace-batches",
        action="store_true",
        help="Synchronize and print per-rank progress around load/forward/loss/backward/step.",
    )
    return parser.parse_args()


def configure_local_cuda_device() -> None:
    """Bind each launched process before Accelerate initializes NCCL."""

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return
    torch.cuda.set_device(int(local_rank))


def trace_rank(accelerator: Accelerator, enabled: bool, message: str) -> None:
    if not enabled:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    print(
        json.dumps(
            {
                "trace": message,
                "rank": accelerator.process_index,
                "local_rank": accelerator.local_process_index,
                "device": str(accelerator.device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def zero_touch_trainable_parameters(model: torch.nn.Module) -> torch.Tensor | None:
    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        term = param.sum() * 0.0
        total = term if total is None else total + term
    return total


def reduce_sum(accelerator: Accelerator, value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    tensor = accelerator.reduce(tensor, reduction="sum")
    return float(tensor.detach().cpu())


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
    }
    return model, config


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: Path,
    name: str,
    config: dict,
    history: list[dict],
    args: argparse.Namespace,
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(
        {
            "model": unwrapped.state_dict(),
            "config": config,
            "history": history,
            "args": vars(args),
        },
        output_dir / name,
    )


def main() -> None:
    args = parse_args()
    configure_local_cuda_device()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    trace_rank(accelerator, args.trace_batches, "accelerator_ready")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, _ = make_loader(args, args.train_index, shuffle=True, distributed=False)
    if train_loader is None:
        raise ValueError("At least one --train-index is required.")
    trace_rank(accelerator, args.trace_batches, "train_loader_ready")
    first_batch = next(iter(train_loader))
    trace_rank(accelerator, args.trace_batches, "first_batch_loaded_cpu")
    model, config = build_model(args, first_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace_rank(accelerator, args.trace_batches, "model_optimizer_built")
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    trace_rank(accelerator, args.trace_batches, "accelerator_prepare_done")

    val_loader = None
    should_eval = bool(args.val_index) and (
        accelerator.num_processes == 1 or args.distributed_eval
    )
    if accelerator.is_main_process and should_eval:
        val_loader, _ = make_loader(args, args.val_index, shuffle=False, distributed=False)

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "status": "using_accelerate",
                    "num_processes": accelerator.num_processes,
                    "device": str(accelerator.device),
                    "distributed_eval": bool(args.distributed_eval),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    history: list[dict] = []
    best_score = float("-inf")
    best_saved = False
    device = accelerator.device

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0.0
        component_sums: dict[str, float] = {}
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            trace_rank(accelerator, args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_raw_loaded")
            batch = move_batch(raw_batch, device)
            trace_rank(accelerator, args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_batch_on_device")
            loss = torch.zeros((), device=device)
            batch_items = batch["sam_tokens"].shape[0]
            ground_mask = batch["has_grounding"].bool()
            answer_mask = batch["has_answer"].bool()
            caption_mask = batch["has_caption"].bool()

            pred_all = forward_task(model, batch, task="multitask")
            trace_rank(accelerator, args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_forward_done")
            param_touch = zero_touch_trainable_parameters(accelerator.unwrap_model(model))
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

            trace_rank(accelerator, args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_loss_done")
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            trace_rank(accelerator, args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_backward_done")
            optimizer.step()
            trace_rank(accelerator, args.trace_batches, f"epoch_{epoch}_batch_{batch_index}_step_done")

            total_loss += float(loss.detach().cpu()) * batch_items
            total_items += float(batch_items)
            if args.debug_max_batches > 0 and batch_index >= args.debug_max_batches:
                break

        total_loss = reduce_sum(accelerator, total_loss, device)
        total_items = reduce_sum(accelerator, total_items, device)
        component_sums = {
            key: reduce_sum(accelerator, component_sums.get(key, 0.0), device)
            for key in TRAIN_COMPONENT_KEYS
        }

        row = {"epoch": epoch, "train_loss": total_loss / max(total_items, 1.0)}
        row.update({f"train_{key}": value / max(total_items, 1.0) for key, value in component_sums.items()})
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and val_loader is not None:
            metrics = evaluate(accelerator.unwrap_model(model), val_loader, device, args)
            row.update({f"val_{key}": value for key, value in metrics.items()})
            score = row.get("val_ground_acc50", 0.0) + row.get("val_answer_acc", 0.0) + row.get("val_caption_cos", 0.0)
            if score > best_score:
                best_score = score
                save_checkpoint(accelerator, model, output_dir, "best_multitask.pt", config, history, args)
                best_saved = True
                row["best_checkpoint"] = str(output_dir / "best_multitask.pt")

        if accelerator.is_main_process:
            history.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        accelerator.wait_for_everyone()

    save_checkpoint(accelerator, model, output_dir, "uav_multitask_adapter.pt", config, history, args)
    if not best_saved:
        save_checkpoint(accelerator, model, output_dir, "best_multitask.pt", config, history, args)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
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


if __name__ == "__main__":
    main()
