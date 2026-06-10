from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader

from uav_adapter.multitask_dataset import UAVITMultiTaskTokenDataset, answer_id, multitask_collate, normalize_answer_text
from uav_adapter.multitask_model import UAVMultiTaskAdapter
from uav_adapter.train_adapter import resolve_device
from uav_adapter.train_multitask_adapter import forward_task, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UAVMultiTaskAdapter answer head with candidate scoring.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--lm-query-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--candidate-mode",
        choices=("auto", "count", "answers"),
        default="auto",
        help="count uses integer candidates; answers uses unique answers from index.",
    )
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--predictions", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalized_answer(row: dict) -> str:
    return normalize_answer_text(str(row.get("answer", "")))


def infer_candidate_mode(rows: list[dict], explicit: str) -> str:
    if explicit != "auto":
        return explicit
    tags = {str(row.get("task_tag", "")) for row in rows}
    if any("count" in tag for tag in tags):
        return "count"
    return "answers"


def build_candidates(rows: list[dict], mode: str, max_count: int | None) -> list[str]:
    if mode == "count":
        answers = []
        for row in rows:
            text = normalized_answer(row)
            if re.fullmatch(r"\d+", text):
                answers.append(int(text))
        upper = max_count if max_count is not None else max(answers, default=20)
        return [str(value) for value in range(max(upper, 0) + 1)]
    candidates = sorted({normalized_answer(row) for row in rows if normalized_answer(row)})
    if not candidates:
        raise ValueError("No answer candidates found in index.")
    return candidates


def build_model(config: dict) -> UAVMultiTaskAdapter:
    return UAVMultiTaskAdapter(
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
    )


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.index)
    mode = infer_candidate_mode(rows, args.candidate_mode)
    candidates = build_candidates(rows, mode, args.max_count)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    answer_vocab_size = int(config.get("answer_vocab_size", 4096))
    candidate_ids = torch.tensor(
        [int(answer_id(candidate, answer_vocab_size)) for candidate in candidates],
        dtype=torch.long,
    )
    id_to_candidates: dict[int, list[str]] = defaultdict(list)
    for candidate, candidate_id in zip(candidates, candidate_ids.tolist()):
        id_to_candidates[candidate_id].append(candidate)
    collisions = {key: value for key, value in id_to_candidates.items() if len(value) > 1}

    dataset = UAVITMultiTaskTokenDataset(
        [args.index],
        args.token_dir,
        lm_query_dir=args.lm_query_dir,
        query_max_len=config.get("query_max_len", 64),
        query_vocab_size=config.get("query_vocab_size", 8192),
        answer_vocab_size=answer_vocab_size,
        max_choices=config.get("max_choices", 8),
        caption_embedding_dim=config.get("caption_embedding_dim", 256),
        max_lm_query_tokens=config.get("max_lm_query_tokens", 64),
        region_vocab_size=config.get("region_vocab_size", 64),
        rule_vocab_size=config.get("rule_vocab_size", 256),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multitask_collate)
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    candidate_ids = candidate_ids.to(device)

    pred_path = Path(args.predictions).expanduser().resolve() if args.predictions else None
    pred_handle = None
    if pred_path is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_handle = pred_path.open("w", encoding="utf-8")

    total = 0
    correct = 0
    per_answer_total: Counter[str] = Counter()
    per_answer_correct: Counter[str] = Counter()
    try:
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                output = forward_task(model, batch, task="answer")
                logits = output["answer_logits"].index_select(dim=1, index=candidate_ids)
                best = logits.argmax(dim=1).cpu().tolist()
                for row_idx, candidate_idx in enumerate(best):
                    target = normalize_answer_text(raw_batch["answer"][row_idx])
                    prediction = candidates[candidate_idx]
                    is_correct = prediction == target
                    total += 1
                    correct += int(is_correct)
                    per_answer_total[target] += 1
                    per_answer_correct[target] += int(is_correct)
                    if pred_handle is not None:
                        pred_handle.write(
                            json.dumps(
                                {
                                    "sample_id": raw_batch["sample_id"][row_idx],
                                    "prediction": prediction,
                                    "answer": target,
                                    "correct": is_correct,
                                    "task_tag": raw_batch["task_tag"][row_idx],
                                    "query": raw_batch["query"][row_idx],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    finally:
        if pred_handle is not None:
            pred_handle.close()

    result = {
        "samples": total,
        "accuracy": correct / max(total, 1),
        "candidate_mode": mode,
        "candidate_count": len(candidates),
        "hash_collision_ids": len(collisions),
        "class_accuracy": {
            key: per_answer_correct[key] / max(per_answer_total[key], 1)
            for key in sorted(per_answer_total)
        },
        "class_counts": dict(sorted(per_answer_total.items())),
        "predictions": str(pred_path) if pred_path is not None else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
