from __future__ import annotations

import json
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from uav_adapter.dataset import _lm_query_tensor, _token_tensor


GROUNDING_TYPES = {"grounding", "detection", "region_answer", "region_caption"}
ANSWER_TYPES = {"image_answer", "region_answer"}
CAPTION_TYPES = {"caption", "region_caption"}


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("\\", "_").replace("/", "_").replace(":", "_")


def word_ids(text: str, max_len: int, vocab_size: int) -> torch.Tensor:
    words = re.findall(r"[a-z0-9_\u4e00-\u9fff]+", str(text).lower())
    ids = torch.zeros(max_len, dtype=torch.long)
    for index, word in enumerate(words[:max_len]):
        digest = blake2b(word.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, byteorder="little")
        ids[index] = value % (vocab_size - 1) + 1
    return ids


def field_id(text: str, vocab_size: int) -> torch.Tensor:
    if not text or vocab_size <= 1:
        return torch.tensor(0, dtype=torch.long)
    digest = blake2b(text.lower().encode("utf-8"), digest_size=4).digest()
    value = int.from_bytes(digest, byteorder="little")
    return torch.tensor(value % (vocab_size - 1) + 1, dtype=torch.long)


def answer_id(text: str, vocab_size: int) -> torch.Tensor:
    return field_id(str(text).strip().lower(), vocab_size)


def scale_label(bbox: list[float]) -> int:
    width = max(float(bbox[2]) - float(bbox[0]), 0.0)
    height = max(float(bbox[3]) - float(bbox[1]), 0.0)
    area = width * height
    if area < 0.01:
        return 0
    if area < 0.08:
        return 1
    return 2


def text_embedding(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    words = re.findall(r"[a-z0-9_\u4e00-\u9fff]+", str(text).lower())
    if not words:
        vec[0] = 1.0
        return vec
    for word in words:
        digest = blake2b(word.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], byteorder="little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return F.normalize(vec, dim=0)


class UAVITMultiTaskTokenDataset(Dataset):
    def __init__(
        self,
        index_paths: list[str | Path],
        token_dir: str | Path,
        lm_query_dir: str | Path | None = None,
        query_max_len: int = 64,
        query_vocab_size: int = 8192,
        answer_vocab_size: int = 4096,
        caption_embedding_dim: int = 256,
        max_lm_query_tokens: int = 64,
        region_vocab_size: int = 64,
        rule_vocab_size: int = 256,
    ) -> None:
        self.index_paths = [Path(path).expanduser().resolve() for path in index_paths]
        self.token_dir = Path(token_dir).expanduser().resolve()
        self.lm_query_dir = Path(lm_query_dir).expanduser().resolve() if lm_query_dir else None
        self.query_max_len = query_max_len
        self.query_vocab_size = query_vocab_size
        self.answer_vocab_size = answer_vocab_size
        self.caption_embedding_dim = caption_embedding_dim
        self.max_lm_query_tokens = max_lm_query_tokens
        self.region_vocab_size = region_vocab_size
        self.rule_vocab_size = rule_vocab_size
        self.records = self._read_records()

    def _read_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.index_paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "sample_id" not in row:
                        raise ValueError(f"{path}:{line_no} is missing sample_id")
                    if row.get("task_type") == "detection" and row.get("bboxes_norm"):
                        for box_idx, bbox in enumerate(row["bboxes_norm"]):
                            item = dict(row)
                            item["token_sample_id"] = row["sample_id"]
                            item["sample_id"] = f"{row['sample_id']}_box{box_idx:03d}"
                            item["bbox_norm"] = bbox
                            item["task_type"] = "grounding"
                            item["task_tag"] = "det_box"
                            records.append(item)
                    else:
                        row["token_sample_id"] = row["sample_id"]
                        records.append(row)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        token_sample_id = str(row.get("token_sample_id") or row["sample_id"])
        token_path = self.token_dir / f"{safe_sample_id(token_sample_id)}.pt"
        cache = torch.load(token_path, map_location="cpu")
        task_type = str(row.get("task_type", "unknown"))
        query = row.get("query") or row.get("human") or ""
        bbox = row.get("bbox_norm") or row.get("region_norm") or [0.0, 0.0, 1.0, 1.0]
        bbox = [float(value) for value in bbox]
        answer = row.get("answer") or ""
        caption = row.get("caption") or answer
        item = {
            "sample_id": row["sample_id"],
            "token_sample_id": token_sample_id,
            "task_type": task_type,
            "task_tag": str(row.get("task_tag", "")),
            "sam_tokens": _token_tensor(cache, "sam", "attended"),
            "dino_tokens": _token_tensor(cache, "dino", "attended"),
            "query_tokens": word_ids(str(query), self.query_max_len, self.query_vocab_size),
            "category_id": torch.tensor(0, dtype=torch.long),
            "region_id": field_id(str(row.get("region", "")), self.region_vocab_size),
            "query_rule_id": field_id(str(row.get("task_tag", "")), self.rule_vocab_size),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "scale_label": torch.tensor(scale_label(bbox), dtype=torch.long),
            "answer_id": answer_id(str(answer), self.answer_vocab_size),
            "caption_target": text_embedding(str(caption), self.caption_embedding_dim),
            "has_grounding": torch.tensor(task_type in GROUNDING_TYPES, dtype=torch.bool),
            "has_answer": torch.tensor(task_type in ANSWER_TYPES, dtype=torch.bool),
            "has_caption": torch.tensor(task_type in CAPTION_TYPES, dtype=torch.bool),
            "query": str(query),
            "answer": str(answer),
            "caption": str(caption),
            "image": row.get("image", ""),
        }
        if self.lm_query_dir is not None:
            lm_path = self.lm_query_dir / f"{safe_sample_id(token_sample_id)}.pt"
            lm_cache = torch.load(lm_path, map_location="cpu")
            lm_hidden, lm_mask = _lm_query_tensor(lm_cache)
            item["lm_query_hidden"] = lm_hidden[: self.max_lm_query_tokens]
            item["lm_query_mask"] = lm_mask[: self.max_lm_query_tokens]
        return item


def multitask_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "sample_id": [item["sample_id"] for item in batch],
        "token_sample_id": [item["token_sample_id"] for item in batch],
        "task_type": [item["task_type"] for item in batch],
        "task_tag": [item["task_tag"] for item in batch],
        "query": [item["query"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "image": [item["image"] for item in batch],
        "sam_tokens": torch.stack([item["sam_tokens"] for item in batch]),
        "dino_tokens": torch.stack([item["dino_tokens"] for item in batch]),
        "query_tokens": torch.stack([item["query_tokens"] for item in batch]),
        "category_id": torch.stack([item["category_id"] for item in batch]),
        "region_id": torch.stack([item["region_id"] for item in batch]),
        "query_rule_id": torch.stack([item["query_rule_id"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
        "scale_label": torch.stack([item["scale_label"] for item in batch]),
        "answer_id": torch.stack([item["answer_id"] for item in batch]),
        "caption_target": torch.stack([item["caption_target"] for item in batch]),
        "has_grounding": torch.stack([item["has_grounding"] for item in batch]),
        "has_answer": torch.stack([item["has_answer"] for item in batch]),
        "has_caption": torch.stack([item["has_caption"] for item in batch]),
    }
    if "lm_query_hidden" in batch[0]:
        max_len = max(item["lm_query_hidden"].shape[0] for item in batch)
        hidden_dim = batch[0]["lm_query_hidden"].shape[-1]
        hidden = batch[0]["lm_query_hidden"].new_zeros(len(batch), max_len, hidden_dim)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for row_idx, item in enumerate(batch):
            token_count = item["lm_query_hidden"].shape[0]
            hidden[row_idx, :token_count] = item["lm_query_hidden"]
            mask[row_idx, :token_count] = item["lm_query_mask"]
        output["lm_query_hidden"] = hidden
        output["lm_query_mask"] = mask
    return output
