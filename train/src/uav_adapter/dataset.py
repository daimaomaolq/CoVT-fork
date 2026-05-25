from __future__ import annotations

import json
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _first_tensor(value: Any):
    if hasattr(value, "shape") and hasattr(value, "detach"):
        return value
    if isinstance(value, list):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _token_tensor(cache: dict[str, Any], anchor_name: str, preferred_key: str) -> torch.Tensor:
    anchor = cache["tokens"][anchor_name]
    tensor = anchor.get(preferred_key)
    if tensor is None:
        tensor = _first_tensor(anchor)
    if tensor is None:
        raise ValueError(f"No tensor found for {anchor_name}.{preferred_key}")
    return tensor.float().squeeze(0)


def _lm_query_tensor(cache: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "lm_query" in cache:
        query_cache = cache["lm_query"]
    elif "query_hidden" in cache:
        query_cache = cache["query_hidden"]
    else:
        query_cache = cache

    hidden = query_cache.get("hidden_states")
    if hidden is None:
        hidden = query_cache.get("hidden")
    if hidden is None:
        raise ValueError("No language query hidden_states tensor found in cache.")
    hidden = hidden.float()
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    if hidden.ndim != 2:
        raise ValueError(f"Expected language query hidden states with shape [tokens, dim], got {list(hidden.shape)}")

    mask = query_cache.get("attention_mask")
    if mask is None:
        mask = torch.ones(hidden.shape[0], dtype=torch.bool)
    else:
        mask = mask.bool().view(-1)
    if mask.shape[0] != hidden.shape[0]:
        raise ValueError(
            "Language query attention_mask length does not match hidden_states: "
            f"{mask.shape[0]} vs {hidden.shape[0]}"
        )
    return hidden, mask


class TokenGroundingDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        token_dir: str | Path,
        lm_query_dir: str | Path | None = None,
        bbox_key: str = "bbox_norm",
        query_max_len: int = 32,
        query_vocab_size: int = 8192,
        max_lm_query_tokens: int = 64,
        region_vocab_size: int = 64,
        rule_vocab_size: int = 256,
    ) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.token_dir = Path(token_dir).expanduser().resolve()
        self.lm_query_dir = Path(lm_query_dir).expanduser().resolve() if lm_query_dir else None
        self.bbox_key = bbox_key
        self.query_max_len = query_max_len
        self.query_vocab_size = query_vocab_size
        self.max_lm_query_tokens = max_lm_query_tokens
        self.region_vocab_size = region_vocab_size
        self.rule_vocab_size = rule_vocab_size
        self.records = self._read_index()

    def _read_index(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "sample_id" not in row:
                    raise ValueError(f"{self.index_path}:{line_no} is missing sample_id")
                if self.bbox_key not in row:
                    raise ValueError(f"{self.index_path}:{line_no} is missing {self.bbox_key}")
                records.append(row)
        return records

    @staticmethod
    def safe_sample_id(sample_id: str) -> str:
        return sample_id.replace("\\", "_").replace("/", "_").replace(":", "_")

    def __len__(self) -> int:
        return len(self.records)

    def _query_token_ids(self, text: str) -> torch.Tensor:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        ids = torch.zeros(self.query_max_len, dtype=torch.long)
        for index, word in enumerate(words[: self.query_max_len]):
            digest = blake2b(word.encode("utf-8"), digest_size=4).digest()
            value = int.from_bytes(digest, byteorder="little")
            ids[index] = value % (self.query_vocab_size - 1) + 1
        return ids

    @staticmethod
    def _field_id(text: str, vocab_size: int) -> torch.Tensor:
        if not text or vocab_size <= 1:
            return torch.tensor(0, dtype=torch.long)
        digest = blake2b(text.lower().encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, byteorder="little")
        return torch.tensor(value % (vocab_size - 1) + 1, dtype=torch.long)

    @staticmethod
    def _scale_label(bbox: list[float]) -> int:
        width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        height = max(float(bbox[3]) - float(bbox[1]), 0.0)
        area = width * height
        if area < 0.01:
            return 0
        if area < 0.08:
            return 1
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        token_path = self.token_dir / f"{self.safe_sample_id(str(row['sample_id']))}.pt"
        cache = torch.load(token_path, map_location="cpu")
        bbox = [float(value) for value in row[self.bbox_key]]
        query = row.get("query", "")
        category_id = int(row.get("category_id", 0))
        item = {
            "sample_id": row["sample_id"],
            "sam_tokens": _token_tensor(cache, "sam", "attended"),
            "dino_tokens": _token_tensor(cache, "dino", "attended"),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "query_tokens": self._query_token_ids(query),
            "category_id": torch.tensor(max(category_id, 0), dtype=torch.long),
            "region_id": self._field_id(str(row.get("region", "")), self.region_vocab_size),
            "query_rule_id": self._field_id(str(row.get("query_rule", "")), self.rule_vocab_size),
            "scale_label": torch.tensor(self._scale_label(bbox), dtype=torch.long),
            "query": query,
            "image": row.get("image", ""),
        }
        if self.lm_query_dir is not None:
            lm_path = self.lm_query_dir / f"{self.safe_sample_id(str(row['sample_id']))}.pt"
            lm_cache = torch.load(lm_path, map_location="cpu")
            lm_hidden, lm_mask = _lm_query_tensor(lm_cache)
            lm_hidden = lm_hidden[: self.max_lm_query_tokens]
            lm_mask = lm_mask[: self.max_lm_query_tokens]
            if lm_hidden.shape[0] == 0:
                raise ValueError(f"{lm_path} contains no language query tokens.")
            item["lm_query_hidden"] = lm_hidden
            item["lm_query_mask"] = lm_mask
        return item
