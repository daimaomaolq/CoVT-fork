#!/usr/bin/env python3
"""Build an oracle-free stage-2 grounding index from I2E stage-1 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_ORACLE_FIELDS = {"question_e", "question_e_cn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", required=True)
    parser.add_argument("--stage1-predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(Path(args.base_index).expanduser().resolve())
    stage1_rows = load_jsonl(Path(args.stage1_predictions).expanduser().resolve())

    stage1_by_id: dict[str, dict[str, Any]] = {}
    for row in stage1_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in stage1_by_id:
            raise ValueError(f"Missing or duplicate stage-1 sample_id: {sample_id!r}")
        protocol = row.get("protocol") or {}
        if protocol.get("question_e_used") is not False:
            raise ValueError(f"Stage-1 oracle-free flag missing for {sample_id}")
        if protocol.get("gt_visible_during_inference") is not False:
            raise ValueError(f"Stage-1 GT visibility flag missing for {sample_id}")
        stage1_by_id[sample_id] = row

    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in base_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate base sample_id: {sample_id!r}")
        seen.add(sample_id)
        forbidden = FORBIDDEN_ORACLE_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"Oracle fields {sorted(forbidden)} found in {sample_id}")
        stage1 = stage1_by_id.get(sample_id)
        if stage1 is None:
            raise ValueError(f"Missing stage-1 prediction for {sample_id}")
        explicit = str(stage1.get("explicit_prediction") or "").strip()
        if not explicit or stage1.get("explicit_parse_ok") is not True:
            raise ValueError(f"Invalid generated explicit description for {sample_id}")

        stage2 = dict(row)
        stage2["query_original"] = str(row.get("query") or "")
        stage2["query"] = explicit
        stage2["i2e_stage1"] = {
            "generated_explicit": explicit,
            "source_schema": stage1.get("schema_version"),
            "question_e_used": False,
            "gt_visible_during_inference": False,
        }
        output_rows.append(stage2)

    extra_ids = set(stage1_by_id).difference(seen)
    if extra_ids:
        raise ValueError(f"Stage-1 predictions contain {len(extra_ids)} unmatched sample_ids")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "schema_version": "dvgbench-qtsa-i2e-stage2-index-v1",
                "rows": len(output_rows),
                "unique_sample_ids": len(seen),
                "question_e_used": False,
                "gt_visible_during_inference": False,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
