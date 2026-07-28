#!/usr/bin/env python3
"""Replace implicit queries with model-generated explicit descriptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.index)
    sidecar_rows = read_jsonl(args.sidecar)
    sidecar = {str(row["sample_id"]): row for row in sidecar_rows}
    if len(sidecar) != len(sidecar_rows):
        raise ValueError("Duplicate sample_id in explicit sidecar")

    forbidden = {"question_e", "question_e_cn", "explicit_reference"}
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if any(field in source for field in forbidden):
            raise ValueError(f"Oracle field present in source row {index}")
        sample_id = str(source.get("sample_id", index))
        generated = sidecar.get(sample_id)
        if generated is None:
            raise KeyError(f"Missing generated explicit query for {sample_id}")
        if generated.get("question_e_used") is not False:
            raise ValueError(f"Invalid question_e protocol flag for {sample_id}")
        row = dict(source)
        row["implicit_query"] = row.get("query")
        row["query"] = generated["generated_explicit"]
        row["query_source"] = "model_generated_explicit"
        row["explicit_sidecar_sample_id"] = generated["sample_id"]
        row["question_e_used"] = False
        row["gt_visible_during_inference"] = False
        output.append(row)

    if len(output) != len(rows) or len(sidecar) != len(rows):
        raise ValueError(
            f"Cardinality mismatch: index={len(rows)} sidecar={len(sidecar)} output={len(output)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "dvgbench-generated-explicit-index-v1",
        "rows": len(output),
        "unique_sample_ids": len(
            {str(row.get("sample_id", i)) for i, row in enumerate(output)}
        ),
        "question_e_used": False,
        "gt_visible_during_inference": False,
        "query_source": "model_generated_explicit",
    }
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
