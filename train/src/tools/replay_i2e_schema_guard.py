from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_dvgbench_generative_grounding import (
    guard_i2e_schema,
    parse_bbox_text,
    parse_explicit_text,
    is_i2e_schema_valid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the deterministic I2E schema guard over prediction JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    raw_explicit_ok = 0
    guarded_explicit_ok = 0
    guard_applied = 0
    raw_schema_ok = 0
    guarded_schema_ok = 0
    bbox_changed = 0
    replayed = []
    for row in rows:
        raw_output = str(row.get("raw_output") or "")
        raw_bbox = parse_bbox_text(raw_output)
        guarded_output, applied, reason = guard_i2e_schema(raw_output)
        guarded_bbox = parse_bbox_text(guarded_output)
        changed = raw_bbox != guarded_bbox
        bbox_changed += int(changed)
        if changed:
            raise RuntimeError(
                f"Schema guard changed bbox for {row.get('sample_id')}: "
                f"raw={raw_bbox}, guarded={guarded_bbox}"
            )
        raw_explicit = parse_explicit_text(raw_output)
        guarded_explicit = parse_explicit_text(guarded_output)
        raw_explicit_ok += int(raw_explicit is not None)
        guarded_explicit_ok += int(guarded_explicit is not None)
        guard_applied += int(applied)
        raw_schema_ok += int(is_i2e_schema_valid(raw_output))
        guarded_schema_ok += int(is_i2e_schema_valid(guarded_output))
        replayed.append(
            {
                **row,
                "raw_explicit_prediction": raw_explicit,
                "explicit_prediction": guarded_explicit,
                "guarded_output": guarded_output,
                "schema_guard_applied": applied,
                "schema_guard_reason": reason,
                "bbox": guarded_bbox,
                "parse_ok": guarded_bbox is not None,
                "raw_explicit_parse_ok": raw_explicit is not None,
                "explicit_parse_ok": guarded_explicit is not None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in replayed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(rows)
    report = {
        "schema_version": "dvgbench-i2e-schema-guard-replay-v1",
        "samples": total,
        "unique_sample_ids": len({str(row.get("sample_id")) for row in rows}),
        "raw_explicit_format_rate": raw_explicit_ok / max(total, 1),
        "guarded_explicit_format_rate": guarded_explicit_ok / max(total, 1),
        "schema_guard_applied": guard_applied,
        "raw_schema_format_rate": raw_schema_ok / max(total, 1),
        "guarded_schema_format_rate": guarded_schema_ok / max(total, 1),
        "bbox_changed": bbox_changed,
        "bbox_invariant": bbox_changed == 0,
        "input": str(input_path),
        "output": str(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()