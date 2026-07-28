#!/usr/bin/env python3
"""Evaluate the disjoint I2E gate against an untouched QTSA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(summary: dict[str, Any], i2e: bool = False) -> dict[str, Any]:
    keys = ["samples", "mIoU", "Acc@0.5", "DVGBench_AVG", "parse_failed"]
    if i2e:
        keys.extend(
            [
                "explicit_parse_failed",
                "raw_explicit_parse_failed",
                "raw_schema_parse_failed",
                "schema_parse_failed",
                "raw_schema_format_rate",
                "schema_format_rate",
                "schema_guard_applied",
            ]
        )
    return {key: summary.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2e-summary", type=Path, required=True)
    parser.add_argument("--trained-direct-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-regression", type=float, default=0.05)
    args = parser.parse_args()

    i2e = load(args.i2e_summary)
    trained_direct = load(args.trained_direct_summary)
    baseline = load(args.baseline_summary)
    split = load(args.split_manifest)
    tolerance = args.max_regression
    expected = int(split["validation_rows"])

    checks = {
        "disjoint_split": split.get("overlap") == 0,
        "sample_count_matches": all(
            int(item["samples"]) == expected for item in (i2e, trained_direct, baseline)
        ),
        "oracle_free_i2e": not i2e["config"].get("question_e_used", True)
        and not i2e["config"].get("gt_visible_during_inference", True),
        "i2e_bbox_parse_100": i2e["parse_failed"] == 0,
        "i2e_explicit_parse_100": i2e["explicit_parse_failed"] == 0,
        "raw_schema_format_ge_0_90": i2e["raw_schema_format_rate"] >= 0.90,
        "guarded_schema_parse_100": i2e["schema_parse_failed"] == 0,
        "trained_direct_parse_100": trained_direct["parse_failed"] == 0,
        "baseline_parse_100": baseline["parse_failed"] == 0,
        "trained_direct_acc_preserved": trained_direct["Acc@0.5"]
        >= baseline["Acc@0.5"] - tolerance,
        "trained_direct_miou_preserved": trained_direct["mIoU"]
        >= baseline["mIoU"] - tolerance,
        "i2e_acc_not_regressed": i2e["Acc@0.5"]
        >= baseline["Acc@0.5"] - tolerance,
        "i2e_miou_not_regressed": i2e["mIoU"] >= baseline["mIoU"] - tolerance,
        "i2e_has_positive_signal": i2e["Acc@0.5"] >= baseline["Acc@0.5"]
        or i2e["mIoU"] >= baseline["mIoU"],
    }
    result = {
        "schema_version": "dvgbench-qtsa-i2e-heldout-gate-v2",
        "split": split,
        "i2e": compact(i2e, i2e=True),
        "trained_direct": compact(trained_direct),
        "untouched_qtsa_baseline": compact(baseline),
        "delta_i2e_vs_baseline": {
            key: i2e[key] - baseline[key]
            for key in ("mIoU", "Acc@0.5", "DVGBench_AVG")
        },
        "delta_trained_direct_vs_baseline": {
            key: trained_direct[key] - baseline[key]
            for key in ("mIoU", "Acc@0.5", "DVGBench_AVG")
        },
        "max_allowed_regression": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 4)


if __name__ == "__main__":
    main()
