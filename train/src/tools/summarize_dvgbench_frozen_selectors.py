#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SELECTORS = ("initial", "stored_fusion", "visual_only", "conservative_visual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen-candidate selector comparison table."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--csv-output")
    return parser.parse_args()


def get(mapping: dict[str, Any], *path: str, default: Any = 0.0) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def selector_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["agentic_inference"]
    candidate = summary.get("candidate_and_selection", {})
    selection = summary.get("frozen_candidate_selection", {})
    verifier = summary.get("candidate_verifier", {})
    actions = summary.get("posthoc_action_quality", {})
    return {
        "Selector": name,
        "Samples": summary.get("samples", 0),
        "mIoU": metrics.get("mIoU", 0.0),
        "Acc@0.5": metrics.get("Acc@0.5", 0.0),
        "DVGBench_AVG": metrics.get("DVGBench_AVG", 0.0),
        "Recovery@0.5": metrics.get("Recovery@0.5", 0.0),
        "Regression@0.5": metrics.get("Regression@0.5", 0.0),
        "Net Recovery Count": metrics.get("Net Recovery Count", 0),
        "Avg Calls": metrics.get("Avg Calls", 0.0),
        "Latency_ms": metrics.get("Latency_ms", 0.0),
        "Dispatch Rate": metrics.get("Dispatch Rate", 0.0),
        "Candidate Oracle Acc@0.5": candidate.get(
            "Candidate Oracle Acc@0.5", 0.0
        ),
        "Alternative Selection Success": candidate.get(
            "Alternative Selection Success", 0.0
        ),
        "Replacements": selection.get("Replacements", 0),
        "Visual-supported Replacements": selection.get(
            "Visual-supported Replacements", 0
        ),
        "Verifier Coverage": verifier.get("Coverage", 0.0),
        "Verifier Abstain Rate": verifier.get("Abstain Rate", 0.0),
        "Verifier Mean Confidence": verifier.get("Mean Confidence", 0.0),
        "Useful Call Rate@DeltaIoU0.1": actions.get(
            "Useful Call Rate@DeltaIoU0.1", 0.0
        ),
        "Mean Action Regret": actions.get("Mean Action Regret", 0.0),
        "question_e_used": get(
            summary, "protocol_guards", "question_e_used", default=True
        ),
        "gt_visible_during_selection": get(
            summary,
            "config",
            "protocol",
            "gt_visible_during_selection",
            default=True,
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    rows = []
    for selector in SELECTORS:
        path = root / f"{selector}.summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing selector summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("samples") != 873:
            raise ValueError(f"{path} must contain 873 samples")
        row = selector_row(selector, summary)
        if row["question_e_used"] is not False:
            raise ValueError(f"Unsafe question_e protocol in {path}")
        if row["gt_visible_during_selection"] is not False:
            raise ValueError(f"Unsafe GT protocol in {path}")
        rows.append(row)

    baseline = next(row for row in rows if row["Selector"] == "initial")
    for row in rows:
        row["Delta mIoU"] = row["mIoU"] - baseline["mIoU"]
        row["Delta Acc@0.5"] = row["Acc@0.5"] - baseline["Acc@0.5"]
        row["Delta DVGBench_AVG"] = (
            row["DVGBench_AVG"] - baseline["DVGBench_AVG"]
        )

    json_path = (
        Path(args.json_output).expanduser().resolve()
        if args.json_output
        else root / "selector_comparison.json"
    )
    csv_path = (
        Path(args.csv_output).expanduser().resolve()
        if args.csv_output
        else root / "selector_comparison.csv"
    )
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "frozen_candidate_pool": True,
                    "question_e_used": False,
                    "gt_visible_during_selection": False,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
