#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "Experiment",
    "Method",
    "Max Specialized Unit Calls",
    "Disabled Units",
    "mIoU",
    "Acc@0.5",
    "DVGBench_AVG",
    "Recovery@0.5",
    "False Repair Rate",
    "Failure Detection Recall",
    "CandidateRecall@2",
    "Alternative Candidate Recall@0.5",
    "Alternative Selection Success",
    "Search Yield@DeltaIoU0.1",
    "Mean Hypothesis Count",
    "Root Verification Rate",
    "Avg Calls",
    "Avg Specialized Unit Calls",
    "Initial Latency_ms",
    "Incremental Agent Latency_ms",
    "End-to-end Latency_ms",
    "Latency_ms",
    "Latency_P95_ms",
    "Dispatch Rate",
    "Coverage",
    "Selective Acc@0.5",
]

PER_CLASS_COLUMNS = [
    "Experiment",
    "Method",
    "Class",
    "Count",
    "One-pass Acc@0.5",
    "Agentic Acc@0.5",
    "Delta Acc@0.5",
]

FAILURE_RECOVERY_COLUMNS = [
    "Experiment",
    "Method",
    "Failure Type",
    "Count",
    "Initial Failures",
    "Recovered",
    "Recovery Rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect agentic summary JSON files")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument(
        "--per-class-csv-output",
        help="Defaults to <csv-output stem>_per_class.csv",
    )
    parser.add_argument(
        "--failure-csv-output",
        help="Defaults to <csv-output stem>_failure_recovery.csv",
    )
    return parser.parse_args()


def flatten(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    agentic = summary["agentic_inference"]
    config = summary.get("config", {}).get("agent", {})
    detection = summary.get("failure_detection", {})
    candidates = summary.get("candidate_and_selection", {})
    selective = summary.get("selective_prediction", {})
    return {
        "Experiment": path.name.removesuffix(".summary.json"),
        "Method": summary.get("method"),
        "Max Specialized Unit Calls": config.get("max_child_perception_calls"),
        "Disabled Units": ",".join(config.get("disabled_agents", [])),
        "mIoU": agentic.get("mIoU"),
        "Acc@0.5": agentic.get("Acc@0.5"),
        "DVGBench_AVG": agentic.get("DVGBench_AVG"),
        "Recovery@0.5": agentic.get("Recovery@0.5"),
        "False Repair Rate": agentic.get("False Repair Rate"),
        "Failure Detection Recall": detection.get("Recall"),
        "CandidateRecall@2": candidates.get("CandidateRecall@2"),
        "Alternative Candidate Recall@0.5": candidates.get(
            "Alternative Candidate Recall@0.5"
        ),
        "Alternative Selection Success": candidates.get(
            "Alternative Selection Success"
        ),
        "Search Yield@DeltaIoU0.1": candidates.get("Search Yield@DeltaIoU0.1"),
        "Mean Hypothesis Count": candidates.get("Mean Hypothesis Count"),
        "Root Verification Rate": candidates.get("Root Verification Rate"),
        "Avg Calls": agentic.get("Avg Calls"),
        "Avg Specialized Unit Calls": agentic.get(
            "Avg Specialized Unit Calls", agentic.get("Avg Child Calls")
        ),
        "Initial Latency_ms": agentic.get("Initial Latency_ms"),
        "Incremental Agent Latency_ms": agentic.get("Incremental Agent Latency_ms"),
        "End-to-end Latency_ms": agentic.get(
            "End-to-end Latency_ms", agentic.get("Latency_ms")
        ),
        "Latency_ms": agentic.get("Latency_ms"),
        "Latency_P95_ms": agentic.get("Latency_P95_ms"),
        "Dispatch Rate": agentic.get("Dispatch Rate"),
        "Coverage": selective.get("Coverage"),
        "Selective Acc@0.5": selective.get("Selective Acc@0.5"),
    }


def flatten_per_class(path: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    initial = summary.get("one_pass", {}).get("class_Acc@0.5", {})
    final = summary.get("agentic_inference", {}).get("class_Acc@0.5", {})
    class_names = sorted(set(initial) | set(final))
    counts = summary.get("one_pass", {}).get("class_counts", {})
    experiment = path.name.removesuffix(".summary.json")
    method = summary.get("method")
    rows = []
    for class_name in class_names:
        initial_value = initial.get(class_name)
        final_value = final.get(class_name)
        rows.append(
            {
                "Experiment": experiment,
                "Method": method,
                "Class": class_name,
                "Count": counts.get(class_name),
                "One-pass Acc@0.5": initial_value,
                "Agentic Acc@0.5": final_value,
                "Delta Acc@0.5": (
                    float(final_value) - float(initial_value)
                    if initial_value is not None and final_value is not None
                    else None
                ),
            }
        )
    return rows


def flatten_failure_recovery(
    path: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    experiment = path.name.removesuffix(".summary.json")
    method = summary.get("method")
    return [
        {
            "Experiment": experiment,
            "Method": method,
            "Failure Type": failure_type,
            "Count": metrics.get("Count"),
            "Initial Failures": metrics.get("Initial Failures"),
            "Recovered": metrics.get("Recovered"),
            "Recovery Rate": metrics.get("Recovery Rate"),
        }
        for failure_type, metrics in sorted(
            summary.get("failure_type_recovery", {}).items()
        )
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    paths = sorted(input_dir.glob("*.summary.json"))
    if not paths:
        raise ValueError(f"No *.summary.json files in {input_dir}")
    rows = []
    per_class_rows = []
    failure_rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("protocol_guards", {}).get("question_e_used") is not False:
            raise ValueError(f"Unsafe or missing question_e guard in {path}")
        rows.append(flatten(path, summary))
        per_class_rows.extend(flatten_per_class(path, summary))
        failure_rows.extend(flatten_failure_recovery(path, summary))
    csv_path = Path(args.csv_output).expanduser().resolve()
    json_path = Path(args.json_output).expanduser().resolve()
    per_class_path = (
        Path(args.per_class_csv_output).expanduser().resolve()
        if args.per_class_csv_output
        else csv_path.with_name(f"{csv_path.stem}_per_class.csv")
    )
    failure_path = (
        Path(args.failure_csv_output).expanduser().resolve()
        if args.failure_csv_output
        else csv_path.with_name(f"{csv_path.stem}_failure_recovery.csv")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows, COLUMNS)
    write_csv(per_class_path, per_class_rows, PER_CLASS_COLUMNS)
    write_csv(failure_path, failure_rows, FAILURE_RECOVERY_COLUMNS)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "per_class_rows": len(per_class_rows),
                "failure_rows": len(failure_rows),
                "csv": str(csv_path),
                "per_class_csv": str(per_class_path),
                "failure_csv": str(failure_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
