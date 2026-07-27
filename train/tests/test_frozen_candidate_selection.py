from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from uav_agentic.frozen_candidate_selection import (
    SelectionConfig,
    apply_selection_to_trace,
    hypothesis_representatives,
    parse_verifier_json,
    render_candidate_overlay,
    select_frozen_candidate,
)


def candidate(
    candidate_id: str,
    bbox: list[float],
    source: str,
    confidence: float = 0.6,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "bbox": bbox,
        "source_agent": source,
        "query_used": "the car left of the road",
        "observation": {
            "observation_id": "full",
            "view_type": "full_image",
            "coordinate_frame": "global_normalized",
            "preserves_context": True,
        },
        "bbox_token_confidence": confidence,
        "confidence_available": True,
        "box_plausibility": 1.0,
        "target_consistency": 0.5,
        "context_consistency": 0.5,
        "relation_consistency": 0.6,
        "global_constraint_score": 0.6,
        "fused_score": confidence,
        "accepted_by_guard": True,
        "rejection_reasons": [],
        "parent_candidate_id": None,
        "hypothesis_id": candidate_id,
    }


def trace() -> dict:
    initial = candidate("c00", [0.1, 0.1, 0.2, 0.2], "BaseGrounder", 0.55)
    alternative = candidate("c01", [0.6, 0.6, 0.8, 0.8], "TargetAgent", 0.62)
    return {
        "schema_version": "dai-uav-agent-v4.1",
        "sample_id": "sample",
        "class": "traffic",
        "method": "hierarchical",
        "inference": {
            "query": "the car left of the road",
            "question_e_used": False,
            "constraint_graph": {
                "target": "car",
                "attributes": [],
                "context": "road",
                "relations": ["left"],
                "global_position": None,
                "ordinal_constraint": None,
            },
            "initial_candidate": initial,
            "target_candidates": [initial, alternative],
            "hypothesis_clusters": [
                {
                    "hypothesis_id": "c00",
                    "root_candidate_id": "c00",
                    "representative_candidate_id": "c00",
                },
                {
                    "hypothesis_id": "c01",
                    "root_candidate_id": "c01",
                    "representative_candidate_id": "c01",
                },
            ],
            "verification_evidence": {
                "fusion": {"ranked_candidate_ids": ["c01", "c00"]}
            },
            "final_candidate_id": "c00",
            "final_bbox": initial["bbox"],
            "confidence": 0.55,
            "decision": "accept",
            "stop_reason": "initial",
            "action_trace": [],
            "agent_calls": [],
            "unit_calls": [],
            "child_calls": [],
        },
        "cost": {
            "perception_calls": 3,
            "executed_perception_calls": 2,
            "incremental_agent_latency_ms": 100.0,
            "end_to_end_latency_ms": 1000.0,
            "latency_ms": 1000.0,
            "dispatch": True,
        },
        "image": "sample.png",
        "bbox": initial["bbox"],
        "parse_ok": True,
        "evaluation": {
            "gt_bbox": [0.6, 0.6, 0.8, 0.8],
            "initial_iou": 0.0,
            "final_iou": 0.0,
        },
    }


def verifier(
    selected: str | None = "c01",
    confidence: float = 0.9,
    abstained: bool = False,
) -> dict:
    return {
        "sample_id": "sample",
        "status": "abstained" if abstained else "completed",
        "abstained": abstained,
        "selected_hypothesis_id": None if abstained else selected,
        "confidence": confidence,
        "hypothesis_scores": (
            {"c00": 0.1, "c01": 0.9}
            if not abstained
            else {"c00": 0.5, "c01": 0.5}
        ),
        "question_e_used": False,
        "gt_visible": False,
        "model_calls": 1,
        "latency_ms": 50.0,
    }


class FrozenSelectionUnitTests(unittest.TestCase):
    def test_high_confidence_visual_evidence_replaces_initial(self):
        result = select_frozen_candidate(
            trace()["inference"],
            verifier(),
            SelectionConfig(minimum_composite_gain=0.0),
        )
        self.assertEqual(result["selected_candidate_id"], "c01")
        self.assertTrue(result["visual_supported"])
        self.assertFalse(result["gt_visible"])

    def test_abstention_keeps_initial(self):
        result = select_frozen_candidate(
            trace()["inference"], verifier(abstained=True), SelectionConfig()
        )
        self.assertEqual(result["selected_candidate_id"], "c00")
        self.assertEqual(result["guard_reason"], "verifier_abstained_keep_initial")

    def test_low_confidence_keeps_initial(self):
        result = select_frozen_candidate(
            trace()["inference"],
            verifier(confidence=0.55),
            SelectionConfig(verifier_confidence_threshold=0.70),
        )
        self.assertEqual(result["selected_candidate_id"], "c00")
        self.assertEqual(result["guard_reason"], "low_verifier_confidence_keep_initial")

    def test_apply_selection_drops_preexisting_evaluation(self):
        result = apply_selection_to_trace(
            trace(),
            verifier(),
            SelectionConfig(minimum_composite_gain=0.0),
        )
        self.assertNotIn("evaluation", result)
        self.assertEqual(result["bbox"], [0.6, 0.6, 0.8, 0.8])
        self.assertEqual(result["cost"]["candidate_verifier_calls"], 1)
        self.assertEqual(result["cost"]["end_to_end_latency_ms"], 1050.0)
        self.assertFalse(
            result["inference"]["candidate_selection"]["question_e_used"]
        )

    def test_representatives_are_unique_hypotheses(self):
        inference = trace()["inference"]
        child = copy.deepcopy(inference["target_candidates"][1])
        child["candidate_id"] = "c02"
        child["parent_candidate_id"] = "c01"
        child["hypothesis_id"] = "c01"
        inference["target_candidates"].append(child)
        representatives = hypothesis_representatives(inference)
        self.assertEqual(
            [item["candidate_id"] for item in representatives], ["c00", "c01"]
        )

    def test_overlay_preserves_full_image_size_and_mapping(self):
        image = Image.new("RGB", (100, 80), "white")
        overlay, mapping = render_candidate_overlay(
            image, trace()["inference"]["target_candidates"]
        )
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(mapping, {"H0": "c00", "H1": "c01"})

    def test_json_parser_accepts_fenced_json(self):
        parsed = parse_verifier_json(
            '```json\n{"selected":"H1","confidence":0.8}\n```'
        )
        self.assertEqual(parsed["selected"], "H1")

    def test_question_e_guard_is_mandatory(self):
        inference = trace()["inference"]
        inference["question_e_used"] = True
        with self.assertRaisesRegex(ValueError, "question_e_used"):
            select_frozen_candidate(inference, verifier(), SelectionConfig())


class FrozenSelectionCliTests(unittest.TestCase):
    def test_cli_joins_gt_only_after_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            sidecar_path = root / "verifier.jsonl"
            index_path = root / "index.jsonl"
            output_path = root / "selected.jsonl"
            summary_path = root / "summary.json"
            trace_path.write_text(
                json.dumps(trace()) + "\n", encoding="utf-8"
            )
            sidecar_path.write_text(
                json.dumps(verifier()) + "\n", encoding="utf-8"
            )
            index_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "image": "sample.png",
                        "question": "the car left of the road",
                        "question_e": "oracle must not be used",
                        "bbox_norm": [0.6, 0.6, 0.8, 0.8],
                        "class": "traffic",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train_root = Path(__file__).resolve().parents[1]
            script = (
                train_root
                / "src"
                / "tools"
                / "eval_dvgbench_frozen_candidate_selection.py"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(train_root / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--candidate-trace",
                    str(trace_path),
                    "--index",
                    str(index_path),
                    "--verifier-evidence",
                    str(sidecar_path),
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                    "--minimum-composite-gain",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["bbox"], [0.6, 0.6, 0.8, 0.8])
            self.assertFalse(result["inference"]["question_e_used"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["agentic_inference"]["Acc@0.5"], 1.0)
            self.assertFalse(
                summary["config"]["protocol"]["gt_visible_during_selection"]
            )


if __name__ == "__main__":
    unittest.main()
