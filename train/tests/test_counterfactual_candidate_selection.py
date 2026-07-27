from __future__ import annotations

import copy
import unittest

from PIL import Image

from train.src.uav_agentic.counterfactual_candidate_selection import (
    COUNTERFACTUAL_SCHEMA_VERSION,
    CounterfactualConfig,
    apply_counterfactual_selection,
    candidate_independent_score,
    eligible_alternatives,
    parse_counterfactual_choice,
    render_counterfactual_sheet,
    select_counterfactual_candidate,
    verify_counterfactual_candidates,
)


def make_candidate(
    candidate_id: str,
    bbox: list[float],
    source: str,
    token: float,
    parent: str | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "bbox": bbox,
        "source_agent": source,
        "parent_candidate_id": parent,
        "hypothesis_id": parent or candidate_id,
        "bbox_token_confidence": token,
        "confidence_available": True,
        "parse_ok": True,
        "target_consistency": 0.5,
        "relation_consistency": 0.5,
        "global_constraint_score": 0.5,
        "box_plausibility": 1.0,
        "accepted_by_guard": True,
        "rejection_reasons": [],
    }


def make_inference(decision: str = "escalate", supported: bool = True) -> dict:
    initial = make_candidate("c00", [0.1, 0.1, 0.25, 0.25], "BaseGrounder", 0.40)
    alternative = make_candidate(
        "c01", [0.55, 0.5, 0.72, 0.72], "TargetAgent", 0.50
    )
    return {
        "query": "the red car near the building",
        "constraint_graph": {
            "target": "red car",
            "attributes": ["red"],
            "context": "building",
            "relations": ["near"],
            "global_position": None,
            "ordinal_constraint": None,
        },
        "initial_candidate": initial,
        "target_candidates": [initial, alternative],
        "hypothesis_clusters": [
            {
                "hypothesis_id": "c00",
                "member_candidate_ids": ["c00"],
                "cross_view_supported": False,
            },
            {
                "hypothesis_id": "c01",
                "member_candidate_ids": ["c01"],
                "cross_view_supported": supported,
                "cross_view_iou": 0.4 if supported else 0.0,
                "independent_verification_ids": ["z01"] if supported else [],
            },
        ],
        "decision": decision,
        "question_e_used": False,
        "final_candidate_id": "c00",
        "final_hypothesis_id": "c00",
        "final_bbox": list(initial["bbox"]),
        "action_trace": [],
        "agent_calls": [],
        "unit_calls": [],
    }


class QueueVerifier:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    def generate_visual_text(self, image, prompt, max_new_tokens=12):
        del image, prompt, max_new_tokens
        return self.outputs.pop(0)


class CounterfactualSelectionTests(unittest.TestCase):
    def test_choice_parser_requires_unambiguous_single_choice(self):
        self.assertEqual(parse_counterfactual_choice("<A>"), "A")
        self.assertEqual(parse_counterfactual_choice('"B"'), "B")
        self.assertEqual(parse_counterfactual_choice('{"selected":"X"}'), "X")
        self.assertIsNone(parse_counterfactual_choice("<A> or <B>"))
        self.assertIsNone(parse_counterfactual_choice("A B X"))

    def test_parent_accept_blocks_route(self):
        alternatives, audit = eligible_alternatives(
            make_inference(decision="accept"), CounterfactualConfig()
        )
        self.assertEqual(alternatives, [])
        self.assertEqual(audit["route_reason"], "parent_already_accepted")

    def test_independent_support_is_required(self):
        alternatives, audit = eligible_alternatives(
            make_inference(supported=False), CounterfactualConfig()
        )
        self.assertEqual(alternatives, [])
        reasons = audit["rejected"][0]["reasons"]
        self.assertIn("no_independent_support", reasons)

    def test_lower_independent_score_is_rejected(self):
        inference = make_inference()
        inference["target_candidates"][1]["bbox_token_confidence"] = 0.10
        alternatives, audit = eligible_alternatives(
            inference, CounterfactualConfig()
        )
        self.assertEqual(alternatives, [])
        reasons = audit["rejected"][0]["reasons"]
        self.assertIn("lower_independent_score", reasons)

    def test_concrete_zoom_box_is_eligible(self):
        inference = make_inference(supported=False)
        zoom = make_candidate(
            "c02", [0.62, 0.55, 0.82, 0.82], "ZoomAgent", 0.55, parent="c01"
        )
        inference["target_candidates"].append(zoom)
        alternatives, _ = eligible_alternatives(
            inference, CounterfactualConfig(max_alternatives=2)
        )
        self.assertEqual(alternatives[0]["candidate_id"], "c02")

    def test_counterbalanced_alternative_win(self):
        row = {"sample_id": "s1", "inference": make_inference()}
        evidence = verify_counterfactual_candidates(
            QueueVerifier(["<B>", "<A>"]),
            Image.new("RGB", (800, 600), "gray"),
            row,
            CounterfactualConfig(max_alternatives=1),
        )
        self.assertEqual(evidence["winner_candidate_id"], "c01")
        self.assertEqual(evidence["model_calls"], 2)
        self.assertEqual(
            evidence["pairwise_records"][0]["outcome"], "alternative_wins"
        )

    def test_label_bias_becomes_counterbalance_disagreement(self):
        row = {"sample_id": "s1", "inference": make_inference()}
        evidence = verify_counterfactual_candidates(
            QueueVerifier(["<A>", "<A>"]),
            Image.new("RGB", (800, 600), "gray"),
            row,
            CounterfactualConfig(max_alternatives=1),
        )
        self.assertIsNone(evidence["winner_candidate_id"])
        self.assertEqual(
            evidence["pairwise_records"][0]["outcome"],
            "counterbalance_disagreement",
        )

    def test_selector_rechecks_consensus(self):
        inference = make_inference()
        forged = {
            "winner_candidate_id": "c01",
            "pairwise_records": [
                {
                    "alternative_candidate_id": "c01",
                    "outcome": "alternative_wins",
                    "passes": [
                        {"chosen_candidate_id": "c01"},
                        {"chosen_candidate_id": "c00"},
                    ],
                }
            ],
        }
        selection = select_counterfactual_candidate(
            inference, forged, CounterfactualConfig()
        )
        self.assertFalse(selection["replaced_initial"])

    def test_apply_selection_removes_gt_before_commit(self):
        row = {
            "schema_version": "old",
            "method": "hierarchical",
            "inference": make_inference(),
            "evaluation": {"gt_bbox": [0.5, 0.5, 0.7, 0.7]},
            "cost": {},
        }
        evidence = {
            "winner_candidate_id": "c01",
            "pairwise_records": [
                {
                    "alternative_candidate_id": "c01",
                    "outcome": "alternative_wins",
                    "passes": [
                        {"chosen_candidate_id": "c01"},
                        {"chosen_candidate_id": "c01"},
                    ],
                }
            ],
            "model_calls": 2,
            "latency_ms": 10.0,
            "status": "completed",
            "question_e_used": False,
            "gt_visible": False,
        }
        result = apply_counterfactual_selection(
            row, evidence, CounterfactualConfig()
        )
        self.assertNotIn("evaluation", result)
        self.assertEqual(result["schema_version"], COUNTERFACTUAL_SCHEMA_VERSION)
        self.assertEqual(result["inference"]["final_candidate_id"], "c01")
        self.assertEqual(result["cost"]["counterfactual_verifier_calls"], 2)

    def test_render_preserves_mapping_and_adds_zoom_panels(self):
        inference = make_inference()
        sheet, mapping = render_counterfactual_sheet(
            Image.new("RGB", (800, 600), "gray"),
            inference["initial_candidate"],
            inference["target_candidates"][1],
            swap=True,
            crop_scale=4.0,
        )
        self.assertEqual(mapping, {"A": "c01", "B": "c00"})
        self.assertEqual(sheet.width, 1008)
        self.assertGreater(sheet.height, 800)

    def test_independent_score_does_not_read_visual_evidence(self):
        candidate = make_inference()["initial_candidate"]
        baseline = candidate_independent_score(candidate)
        candidate["visual_score"] = 1.0
        candidate["verifier_confidence"] = 1.0
        self.assertEqual(candidate_independent_score(candidate), baseline)


if __name__ == "__main__":
    unittest.main()
