from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from uav_agentic.agents.base import AgentContext
from uav_agentic.agents.relation_reasoner import pair_relation_score
from uav_agentic.agents.target_agent import TargetAgent, build_overlapping_tiles
from uav_agentic.agents.zoom_agent import ZoomAgent
from uav_agentic.evaluation import attach_evaluation, summarize
from uav_agentic.fusion import rank_candidates
from uav_agentic.io import (
    candidate_from_prediction,
    inference_input_from_row,
    load_cached_predictions,
)
from uav_agentic.parent_agent import HierarchicalParentAgent
from uav_agentic.query_constraints import parse_query_constraints
from uav_agentic.routing import DependencyAwareRouter
from uav_agentic.schema import AgenticConfig, Candidate, Method, Observation


class FakeGrounder:
    def __init__(self):
        self.calls = []

    def ground(
        self,
        image,
        query,
        candidate_id,
        source_agent,
        observation,
        parent_candidate_id=None,
    ):
        self.calls.append(
            {
                "query": query,
                "candidate_id": candidate_id,
                "source_agent": source_agent,
                "observation": observation,
            }
        )
        boxes = {
            "BaseGrounder": [0.70, 0.70, 0.80, 0.80],
            "TargetAgent": [0.10, 0.50, 0.20, 0.60],
            "ContextAgent": [0.00, 0.40, 0.40, 0.70],
            "ConfidenceGatedRerun": [0.68, 0.68, 0.80, 0.80],
            "ParentVerification": [0.68, 0.68, 0.80, 0.80],
            "ZoomAgent": [0.25, 0.25, 0.75, 0.75],
        }
        confidence = 0.90 if source_agent != "BaseGrounder" else 0.30
        return Candidate(
            candidate_id=candidate_id,
            bbox=boxes[source_agent],
            source_agent=source_agent,
            query_used=query,
            observation=observation,
            bbox_token_confidence=confidence,
            bbox_token_count=4,
            parent_candidate_id=parent_candidate_id,
        )

    def generate_base_text(self, prompt, max_new_tokens=256):
        return "{}"


def cached_candidate(box, confidence=0.9):
    return Candidate(
        candidate_id="c00",
        bbox=box,
        source_agent="BaseGrounder",
        query_used="",
        observation=Observation("full", "full_image"),
        bbox_token_confidence=confidence,
    )


class InputSafetyTests(unittest.TestCase):
    def test_question_e_is_rejected(self):
        row = {
            "sample_id": "s1",
            "image": "a.jpg",
            "question": "the car",
            "question_e": "the white car in the upper left",
            "bbox_norm": [0.1, 0.1, 0.2, 0.2],
        }
        sample = inference_input_from_row(row, "question")
        self.assertEqual(sample["query"], "the car")
        self.assertNotIn("bbox_norm", sample)
        self.assertNotIn("question_e", sample)
        with self.assertRaises(ValueError):
            inference_input_from_row(row, "question_e")

    def test_generic_query_with_oracle_provenance_is_rejected(self):
        row = {
            "sample_id": "s1",
            "image": "a.jpg",
            "query": "white car",
            "task_tag": "dvgbench_question_e_generative",
        }
        with self.assertRaises(ValueError):
            inference_input_from_row(row, "query")

    def test_dataset_index_cannot_be_used_as_prediction_cache(self):
        row = {
            "sample_id": "s1",
            "question": "the car",
            "question_e": "the white car",
            "answer": "{<100><200><300><400>}",
            "bbox": [64, 128, 192, 256],
            "bbox_norm": [0.1, 0.2, 0.3, 0.4],
            "task_type": "generative_grounding",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index_disguised_as_predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no prediction provenance"):
                load_cached_predictions(path)

    def test_prediction_token_coordinates_are_normalized(self):
        candidate = candidate_from_prediction(
            {
                "sample_id": "s1",
                "bbox": [100, 200, 300, 400],
                "raw_output": "{<100><200><300><400>}",
            }
        )
        self.assertEqual(candidate.bbox, [0.1, 0.2, 0.3, 0.4])


class QueryRoutingTests(unittest.TestCase):
    def test_relation_query_routes_target_context_and_relation(self):
        graph = parse_query_constraints("the tricycle parked on the crosswalk")
        initial = cached_candidate([0.6, 0.6, 0.7, 0.7])
        plan = DependencyAwareRouter().plan(
            initial,
            graph,
            AgenticConfig(max_child_perception_calls=2),
        )
        self.assertEqual(plan.perception_actions, ["target", "context"])
        self.assertTrue(plan.run_relation_reasoner)

    def test_hierarchical_runs_minimum_competition_probe(self):
        graph = parse_query_constraints("the red car")
        initial = cached_candidate([0.2, 0.2, 0.4, 0.4])
        plan = DependencyAwareRouter().plan(initial, graph, AgenticConfig())
        self.assertEqual(plan.perception_actions, ["target"])
        self.assertFalse(plan.run_relation_reasoner)
        disabled = DependencyAwareRouter().plan(
            initial,
            graph,
            AgenticConfig(competition_probe_mode="off"),
        )
        self.assertEqual(disabled.perception_actions, [])

    def test_no_query_rewrite_child_exists_in_routing(self):
        graph = parse_query_constraints(
            "the small red car parked on the crosswalk in the upper left"
        )
        initial = cached_candidate([0.1, 0.1, 0.12, 0.12], confidence=0.2)
        plan = DependencyAwareRouter().plan(initial, graph, AgenticConfig())
        self.assertNotIn("rewrite", plan.perception_actions)
        self.assertEqual(graph.local_target_query, "small red car")

    def test_target_clause_excludes_relation_context_attributes(self):
        graph = parse_query_constraints(
            "The grey car parked in front of the yellow fence"
        )
        self.assertEqual(graph.target, "car")
        self.assertEqual(graph.attributes, ["grey", "parked"])
        self.assertEqual(graph.context, "yellow fence")
        self.assertEqual(graph.local_target_query, "grey car parked")

    def test_implicit_target_query_preserves_semantic_clause(self):
        farmland = parse_query_constraints("Farmland waiting to be harvested")
        self.assertEqual(farmland.target, "farmland")
        self.assertEqual(
            farmland.local_target_query, "Farmland waiting to be harvested"
        )
        relation = parse_query_constraints(
            "The attacking player behind the defensive player"
        )
        self.assertEqual(relation.target, "player")
        self.assertEqual(relation.local_target_query, "attacking player")
        self.assertEqual(relation.context, "defensive player")
        temporal_relation = parse_query_constraints("A rider who is temporarily behind")
        self.assertEqual(temporal_relation.local_target_query, "rider")


class RelationAndZoomTests(unittest.TestCase):
    def test_overlap_relation_uses_target_bottom_center(self):
        score, details = pair_relation_score(
            [0.10, 0.50, 0.20, 0.60],
            [0.00, 0.40, 0.40, 0.70],
            ["overlap"],
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(details["overlap"], 1.0)

    def test_zoom_preserves_original_spatial_semantics_and_maps_box(self):
        grounder = FakeGrounder()
        graph = parse_query_constraints("the small car in the upper left")
        seed = cached_candidate([0.10, 0.10, 0.14, 0.14])
        context = AgentContext(
            image=Image.new("RGB", (100, 100), "white"),
            graph=graph,
            grounder=grounder,
            config=AgenticConfig(),
            target_candidates=[seed],
            context_candidates=[],
        )
        result = ZoomAgent().run(context, seed, "c01")
        query = grounder.calls[-1]["query"]
        self.assertEqual(query, "small car")
        self.assertNotIn("upper left", query)
        self.assertTrue(result.call.input["global_constraints_reapplied_after_mapping"])
        self.assertEqual(result.call.output["local_bbox"], [0.25, 0.25, 0.75, 0.75])
        self.assertIsNotNone(result.call.output["global_bbox"])
        self.assertEqual(result.candidates[0].hypothesis_id, seed.candidate_id)

    def test_target_transformed_view_maps_to_global_frame(self):
        grounder = FakeGrounder()
        graph = parse_query_constraints("the small car in the upper left")
        seed = cached_candidate([0.75, 0.75, 0.80, 0.80])
        context = AgentContext(
            image=Image.new("RGB", (100, 100), "white"),
            graph=graph,
            grounder=grounder,
            config=AgenticConfig(),
            target_candidates=[seed],
            context_candidates=[],
        )
        result = TargetAgent().run_transformed_view(
            context, "c01", [0.0, 0.0, 0.55, 0.55]
        )
        self.assertEqual(result.call.input["query"], "small car")
        self.assertEqual(result.call.output["local_bbox"], [0.10, 0.50, 0.20, 0.60])
        self.assertEqual(
            result.call.output["global_bbox"],
            [0.05500000000000001, 0.275, 0.11000000000000001, 0.33],
        )
        self.assertEqual(len(build_overlapping_tiles(2, 0.10)), 4)

    def test_hypothesis_fusion_uses_zoom_support_not_competitor_overlap(self):
        initial = cached_candidate([0.1, 0.1, 0.2, 0.2], confidence=0.65)
        alternative = Candidate(
            candidate_id="c01",
            bbox=[0.60, 0.60, 0.70, 0.70],
            source_agent="TargetAgent",
            query_used="car",
            observation=Observation("target", "full_image_target_probe"),
            bbox_token_confidence=0.45,
        )
        verification = Candidate(
            candidate_id="c02",
            bbox=[0.61, 0.61, 0.71, 0.71],
            source_agent="ZoomAgent",
            query_used="car",
            observation=Observation(
                "zoom",
                "crop_zoom",
                crop_region=[0.55, 0.55, 0.80, 0.80],
            ),
            bbox_token_confidence=0.50,
            parent_candidate_id="c01",
        )
        fusion = rank_candidates(
            [initial, alternative, verification],
            parse_query_constraints("the car"),
            AgenticConfig(),
        )
        self.assertEqual(fusion.evidence["top_hypothesis_id"], "c01")
        selected = fusion.evidence["selected_hypothesis"]
        self.assertTrue(selected["cross_view_supported"])
        self.assertEqual(selected["supporting_verification_ids"], ["c02"])

    def test_full_image_zoom_cannot_count_as_independent_support(self):
        initial = cached_candidate([0.1, 0.1, 0.2, 0.2], confidence=0.65)
        alternative = Candidate(
            candidate_id="c01",
            bbox=[0.60, 0.60, 0.70, 0.70],
            source_agent="TargetAgent",
            query_used="car",
            observation=Observation("target", "full_image_target_probe"),
            bbox_token_confidence=0.45,
        )
        repeated = Candidate(
            candidate_id="c02",
            bbox=[0.60, 0.60, 0.70, 0.70],
            source_agent="ZoomAgent",
            query_used="car",
            observation=Observation(
                "zoom",
                "crop_zoom",
                crop_region=[0.0, 0.0, 1.0, 1.0],
            ),
            bbox_token_confidence=0.50,
            parent_candidate_id="c01",
        )
        fusion = rank_candidates(
            [initial, alternative, repeated],
            parse_query_constraints("the car"),
            AgenticConfig(),
        )
        hypothesis = next(
            item
            for item in fusion.evidence["hypotheses"]
            if item["hypothesis_id"] == "c01"
        )
        self.assertFalse(hypothesis["cross_view_supported"])
        self.assertEqual(hypothesis["independent_verification_ids"], [])

    def test_object_relative_zoom_stays_a_transformed_target_view(self):
        grounder = FakeGrounder()
        graph = parse_query_constraints("A child in yellow standing on the road")
        seed = cached_candidate([0.07, 0.15, 0.09, 0.19])
        road = Candidate(
            candidate_id="x00",
            bbox=[0.0, 0.0, 1.0, 1.0],
            source_agent="ContextAgent",
            query_used="road",
            observation=Observation("context", "full_image"),
            bbox_token_confidence=0.8,
        )
        context = AgentContext(
            image=Image.new("RGB", (100, 100), "white"),
            graph=graph,
            grounder=grounder,
            config=AgenticConfig(),
            target_candidates=[seed],
            context_candidates=[road],
        )
        result = ZoomAgent().run(context, seed, "c01")
        crop = result.call.input["crop_region"]
        crop_area = (crop[2] - crop[0]) * (crop[3] - crop[1])
        self.assertLessEqual(crop_area, context.config.verification_max_crop_area)
        self.assertTrue(result.call.evidence["independent_transformed_view"])
        self.assertTrue(result.call.input["relation_checked_in_global_frame"])
        self.assertNotIn(
            "context_not_preserved", result.call.evidence["rejection_reasons"]
        )


class ParentIntegrationTests(unittest.TestCase):
    def test_hierarchy_recovers_relation_constrained_target(self):
        grounder = FakeGrounder()
        config = AgenticConfig(
            method=Method.HIERARCHICAL,
            max_child_perception_calls=2,
            feedback_mode="off",
        )
        result = HierarchicalParentAgent(grounder, config).run(
            "s1",
            Image.new("RGB", (100, 100), "white"),
            "the tricycle parked on the crosswalk",
        )
        self.assertEqual(result["inference"]["final_bbox"], [0.10, 0.50, 0.20, 0.60])
        self.assertEqual(result["cost"]["perception_calls"], 3)
        agents = [call["agent"] for call in result["inference"]["child_calls"]]
        self.assertEqual(agents, ["TargetAgent", "ContextAgent", "RelationAgent"])
        self.assertIn("diagnosis", result["inference"])
        self.assertIn("action_trace", result["inference"])
        self.assertFalse(result["inference"]["question_e_used"])
        self.assertLessEqual(
            result["inference"]["routing_plan"]["budget_used"],
            config.max_child_perception_calls,
        )
        self.assertIn("hypothesis_clusters", result["inference"])

    def test_small_target_uses_probe_tile_then_zoom_within_budget(self):
        grounder = FakeGrounder()
        config = AgenticConfig(
            method=Method.HIERARCHICAL,
            max_child_perception_calls=3,
            small_area_threshold=0.02,
            feedback_mode="off",
        )
        result = HierarchicalParentAgent(grounder, config).run(
            "s_tile",
            Image.new("RGB", (100, 100), "white"),
            "the small car",
            cached_initial=cached_candidate([0.10, 0.50, 0.20, 0.60], confidence=0.5),
        )
        actions = result["inference"]["routing_plan"]["executed_perception_actions"]
        self.assertEqual(
            actions,
            [
                "TargetAgent:global_competition_probe",
                "TargetAgent:transformed_view_target_search",
                "ZoomAgent:semantic_frame_preserving_zoom",
            ],
        )
        self.assertEqual(result["inference"]["routing_plan"]["budget_used"], 3)
        zoom = next(
            item
            for item in result["inference"]["target_candidates"]
            if item["source_agent"] == "ZoomAgent"
        )
        parent = next(
            item
            for item in result["inference"]["target_candidates"]
            if item["candidate_id"] == zoom["parent_candidate_id"]
        )
        self.assertEqual(parent["observation"]["view_type"], "target_search_tile")
        self.assertEqual(zoom["hypothesis_id"], parent["candidate_id"])

    def test_dynamic_scheduler_never_exceeds_budget_across_query_types(self):
        queries = [
            "the red car",
            "the tiny person",
            "the tricycle parked on the crosswalk",
            "the second vehicle from the left",
            "the car about to enter the road",
        ]
        disabled_sets = [set(), {"zoom"}, {"context"}, {"target"}]
        for budget in range(4):
            for query in queries:
                for disabled in disabled_sets:
                    with self.subTest(
                        budget=budget, query=query, disabled=sorted(disabled)
                    ):
                        config = AgenticConfig(
                            method=Method.HIERARCHICAL,
                            max_child_perception_calls=budget,
                            disabled_agents=disabled,
                            feedback_mode="off",
                        )
                        result = HierarchicalParentAgent(FakeGrounder(), config).run(
                            "budget_audit",
                            Image.new("RGB", (64, 64), "white"),
                            query,
                            cached_initial=cached_candidate(
                                [0.40, 0.40, 0.50, 0.50], confidence=0.5
                            ),
                        )
                        self.assertLessEqual(
                            result["cost"]["specialized_model_calls"], budget
                        )
                        self.assertEqual(
                            result["cost"]["specialized_model_calls"],
                            result["inference"]["routing_plan"]["budget_used"],
                        )

    def test_one_pass_cached_uses_no_model_call(self):
        grounder = FakeGrounder()
        config = AgenticConfig(method=Method.ONE_PASS, feedback_mode="off")
        result = HierarchicalParentAgent(grounder, config).run(
            "s2",
            Image.new("RGB", (20, 20), "white"),
            "the car",
            cached_initial=cached_candidate([0.2, 0.2, 0.4, 0.4]),
        )
        self.assertEqual(grounder.calls, [])
        self.assertEqual(result["cost"]["perception_calls"], 1)
        self.assertTrue(result["cost"]["cached_initial"])

    def test_temporal_uncertainty_escalates_with_operator_feedback(self):
        grounder = FakeGrounder()
        config = AgenticConfig(
            method=Method.HIERARCHICAL,
            max_child_perception_calls=1,
            feedback_mode="template",
        )
        result = HierarchicalParentAgent(grounder, config).run(
            "s3",
            Image.new("RGB", (20, 20), "white"),
            "the car about to enter the road",
            cached_initial=cached_candidate([0.2, 0.2, 0.4, 0.4]),
        )
        self.assertEqual(result["inference"]["decision"], "escalate")
        feedback = result["inference"]["human_feedback"]
        self.assertEqual(feedback["recommended_action"], "request_temporal_observation")


class CliSmokeTests(unittest.TestCase):
    def test_cached_one_pass_cli_runs_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            index_path = root / "index.jsonl"
            prediction_path = root / "cached.jsonl"
            output_path = root / "output.jsonl"
            summary_path = root / "summary.json"
            index_row = {
                "sample_id": "cli_1",
                "image": str(image_path),
                "question": "the car",
                "question_e": "the white car in the upper left",
                "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                "class": "vehicle",
            }
            cached_row = {
                "sample_id": "cli_1",
                "bbox": [0.1, 0.1, 0.3, 0.3],
                "bbox_token_confidence": 0.9,
            }
            index_path.write_text(json.dumps(index_row) + "\n", encoding="utf-8")
            prediction_path.write_text(json.dumps(cached_row) + "\n", encoding="utf-8")
            train_root = Path(__file__).resolve().parents[1]
            cli_path = train_root / "src" / "tools" / "eval_dvgbench_agentic_v3.py"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(train_root / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(cli_path),
                    "--index",
                    str(index_path),
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                    "--initial-predictions",
                    str(prediction_path),
                    "--method",
                    "one_pass",
                    "--feedback-mode",
                    "off",
                ],
                cwd=str(train_root.parent),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["agentic_inference"]["Acc@0.5"], 1.0)
            trace = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertFalse(trace["inference"]["question_e_used"])
            self.assertEqual(trace["cost"]["executed_perception_calls"], 0)
            resumed = subprocess.run(
                [*completed.args, "--resume"],
                cwd=str(train_root.parent),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("[resume] validated 1/1", resumed.stdout)
            self.assertEqual(
                len(output_path.read_text(encoding="utf-8").splitlines()), 1
            )


class MatrixSummaryTests(unittest.TestCase):
    def test_matrix_summary_writes_overall_class_and_failure_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "method": "hierarchical",
                "config": {
                    "agent": {
                        "max_child_perception_calls": 2,
                        "disabled_agents": [],
                    }
                },
                "protocol_guards": {"question_e_used": False},
                "one_pass": {
                    "class_Acc@0.5": {"traffic": 0.25, "vehicle": 0.50},
                    "class_counts": {"traffic": 4, "vehicle": 6},
                },
                "agentic_inference": {
                    "mIoU": 0.4,
                    "Acc@0.5": 0.5,
                    "DVGBench_AVG": 0.5,
                    "Recovery@0.5": 0.3,
                    "False Repair Rate": 0.1,
                    "Avg Calls": 2.0,
                    "Avg Child Calls": 1.0,
                    "Latency_ms": 100.0,
                    "Latency_P95_ms": 150.0,
                    "Dispatch Rate": 0.5,
                    "class_Acc@0.5": {"traffic": 0.50, "vehicle": 0.50},
                },
                "failure_detection": {"Recall": 0.8},
                "candidate_and_selection": {"CandidateRecall@2": 0.7},
                "selective_prediction": {
                    "Coverage": 0.9,
                    "Selective Acc@0.5": 0.6,
                },
                "failure_type_recovery": {
                    "relation_wrong": {
                        "Count": 4,
                        "Initial Failures": 3,
                        "Recovered": 2,
                        "Recovery Rate": 2 / 3,
                    }
                },
            }
            (root / "full.summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            train_root = Path(__file__).resolve().parents[1]
            cli_path = (
                train_root / "src" / "tools" / "summarize_dvgbench_agentic_v3_matrix.py"
            )
            csv_path = root / "matrix.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(cli_path),
                    "--input-dir",
                    str(root),
                    "--csv-output",
                    str(csv_path),
                    "--json-output",
                    str(root / "matrix.json"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            class_csv = (root / "matrix_per_class.csv").read_text(encoding="utf-8-sig")
            failure_csv = (root / "matrix_failure_recovery.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("Delta Acc@0.5", class_csv)
            self.assertIn("traffic,4,", class_csv)
            self.assertIn("relation_wrong", failure_csv)


class EvaluationTests(unittest.TestCase):
    def test_summary_has_grounding_agent_and_recall_metrics(self):
        grounder = FakeGrounder()
        config = AgenticConfig(
            method=Method.HIERARCHICAL,
            max_child_perception_calls=2,
            feedback_mode="off",
        )
        parent = HierarchicalParentAgent(grounder, config)
        first = parent.run(
            "s1",
            Image.new("RGB", (100, 100), "white"),
            "the tricycle parked on the crosswalk",
        )
        first = attach_evaluation(first, [0.10, 0.50, 0.20, 0.60])
        second = copy.deepcopy(first)
        second["sample_id"] = "s2"
        second["class"] = "vehicle"
        summary = summarize([first, second], {"question_e_used": False})
        self.assertIn("mIoU", summary["agentic_inference"])
        self.assertIn("Recovery@0.5", summary["agentic_inference"])
        self.assertIn("Recall", summary["failure_detection"])
        self.assertIn("CandidateRecall@2", summary["candidate_and_selection"])
        self.assertIn(
            "Alternative Candidate Recall@0.5",
            summary["candidate_and_selection"],
        )
        self.assertIn("Search Yield@DeltaIoU0.1", summary["candidate_and_selection"])
        self.assertIn("Latency_P95_ms", summary["agentic_inference"])
        self.assertEqual(summary["one_pass"]["class_counts"]["vehicle"], 1)
        self.assertFalse(summary["protocol_guards"]["question_e_used"])


if __name__ == "__main__":
    unittest.main()
