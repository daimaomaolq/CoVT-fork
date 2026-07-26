from __future__ import annotations

import json
import inspect
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from uav_agentic.agents.base import AgentContext
from uav_agentic.agents.zoom_agent import ZoomAgent
from uav_agentic.fusion import FusionResult, rank_candidates
from uav_agentic.io import load_cached_predictions
from uav_agentic.grounder import CoVTGrounder
from uav_agentic.parent_agent import HierarchicalParentAgent, _apply_false_repair_guard
from uav_agentic.query_constraints import parse_query_constraints
from uav_agentic.schema import AgenticConfig, Candidate, Method, Observation


def make_candidate(candidate_id="c00", bbox=None, confidence=0.8, latency_ms=0.0):
    return Candidate(
        candidate_id=candidate_id,
        bbox=bbox or [0.1, 0.1, 0.2, 0.2],
        source_agent="BaseGrounder" if candidate_id == "c00" else "TargetAgent",
        query_used="the car",
        observation=Observation("full", "full_image"),
        bbox_token_confidence=confidence,
        bbox_token_count=4,
        latency_ms=latency_ms,
    )


class NoModelGrounder:
    def ground(self, *args, **kwargs):
        raise AssertionError("model grounding must not be called")

    def generate_base_text(self, *args, **kwargs):
        raise AssertionError("text generation must not be called")


class RecordingZoomGrounder:
    def __init__(self):
        self.query = None

    def ground(
        self,
        image,
        query,
        candidate_id,
        source_agent,
        observation,
        parent_candidate_id=None,
    ):
        self.query = query
        return Candidate(
            candidate_id=candidate_id,
            bbox=[0.25, 0.25, 0.75, 0.75],
            source_agent=source_agent,
            query_used=query,
            observation=observation,
            bbox_token_confidence=0.8,
            bbox_token_count=4,
            parent_candidate_id=parent_candidate_id,
        )


class ReleaseAuditTests(unittest.TestCase):
    def test_grounder_has_no_generic_qwen_fallback(self):
        source = inspect.getsource(CoVTGrounder.load)
        self.assertIn("CoVTForConditionalGeneration", source)
        self.assertNotIn("Qwen2_5_VLForConditionalGeneration", source)
        self.assertIn("Path(__file__).resolve().parents[3]", source)
        self.assertIn("sys.path.insert", source)
        ground_source = inspect.getsource(CoVTGrounder.ground)
        feedback_source = inspect.getsource(CoVTGrounder.generate_base_text)
        self.assertGreaterEqual(ground_source.count("torch.cuda.empty_cache()"), 2)
        self.assertGreaterEqual(feedback_source.count("torch.cuda.empty_cache()"), 2)
        self.assertIn("_autocast_context", ground_source)
        self.assertIn("_autocast_context", feedback_source)
        autocast_source = inspect.getsource(CoVTGrounder._autocast_context)
        self.assertIn("torch_module.autocast", autocast_source)

    def test_formal_cache_requires_measured_token_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "bbox": [0.1, 0.1, 0.2, 0.2],
                        "raw_output": "{<100><100><200><200>}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "no measured bbox-token confidence"
            ):
                load_cached_predictions(path, require_confidence=True)

    def test_cached_initial_latency_is_in_end_to_end_cost(self):
        config = AgenticConfig(method=Method.ONE_PASS, feedback_mode="off")
        result = HierarchicalParentAgent(NoModelGrounder(), config).run(
            "s1",
            Image.new("RGB", (16, 16), "white"),
            "the car",
            cached_initial=make_candidate(latency_ms=12.5),
        )
        self.assertEqual(result["cost"]["initial_latency_ms"], 12.5)
        self.assertGreaterEqual(result["cost"]["end_to_end_latency_ms"], 12.5)
        self.assertIn("unit_calls", result["inference"])
        self.assertEqual(
            result["inference"]["unit_calls"], result["inference"]["child_calls"]
        )

    def test_constraint_graph_ablation_is_explicit_in_trace(self):
        config = AgenticConfig(
            method=Method.ONE_PASS,
            feedback_mode="off",
            enable_constraint_graph=False,
        )
        result = HierarchicalParentAgent(NoModelGrounder(), config).run(
            "s1",
            Image.new("RGB", (16, 16), "white"),
            "the car left of the road",
            cached_initial=make_candidate(),
        )
        graph = result["inference"]["constraint_graph"]
        self.assertEqual(graph["parser_version"], "constraint-graph-disabled-ablation")
        self.assertEqual(graph["relations"], [])

    def test_false_repair_guard_restores_initial_candidate(self):
        initial = make_candidate("c00")
        initial.fused_score = 0.60
        refined = make_candidate("c01", [0.5, 0.5, 0.6, 0.6])
        refined.fused_score = 0.61
        fusion = FusionResult([refined, initial], refined, {})
        guarded = _apply_false_repair_guard(
            fusion,
            initial,
            0.60,
            AgenticConfig(false_repair_margin=0.02),
        )
        self.assertEqual(guarded.final.candidate_id, "c00")
        self.assertTrue(guarded.evidence["false_repair_guard_applied"])

    def test_false_repair_guard_accepts_strong_confidence_gain(self):
        initial = make_candidate("c00", confidence=0.40)
        initial.fused_score = 0.50
        refined = make_candidate("c01", [0.11, 0.11, 0.21, 0.21], confidence=0.65)
        refined.fused_score = 0.70
        fusion = FusionResult([refined, initial], refined, {})
        guarded = _apply_false_repair_guard(
            fusion,
            initial,
            0.30,
            AgenticConfig(),
        )
        self.assertEqual(guarded.final.candidate_id, "c01")
        self.assertEqual(
            guarded.evidence["replacement_support_evidence"],
            ["strong_token_confidence_gain"],
        )
        self.assertAlmostEqual(guarded.evidence["comparable_initial_score"], 0.50)

    def test_false_repair_guard_accepts_verification_advantage(self):
        initial = make_candidate("c00", bbox=[0.39, 0.39, 0.59, 0.59], confidence=0.55)
        initial.fused_score = 0.50
        refined = make_candidate("c01", [0.4, 0.4, 0.6, 0.6], confidence=0.45)
        refined.fused_score = 0.65
        zoom = Candidate(
            candidate_id="c02",
            bbox=[0.41, 0.41, 0.61, 0.61],
            source_agent="ZoomAgent",
            query_used="the car",
            observation=Observation(
                "zoom",
                "crop_zoom",
                crop_region=[0.30, 0.30, 0.70, 0.70],
            ),
            bbox_token_confidence=0.55,
            bbox_token_count=4,
            parent_candidate_id="c01",
        )
        zoom.fused_score = 0.60
        fusion = FusionResult(
            [refined, zoom, initial],
            refined,
            {
                "hypotheses": [
                    {
                        "hypothesis_id": "c00",
                        "cross_view_supported": True,
                        "verification_strength": 0.10,
                    },
                    {
                        "hypothesis_id": "c01",
                        "cross_view_supported": True,
                        "verification_strength": 0.30,
                        "supporting_verification_ids": ["c02"],
                    },
                ]
            },
        )
        guarded = _apply_false_repair_guard(
            fusion,
            initial,
            0.30,
            AgenticConfig(),
        )
        self.assertEqual(guarded.final.candidate_id, "c01")
        self.assertIn(
            "cross_view_zoom_confirmation",
            guarded.evidence["replacement_support_evidence"],
        )
        self.assertEqual(guarded.evidence["cross_view_partner_id"], "c02")
        self.assertTrue(guarded.evidence["verification_advantage"])
        self.assertIn(
            "cross_view_verification_advantage",
            guarded.evidence["replacement_support_evidence"],
        )

    def test_false_repair_guard_rejects_one_sided_self_confirmation(self):
        initial = make_candidate("c00", bbox=[0.10, 0.10, 0.20, 0.20], confidence=0.55)
        initial.fused_score = 0.50
        switched = make_candidate("c01", [0.70, 0.70, 0.80, 0.80], confidence=0.55)
        switched.fused_score = 0.70
        fusion = FusionResult(
            [switched, initial],
            switched,
            {
                "hypotheses": [
                    {
                        "hypothesis_id": "c00",
                        "cross_view_supported": True,
                        "verification_strength": 0.25,
                    },
                    {
                        "hypothesis_id": "c01",
                        "cross_view_supported": True,
                        "verification_strength": 0.25,
                        "supporting_verification_ids": ["c02"],
                    },
                ]
            },
        )
        guarded = _apply_false_repair_guard(
            fusion,
            initial,
            0.50,
            AgenticConfig(),
        )
        self.assertEqual(guarded.final.candidate_id, "c00")
        self.assertFalse(guarded.evidence["verification_advantage"])

    def test_false_repair_guard_rejects_identity_switch(self):
        initial = make_candidate("c00", confidence=0.40)
        initial.fused_score = 0.50
        switched = make_candidate("c01", [0.70, 0.70, 0.80, 0.80], confidence=0.80)
        switched.fused_score = 0.80
        fusion = FusionResult([switched, initial], switched, {})
        guarded = _apply_false_repair_guard(
            fusion,
            initial,
            0.50,
            AgenticConfig(),
        )
        self.assertEqual(guarded.final.candidate_id, "c00")
        self.assertEqual(
            guarded.evidence["false_repair_guard_reason"],
            "identity_not_preserved",
        )

    def test_target_candidate_cannot_self_certify_in_fusion(self):
        initial = make_candidate("c00", confidence=0.60)
        refined = make_candidate("c01", [0.5, 0.5, 0.6, 0.6], confidence=0.60)
        fusion = rank_candidates(
            [initial, refined],
            parse_query_constraints("the car"),
            AgenticConfig(),
        )
        self.assertEqual(fusion.final.candidate_id, "c00")
        self.assertEqual(initial.target_consistency, refined.target_consistency)

    def test_zoom_ablation_uses_naive_crop_query(self):
        graph = parse_query_constraints("the car in the upper left")
        grounder = RecordingZoomGrounder()
        seed = make_candidate(bbox=[0.1, 0.1, 0.14, 0.14])
        context = AgentContext(
            image=Image.new("RGB", (100, 100), "white"),
            graph=graph,
            grounder=grounder,
            config=AgenticConfig(enable_semantic_frame_protection=False),
            target_candidates=[seed],
            context_candidates=[],
        )
        result = ZoomAgent().run(context, seed, "c01")
        self.assertEqual(grounder.query, graph.original)
        self.assertEqual(result.call.action, "naive_crop_zoom")
        self.assertFalse(result.call.input["semantic_frame_protection"])


if __name__ == "__main__":
    unittest.main()
