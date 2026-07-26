from __future__ import annotations

import json
import inspect
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from uav_agentic.agents.base import AgentContext
from uav_agentic.agents.zoom_agent import ZoomAgent
from uav_agentic.fusion import FusionResult
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
