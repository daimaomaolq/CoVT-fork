from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "train" / "src" / "tools"
sys.path.insert(0, str(REPO_ROOT / "train" / "src"))
sys.path.insert(0, str(REPO_ROOT / "train"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_dvgbench_generative_sft_i2e_test",
    TOOLS_DIR / "build_dvgbench_generative_sft.py",
)
evaluator = load_module(
    "eval_dvgbench_generative_grounding_i2e_test",
    TOOLS_DIR / "eval_dvgbench_generative_grounding.py",
)


def test_i2e_prompt_target_and_parser() -> None:
    prompt = builder.prompt_for_query("the car violating the traffic rule", "i2e")
    target = builder.answer_for_bbox(
        "{<100><200><300><400>}",
        "the car violating the traffic rule",
        "i2e",
        explicit_reference="the white car on the left",
    )
    assert "<think>" in prompt
    assert "<explicit>" in prompt
    assert "<answer>" in prompt
    assert target == (
        '<think>I resolve the implicit scene clue "the car violating the traffic '
        'rule" and verify the target using visible evidence.</think>\n'
        "<explicit>the white car on the left</explicit>\n"
        "<answer>{<100><200><300><400>}</answer>"
    )
    assert evaluator.parse_explicit_text(target) == "the white car on the left"
    assert evaluator.parse_bbox_text(target) == [0.1, 0.2, 0.3, 0.4]


def test_i2e_schema_guard_repairs_tags_without_changing_bbox() -> None:
    malformed = (
        '<think>I resolve the clue.</think>\n'
        '<explicit>A big van on the right side of the image><<answer>'
        '{523><506><575><545}</elementGuidId>\n<|im_end|>'
    )
    before = evaluator.parse_bbox_text(malformed)
    guarded, applied, reason = evaluator.guard_i2e_schema(malformed)

    assert applied is True
    assert reason == "repaired_tag_boundaries"
    assert evaluator.parse_explicit_text(guarded) == (
        "A big van on the right side of the image"
    )
    assert evaluator.parse_bbox_text(guarded) == before
    assert guarded.endswith("<answer>{<523><506><575><545>}</answer>")

    valid = (
        "<think>reason</think>\n"
        "<explicit>white car</explicit>\n"
        "<answer>{<1><2><3><4>}</answer>"
    )
    unchanged, applied, reason = evaluator.guard_i2e_schema(valid)
    assert unchanged == valid
    assert applied is False
    assert reason is None


def test_non_lora_prefix_mapping_loads_anchor_weights() -> None:
    class Tensor:
        shape = (2, 2)

    class Incompatible:
        missing_keys = ["unrelated.weight"]
        unexpected_keys = []

    class Model:
        def __init__(self) -> None:
            self.loaded = {}

        def state_dict(self):
            return {"sam_projection.weight": Tensor()}

        def load_state_dict(self, state_dict, strict=False):
            assert strict is False
            self.loaded = state_dict
            return Incompatible()

    model = Model()
    stats = evaluator.load_matching_non_lora_state(
        model,
        {"base_model.model.sam_projection.weight": Tensor()},
        "checkpoint/non_lora_state_dict.bin",
    )
    assert list(model.loaded) == ["sam_projection.weight"]
    assert stats["loaded"] == 1
    assert stats["skipped"] == 0
    assert stats["anchor_loaded"] == 1

def test_i2e_builder_and_oracle_free_eval_index(tmp_path: Path, monkeypatch) -> None:
    image_root = tmp_path / "images"
    image_dir = image_root / "era"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image_dir / "sample.png")

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "question_id": 7,
                "image_id": "sample.png",
                "bbox": [10, 20, 30, 40],
                "question": "the vehicle that should stop",
                "question_e": "the white vehicle at the upper left",
                "dataset": "era",
                "class": "traffic",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.json"
    index = tmp_path / "eval.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--input-jsonl",
            str(source),
            "--output",
            str(output),
            "--image-root",
            str(image_root),
            "--image-folder",
            str(image_root),
            "--query-field",
            "question",
            "--explicit-field",
            "question_e",
            "--mode",
            "i2e",
            "--i2e-answer-only-copy-ratio",
            "1.0",
            "--i2e-explicit-only-copy-ratio",
            "1.0",
            "--omit-oracle-fields-from-eval-index",
            "--write-eval-index",
            str(index),
        ],
    )
    builder.main()

    train_rows = json.loads(output.read_text(encoding="utf-8"))
    eval_row = json.loads(index.read_text(encoding="utf-8"))
    assert len(train_rows) == 3
    assert train_rows[0]["metadata"]["protocol"] == "i2e"
    assert train_rows[1]["metadata"]["protocol"] == "answer_only_preservation"
    assert train_rows[2]["metadata"]["protocol"] == "implicit_to_explicit_auxiliary"
    assert train_rows[2]["conversations"][1]["value"] == (
        "<explicit>the white vehicle at the upper left</explicit>"
    )
    assert "<think>" in train_rows[0]["conversations"][1]["value"]
    assert "<explicit>the white vehicle at the upper left</explicit>" in (
        train_rows[0]["conversations"][1]["value"]
    )
    assert "question_e" not in eval_row
    assert "question_e_cn" not in eval_row
    assert eval_row["oracle_fields_present"] is False
