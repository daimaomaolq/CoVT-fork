import pytest

pytest.importorskip("torch")

from training.train_utils import keep_compact_non_lora_parameter


def test_compact_i2e_checkpoint_keeps_schema_and_anchor_state() -> None:
    assert keep_compact_non_lora_parameter(
        "base_model.model.model.embed_tokens.base_layer.weight"
    )
    assert keep_compact_non_lora_parameter(
        "base_model.model.lm_head.base_layer.weight"
    )
    assert keep_compact_non_lora_parameter("sam_projection.weight")
    assert keep_compact_non_lora_parameter("dino_cross_attention.in_proj_weight")
    assert not keep_compact_non_lora_parameter(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    )
