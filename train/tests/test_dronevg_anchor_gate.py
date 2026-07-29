from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gated44_script_keeps_joint_anchor_loss_enabled():
    text = (ROOT / "train/scripts/run_dvgbench_dronevg_qtsa_gated44.sh").read_text(encoding="utf-8")
    assert '--anchor_token_counts "[4,4,4,4,4,4,4,4]"' in text
    assert "--anchor_gate_mode query_conditioned" in text
    assert "--vqa_only_stage 100000" in text
    assert "dvgbench_dronevg_r1_qtsa_gated44_" in text


def test_gate_parameters_are_trainable_under_frozen_warmstart():
    text = (ROOT / "train/src/training/train.py").read_text(encoding="utf-8")
    assert '"_prompt_embeddings"' in text
    assert '"_gate_"' in text


def test_sam_queries_follow_prompt_token_count():
    text = (ROOT / "train/src/training/covt_qwen2_5_vl.py").read_text(encoding="utf-8")
    assert "self.sam_query_vectors[:sam_projected.shape[1]]" in text
    assert "self.sam_query_vectors[:sam_token_embeddings.shape[1]]" in text
    assert "supports at most 4 tokens" in text
