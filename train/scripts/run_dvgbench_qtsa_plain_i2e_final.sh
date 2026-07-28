#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/CoVT-fork_i2e_6c766d5}"
PY="${PY:-/root/autodl-tmp/envs/covt-v8-py310/bin/python}"
DVG_ROOT="${DVG_ROOT:-/root/autodl-tmp/datasets/DVGBench}"
IMAGE_ROOT="$DVG_ROOT/images"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886}"
QTSA_ROOT="$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1"
TAG="${TAG:-plain_i2e_final_v1}"
OUT_DIR="$RUN_ROOT/checkpoints/dvgbench_qtsa_$TAG"
DATA_DIR="$DVG_ROOT/generative_qwen_i2e/$TAG"
PRED_DIR="$RUN_ROOT/predictions"
LOG_DIR="$RUN_ROOT/logs"
TRAIN_JSON="$DATA_DIR/train.json"
TRAIN_MANIFEST="$DATA_DIR/train.manifest.json"
TEST_INDEX="$DVG_ROOT/generative_qwen_i2e/dvg_test_question_oracle_free_eval.jsonl"
GPU_ID="${GPU_ID:-0}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-802816}"

mkdir -p "$OUT_DIR" "$DATA_DIR" "$PRED_DIR" "$LOG_DIR"
cd "$REPO"

QTSA_CKPT="$QTSA_ROOT"
if [[ ! -f "$QTSA_CKPT/adapter_model.safetensors" ]]; then
  QTSA_CKPT="$(find "$QTSA_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
test -f "$QTSA_CKPT/adapter_model.safetensors"
test -f "$TEST_INDEX"

"$PY" train/src/tools/build_dvgbench_plain_i2e_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_train.jsonl" \
  --image-root "$IMAGE_ROOT" \
  --image-folder "$IMAGE_ROOT" \
  --output "$TRAIN_JSON" \
  --manifest-output "$TRAIN_MANIFEST"

"$PY" - "$TRAIN_JSON" "$TEST_INDEX" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
test = [json.loads(x) for x in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if x.strip()]
assert len(train) == 3980
assert sum(x["metadata"]["protocol"] == "plain_i2e_joint" for x in train) == 1990
assert sum(x["metadata"]["protocol"] == "answer_only_preservation" for x in train) == 1990
assert all("<think>" not in x["conversations"][1]["value"] for x in train)
assert all("<explicit>" not in x["conversations"][1]["value"] for x in train)
assert len(test) == 873 and len({str(x["sample_id"]) for x in test}) == 873
assert all("question_e" not in x and "question_e_cn" not in x for x in test)
assert all(x.get("oracle_fields_present") is False for x in test)
print({"train": len(train), "joint": 1990, "direct": 1990, "test": 873, "oracle_free": True})
PY

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "['sam','dino']" \
  --load_anchor_teachers False \
  --anchor_loss_weight "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" \
  --anchor_token_counts "[8,4,4,4,4,4,4,4]" \
  --data_path "$TRAIN_JSON" \
  --image_folder "$IMAGE_ROOT" \
  --image_min_pixels "$MIN_PIXELS" \
  --image_max_pixels "$MAX_PIXELS" \
  --output_dir "$OUT_DIR" \
  --lora_weight_path "$QTSA_CKPT" \
  --anchor_prompt_mode query_tail \
  --anchor_response_mode none \
  --train_anchor_adapters False \
  --compact_non_lora_checkpoint True \
  --add_i2e_schema_tokens False \
  --i2e_answer_token_weight 8.0 \
  --i2e_format_token_weight 1.0 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_strategy no \
  --save_total_limit 1 \
  --logging_steps 10 \
  --bf16 True \
  --gradient_checkpointing True \
  --freeze_llm True \
  --freeze_vision_tower True \
  --lora_enable True \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --stage_0_step 0 \
  --stage_1_step 0 \
  --stage_2_step 0 \
  --vqa_only_stage -1 \
  --use_liger False \
  --disable_flash_attn2 True \
  --report_to none \
  2>&1 | tee "$LOG_DIR/train_dvgbench_qtsa_$TAG.log"

test -f "$OUT_DIR/adapter_model.safetensors"

GATE_PRED="$PRED_DIR/dvgbench_qtsa_$TAG.gate32.jsonl"
GATE_TRACE="$PRED_DIR/dvgbench_qtsa_$TAG.gate32.trace.jsonl"
GATE_SUMMARY="$PRED_DIR/dvgbench_qtsa_$TAG.gate32.summary.json"
DIRECT_PRED="$PRED_DIR/dvgbench_qtsa_$TAG.gate32.direct.jsonl"
DIRECT_SUMMARY="$PRED_DIR/dvgbench_qtsa_$TAG.gate32.direct.summary.json"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" train/src/tools/eval_dvgbench_plain_i2e_bbox_only.py \
  --index "$TEST_INDEX" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OUT_DIR" \
  --output "$GATE_PRED" \
  --trace-output "$GATE_TRACE" \
  --summary-output "$GATE_SUMMARY" \
  --limit 32 \
  --anchor-model-id "['sam','dino']" \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --anchor-prompt-mode query_tail \
  --image-min-pixels "$MIN_PIXELS" \
  --image-max-pixels "$MAX_PIXELS" \
  --max-new-tokens 96

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$TEST_INDEX" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OUT_DIR" \
  --output "$DIRECT_PRED" \
  --summary-output "$DIRECT_SUMMARY" \
  --limit 32 \
  --query-field query \
  --prompt-mode answer_only \
  --require-oracle-free-index \
  --anchor-model-id "['sam','dino']" \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --anchor-prompt-mode query_tail \
  --image-min-pixels "$MIN_PIXELS" \
  --image-max-pixels "$MAX_PIXELS" \
  --max-new-tokens 64 \
  --temperature 0

"$PY" - "$GATE_SUMMARY" "$DIRECT_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

joint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
direct = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
checks = {
    "joint_bbox_parse": joint["parse_failed"] == 0,
    "joint_explicit_parse": joint["explicit_parse_failed"] <= 1,
    "joint_schema_rate": joint["schema_format_rate"] >= 0.95,
    "direct_bbox_parse": direct["parse_failed"] == 0,
}
print({"joint": joint, "direct": direct, "checks": checks})
if not all(checks.values()):
    raise SystemExit("Plain-I2E gate failed; full evaluation is stopped.")
PY

FINAL_PRED="$PRED_DIR/dvgbench_qtsa_$TAG.jsonl"
FINAL_TRACE="$PRED_DIR/dvgbench_qtsa_$TAG.trace.jsonl"
FINAL_SUMMARY="$PRED_DIR/dvgbench_qtsa_$TAG.summary.json"
COMPARISON="$PRED_DIR/dvgbench_qtsa_$TAG.comparison.json"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" train/src/tools/eval_dvgbench_plain_i2e_bbox_only.py \
  --index "$TEST_INDEX" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OUT_DIR" \
  --output "$FINAL_PRED" \
  --trace-output "$FINAL_TRACE" \
  --summary-output "$FINAL_SUMMARY" \
  --anchor-model-id "['sam','dino']" \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --anchor-prompt-mode query_tail \
  --image-min-pixels "$MIN_PIXELS" \
  --image-max-pixels "$MAX_PIXELS" \
  --max-new-tokens 96

"$PY" - "$FINAL_PRED" "$FINAL_SUMMARY" "$COMPARISON" <<'PY'
import json
import sys
from pathlib import Path

pred_path, summary_path, output_path = map(Path, sys.argv[1:])
rows = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert len(rows) == 873 and len({str(x["sample_id"]) for x in rows}) == 873
assert all(set(x) == {"sample_id", "bbox", "protocol"} for x in rows)
assert all(x["protocol"]["final_output"] == "bbox_only" for x in rows)
assert all(x["protocol"]["question_e_used"] is False for x in rows)
assert all(x["protocol"]["gt_visible_during_inference"] is False for x in rows)
baselines = {
    "matched_resolution_qtsa": {
        "mIoU": 0.32589248236596774,
        "Acc@0.5": 0.3230240549828179,
        "DVGBench_AVG": 0.35870769347166687,
    },
    "formal_qtsa": {
        "mIoU": 0.33379627010060053,
        "Acc@0.5": 0.32989690721649484,
        "DVGBench_AVG": 0.36270966618744965,
    },
}
keys = ("mIoU", "Acc@0.5", "DVGBench_AVG")
result = {
    "protocol": "single_trajectory_plain_i2e_final_bbox_only",
    "metrics": {key: summary[key] for key in keys},
    "format": {
        "parse_failed": summary["parse_failed"],
        "explicit_parse_failed": summary["explicit_parse_failed"],
        "schema_format_rate": summary["schema_format_rate"],
    },
    "baselines": baselines,
    "delta": {
        name: {key: summary[key] - values[key] for key in keys}
        for name, values in baselines.items()
    },
    "question_e_train_only": True,
    "question_e_used_at_inference": False,
    "final_output_bbox_only": True,
}
output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
