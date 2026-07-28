#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/CoVT-fork_i2e_6c766d5}"
ENV_PY="${ENV_PY:-/root/autodl-tmp/envs/covt-v8-py310/bin/python}"
DVG_ROOT="${DVG_ROOT:-/root/autodl-tmp/datasets/DVGBench}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886}"
QTSA_CKPT="${QTSA_CKPT:-$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1}"
TAG="${TAG:-v3b_gate32_format}"
OUT_DIR="$RUN_ROOT/checkpoints/dvgbench_qtsa_i2e_$TAG"
WORK_DIR="$DVG_ROOT/generative_qwen_i2e/$TAG"
PRED_DIR="$RUN_ROOT/predictions"
LOG_DIR="$RUN_ROOT/logs"
TRAIN_JSON="$WORK_DIR/train32_three_task.json"
GATE_INDEX="$WORK_DIR/train32_oracle_free_eval.jsonl"
I2E_PRED="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}.jsonl"
DIRECT_PRED="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}_direct.jsonl"
GATE_RESULT="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}.gate.json"

if [[ -e "$OUT_DIR" ]]; then
  echo "Gate output already exists: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$WORK_DIR" "$OUT_DIR" "$PRED_DIR" "$LOG_DIR"
cd "$REPO"

"$ENV_PY" train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_train.jsonl" \
  --output "$TRAIN_JSON" \
  --image-root "$DVG_ROOT/images" \
  --image-folder "$DVG_ROOT/images" \
  --query-field question \
  --explicit-field question_e \
  --mode i2e \
  --i2e-answer-only-copy-ratio 1.0 \
  --i2e-explicit-only-copy-ratio 1.0 \
  --limit 32 \
  --omit-oracle-fields-from-eval-index \
  --write-eval-index "$GATE_INDEX"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES=0 "$ENV_PY" -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "['sam','dino']" \
  --load_anchor_teachers False \
  --anchor_loss_weight "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" \
  --anchor_token_counts "[8,4,4,4,4,4,4,4]" \
  --data_path "$TRAIN_JSON" \
  --image_folder "$DVG_ROOT/images" \
  --image_min_pixels 200704 \
  --image_max_pixels 802816 \
  --output_dir "$OUT_DIR" \
  --lora_weight_path "$QTSA_CKPT" \
  --anchor_prompt_mode query_tail \
  --anchor_response_mode none \
  --train_anchor_adapters False \
  --compact_non_lora_checkpoint True \
  --i2e_answer_token_weight 5.0 \
  --i2e_format_token_weight 8.0 \
  --num_train_epochs 12 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --save_strategy no \
  --save_total_limit 1 \
  --logging_steps 5 \
  --bf16 True \
  --gradient_checkpointing True \
  --freeze_llm True \
  --freeze_vision_tower True \
  --lora_enable True \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.0 \
  --stage_0_step 0 \
  --stage_1_step 0 \
  --stage_2_step 0 \
  --vqa_only_stage -1 \
  --use_liger False \
  --disable_flash_attn2 True \
  --report_to none \
  2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_train.log"

run_eval() {
  local mode="$1"
  local output="$2"
  local max_tokens="$3"
  PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
  CUDA_VISIBLE_DEVICES=0 "$ENV_PY" train/src/tools/eval_dvgbench_generative_grounding.py \
    --index "$GATE_INDEX" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$OUT_DIR" \
    --output "$output" \
    --summary-output "${output%.jsonl}.summary.json" \
    --image-min-pixels 200704 \
    --image-max-pixels 802816 \
    --query-field query \
    --prompt-mode "$mode" \
    --i2e-schema-guard \
    --require-oracle-free-index \
    --anchor-model-id sam,dino \
    --anchor-prompt-mode query_tail \
    --max-new-tokens "$max_tokens" \
    --temperature 0 \
    --batch-size 1
}

run_eval i2e "$I2E_PRED" 128 2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_eval.log"
run_eval answer_only "$DIRECT_PRED" 64 2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_direct_eval.log"

"$ENV_PY" - "$I2E_PRED" "$DIRECT_PRED" "$GATE_RESULT" <<'PY'
import json
import sys
from pathlib import Path

def load_summary(pred):
    return json.loads(Path(pred).with_suffix('.summary.json').read_text(encoding='utf-8'))

i2e = load_summary(sys.argv[1])
direct = load_summary(sys.argv[2])
checks = {
    'i2e_parse_100': i2e['parse_failed'] == 0,
    'raw_schema_format_ge_0_95': i2e['raw_schema_format_rate'] >= 0.95,
    'guarded_schema_format_100': i2e['schema_parse_failed'] == 0,
    'i2e_acc_at_0_5_ge_0_90': i2e['Acc@0.5'] >= 0.90,
    'i2e_miou_ge_0_80': i2e['mIoU'] >= 0.80,
    'direct_parse_100': direct['parse_failed'] == 0,
    'direct_acc_at_0_5_ge_0_90': direct['Acc@0.5'] >= 0.90,
}
result = {
    'schema_version': 'dvgbench-qtsa-i2e-overfit-gate-v1',
    'samples': i2e['samples'],
    'i2e': {key: i2e[key] for key in ('mIoU', 'Acc@0.5', 'DVGBench_AVG', 'parse_failed', 'explicit_parse_failed', 'raw_explicit_parse_failed', 'raw_explicit_format_rate', 'explicit_format_rate', 'raw_schema_parse_failed', 'schema_parse_failed', 'raw_schema_format_rate', 'schema_format_rate', 'schema_guard_applied')},
    'direct': {key: direct[key] for key in ('mIoU', 'Acc@0.5', 'DVGBench_AVG', 'parse_failed')},
    'checks': checks,
    'passed': all(checks.values()),
    'question_e_used_at_inference': False,
    'gt_visible_during_inference': False,
}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['passed'] else 4)
PY