#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/CoVT-fork_i2e_6c766d5}"
ENV_PY="${ENV_PY:-/root/autodl-tmp/envs/covt-v8-py310/bin/python}"
DVG_ROOT="${DVG_ROOT:-/root/autodl-tmp/datasets/DVGBench}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886}"
QTSA_CKPT="${QTSA_CKPT:-$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1}"
TAG="${TAG:-v4_heldout_gate}"
OUT_DIR="$RUN_ROOT/checkpoints/dvgbench_qtsa_i2e_$TAG"
WORK_DIR="$DVG_ROOT/generative_qwen_i2e/$TAG"
PRED_DIR="$RUN_ROOT/predictions"
LOG_DIR="$RUN_ROOT/logs"
TRAIN_SOURCE="$WORK_DIR/source_train.jsonl"
VAL_SOURCE="$WORK_DIR/source_validation.jsonl"
SPLIT_MANIFEST="$WORK_DIR/split.manifest.json"
TRAIN_JSON="$WORK_DIR/train_three_task.json"
VAL_UNUSED_JSON="$WORK_DIR/validation_unused_sft.json"
VAL_INDEX="$WORK_DIR/validation_oracle_free_eval.jsonl"
I2E_PRED="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}.jsonl"
TRAINED_DIRECT_PRED="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}_trained_direct.jsonl"
BASELINE_PRED="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}_qtsa_baseline.jsonl"
GATE_RESULT="$PRED_DIR/dvgbench_qtsa_i2e_${TAG}.gate.json"

if [[ -e "$OUT_DIR" ]]; then
  echo "Gate output already exists: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$WORK_DIR" "$OUT_DIR" "$PRED_DIR" "$LOG_DIR"
cd "$REPO"

"$ENV_PY" train/src/tools/split_dvgbench_i2e_gate.py \
  --input "$DVG_ROOT/dvg_train.jsonl" \
  --train-output "$TRAIN_SOURCE" \
  --validation-output "$VAL_SOURCE" \
  --manifest-output "$SPLIT_MANIFEST" \
  --train-per-class 24 \
  --validation-per-class 8 \
  --seed 20260728

"$ENV_PY" train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$TRAIN_SOURCE" \
  --output "$TRAIN_JSON" \
  --image-root "$DVG_ROOT/images" \
  --image-folder "$DVG_ROOT/images" \
  --query-field question \
  --explicit-field question_e \
  --mode i2e \
  --i2e-answer-only-copy-ratio 1.0 \
  --i2e-explicit-only-copy-ratio 1.0

"$ENV_PY" train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$VAL_SOURCE" \
  --output "$VAL_UNUSED_JSON" \
  --image-root "$DVG_ROOT/images" \
  --image-folder "$DVG_ROOT/images" \
  --query-field question \
  --explicit-field question_e \
  --mode i2e \
  --omit-oracle-fields-from-eval-index \
  --write-eval-index "$VAL_INDEX"

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
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
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
  --lora_dropout 0.05 \
  --stage_0_step 0 \
  --stage_1_step 0 \
  --stage_2_step 0 \
  --vqa_only_stage -1 \
  --use_liger False \
  --disable_flash_attn2 True \
  --report_to none \
  2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_train.log"

run_eval() {
  local adapter="$1"
  local mode="$2"
  local output="$3"
  local max_tokens="$4"
  PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
  CUDA_VISIBLE_DEVICES=0 "$ENV_PY" train/src/tools/eval_dvgbench_generative_grounding.py \
    --index "$VAL_INDEX" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$adapter" \
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

run_eval "$OUT_DIR" i2e "$I2E_PRED" 192 \
  2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_eval.log"
run_eval "$OUT_DIR" answer_only "$TRAINED_DIRECT_PRED" 64 \
  2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_trained_direct_eval.log"
run_eval "$QTSA_CKPT" answer_only "$BASELINE_PRED" 64 \
  2>&1 | tee "$LOG_DIR/dvgbench_qtsa_i2e_${TAG}_qtsa_baseline_eval.log"

"$ENV_PY" train/src/tools/check_i2e_heldout_gate.py \
  --i2e-summary "${I2E_PRED%.jsonl}.summary.json" \
  --trained-direct-summary "${TRAINED_DIRECT_PRED%.jsonl}.summary.json" \
  --baseline-summary "${BASELINE_PRED%.jsonl}.summary.json" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output "$GATE_RESULT" \
  --max-regression 0.05
