#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/CoVT-fork_dronevg_qtsa}"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/covt-v8-py310/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct}"
DRONE_ADAPTER="${DRONE_ADAPTER:-/root/autodl-tmp/models/DroneVG-R1-7B-peft010-converted-v2}"
DVG_ROOT="${DVG_ROOT:-/root/autodl-tmp/datasets/DVGBench}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527}"
TRAIN_DATA="${TRAIN_DATA:-$DVG_ROOT/generative_qwen/dvg_train_dronevg_qtsa_i2e_1692.json}"
TEST_INDEX="${TEST_INDEX:-$DVG_ROOT/generative_qwen/dvg_test_question_eval.jsonl}"
TAG="${TAG:-dvgbench_dronevg_r1_qtsa_samdino_v1}"
OUT_DIR="$RUN_ROOT/checkpoints/$TAG"
PRED="$RUN_ROOT/predictions/$TAG.jsonl"
TRAIN_LOG="$RUN_ROOT/logs/train_$TAG.log"
EVAL_LOG="$RUN_ROOT/logs/eval_$TAG.log"

case "$TAG" in
  dvgbench_dronevg_r1_qtsa_*) ;;
  *)
    echo "Refusing unsafe TAG outside the isolated DroneVG-QTSA namespace: $TAG" >&2
    exit 2
    ;;
esac
case "$OUT_DIR" in
  *dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1*)
    echo "Refusing to write into the existing QTSA checkpoint." >&2
    exit 2
    ;;
esac

for path in "$REPO" "$BASE_MODEL" "$DRONE_ADAPTER" "$TRAIN_DATA" "$TEST_INDEX"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 3
  fi
done
if [[ -e "$OUT_DIR" || -e "$PRED" ]]; then
  echo "Refusing to overwrite an existing isolated experiment: $OUT_DIR or $PRED" >&2
  exit 4
fi

mkdir -p "$OUT_DIR" "$RUN_ROOT/logs" "$RUN_ROOT/predictions"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TORCH_HOME=/root/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m training.train \
  --model_id "$BASE_MODEL" \
  --model_path "$BASE_MODEL" \
  --anchor_model_id "['sam','dino']" \
  --anchor_loss_weight "[0.05, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --anchor_token_counts "[8,4,4,4,4,4,4,4]" \
  --data_path "$TRAIN_DATA" \
  --image_folder "$DVG_ROOT/images" \
  --image_min_pixels 200704 \
  --image_max_pixels 401408 \
  --output_dir "$OUT_DIR" \
  --lora_weight_path "$DRONE_ADAPTER" \
  --freeze_warmstart_lora True \
  --freeze_token_embeddings True \
  --anchor_prompt_mode query_tail \
  --anchor_response_mode none \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --projection_layer_lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_steps 10000 \
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
  --use_liger False \
  --disable_flash_attn2 True \
  --report_to none \
  2>&1 | tee "$TRAIN_LOG"

if [[ ! -f "$OUT_DIR/adapter_model.safetensors" || ! -f "$OUT_DIR/non_lora_state_dict.bin" ]]; then
  echo "Training finished without the required adapter/non-LoRA checkpoint." >&2
  exit 5
fi

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$TEST_INDEX" \
  --model-path "$BASE_MODEL" \
  --adapter-path "$OUT_DIR" \
  --output "$PRED" \
  --prompt-mode official_i2e \
  --anchor-model-id "['sam','dino']" \
  --anchor-prompt-mode query_tail \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --max-new-tokens 96 \
  --batch-size 1 \
  --image-min-pixels 200704 \
  --image-max-pixels 401408 \
  2>&1 | tee "$EVAL_LOG"

