#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/CoVT-fork_v8_3}"
ENV_PY="${ENV_PY:-/root/autodl-tmp/envs/covt-v8-py310/bin/python}"
DVG_ROOT="${DVG_ROOT:-/root/autodl-tmp/datasets/DVGBench}"
DVG_IMAGE_ROOT="${DVG_IMAGE_ROOT:-$DVG_ROOT/images}"
DVG_GEN_ROOT="${DVG_GEN_ROOT:-$DVG_ROOT/generative_qwen_i2e}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886}"
QTSA_ROOT="${QTSA_ROOT:-$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1}"
RUN_TAG="${RUN_TAG:-v4_schema_state_preserved}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/checkpoints/dvgbench_qtsa_i2e_sft_$RUN_TAG}"
PRED_DIR="${PRED_DIR:-$RUN_ROOT/predictions}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
GPU_ID="${GPU_ID:-0}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-200704}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-802816}"

mkdir -p "$DVG_GEN_ROOT" "$OUT_DIR" "$PRED_DIR" "$LOG_ROOT"
cd "$REPO"

if [[ ! -x "$ENV_PY" ]]; then
  echo "Python environment not found: $ENV_PY" >&2
  exit 2
fi
if [[ ! -f "$DVG_ROOT/dvg_train.jsonl" || ! -f "$DVG_ROOT/dvg_test.jsonl" ]]; then
  echo "DVGBench train/test JSONL files are missing under $DVG_ROOT" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  MODEL_PATH="Wakals/CoVT-7B-seg_depth_dino"
fi

QTSA_CKPT="$QTSA_ROOT"
if [[ ! -f "$QTSA_CKPT/adapter_model.safetensors" ]]; then
  QTSA_CKPT="$(find "$QTSA_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "$QTSA_CKPT" || ! -f "$QTSA_CKPT/adapter_model.safetensors" ]]; then
  echo "Best QTSA LoRA checkpoint not found under $QTSA_ROOT" >&2
  exit 2
fi

TRAIN_JSON="$DVG_GEN_ROOT/dvg_train_question_i2e_three_task.json"
TEST_INDEX="$DVG_GEN_ROOT/dvg_test_question_oracle_free_eval.jsonl"
TRAIN_LOG="$LOG_ROOT/train_dvgbench_qtsa_i2e_sft_$RUN_TAG.log"
EVAL_LOG="$LOG_ROOT/eval_dvgbench_qtsa_i2e_sft_$RUN_TAG.log"
PRED="$PRED_DIR/dvgbench_qtsa_i2e_sft_$RUN_TAG.jsonl"
SUMMARY="$PRED_DIR/dvgbench_qtsa_i2e_sft_$RUN_TAG.summary.json"

"$ENV_PY" train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_train.jsonl" \
  --image-root "$DVG_IMAGE_ROOT" \
  --image-folder "$DVG_IMAGE_ROOT" \
  --query-field question \
  --explicit-field question_e \
  --mode i2e \
  --i2e-answer-only-copy-ratio 1.0 \
  --i2e-explicit-only-copy-ratio 1.0 \
  --shuffle \
  --seed 20260727 \
  --output "$TRAIN_JSON"

"$ENV_PY" train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_test.jsonl" \
  --image-root "$DVG_IMAGE_ROOT" \
  --image-folder "$DVG_IMAGE_ROOT" \
  --query-field question \
  --mode answer_only \
  --omit-oracle-fields-from-eval-index \
  --output "$DVG_GEN_ROOT/dvg_test_unused_sft.json" \
  --write-eval-index "$TEST_INDEX"

"$ENV_PY" - "$TRAIN_JSON" "$TEST_INDEX" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
test = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()]
assert sum(row["metadata"]["protocol"] == "i2e" for row in train) == 1990
assert sum(row["metadata"]["protocol"] == "answer_only_preservation" for row in train) == 1990
assert sum(row["metadata"]["protocol"] == "implicit_to_explicit_auxiliary" for row in train) == 1990
assert len(train) == 5970
assert len(test) == 873
assert len({row["sample_id"] for row in test}) == 873
assert all("question_e" not in row and "question_e_cn" not in row for row in test)
assert all(row["oracle_fields_present"] is False for row in test)
print({"train_rows": len(train), "test_rows": len(test), "oracle_free": True})
PY

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_PY" -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "['sam','dino']" \
  --load_anchor_teachers False \
  --anchor_loss_weight "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" \
  --anchor_token_counts "[8,4,4,4,4,4,4,4]" \
  --data_path "$TRAIN_JSON" \
  --image_folder "$DVG_IMAGE_ROOT" \
  --image_min_pixels "$IMAGE_MIN_PIXELS" \
  --image_max_pixels "$IMAGE_MAX_PIXELS" \
  --output_dir "$OUT_DIR" \
  --lora_weight_path "$QTSA_CKPT" \
  --anchor_prompt_mode query_tail \
  --anchor_response_mode none \
  --train_anchor_adapters False \
  --compact_non_lora_checkpoint True \
  --i2e_answer_token_weight 5.0 \
  --i2e_format_token_weight 8.0 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
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
  2>&1 | tee "$TRAIN_LOG"

I2E_CKPT="$OUT_DIR"
if [[ ! -f "$I2E_CKPT/adapter_model.safetensors" ]]; then
  I2E_CKPT="$(find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "$I2E_CKPT" || ! -f "$I2E_CKPT/adapter_model.safetensors" ]]; then
  echo "I2E checkpoint was not produced." >&2
  exit 3
fi

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_PY" train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$TEST_INDEX" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$I2E_CKPT" \
  --output "$PRED" \
  --summary-output "$SUMMARY" \
  --image-min-pixels "$IMAGE_MIN_PIXELS" \
  --image-max-pixels "$IMAGE_MAX_PIXELS" \
  --query-field query \
  --prompt-mode i2e \
  --i2e-schema-guard \
  --require-oracle-free-index \
  --anchor-model-id "['sam','dino']" \
  --anchor-prompt-mode query_tail \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --max-new-tokens 192 \
  --temperature 0 \
  --batch-size 1 \
  2>&1 | tee "$EVAL_LOG"

"$ENV_PY" - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
baselines = {
    "matched_fixed_loader": {
        "mIoU": 0.32589248236596774,
        "Acc@0.5": 0.3230240549828179,
        "DVGBench_AVG": 0.35870769347166687,
    },
    "old_full_resolution_formal": {
        "mIoU": 0.33379627010060053,
        "Acc@0.5": 0.32989690721649484,
        "DVGBench_AVG": 0.36270966618744965,
    },
}
metrics = tuple(baselines["matched_fixed_loader"])
comparison = {
    "baselines": baselines,
    "i2e": {key: summary[key] for key in metrics},
    "delta": {
        name: {key: summary[key] - values[key] for key in metrics}
        for name, values in baselines.items()
    },
    "raw_schema_format_rate": summary.get("raw_schema_format_rate"),
    "guarded_schema_format_rate": summary.get("schema_format_rate"),
    "schema_guard_applied": summary.get("schema_guard_applied"),
    "i2e_improves_matched_acc50": (
        summary["Acc@0.5"] > baselines["matched_fixed_loader"]["Acc@0.5"]
    ),
    "i2e_improves_matched_macro": (
        summary["DVGBench_AVG"]
        > baselines["matched_fixed_loader"]["DVGBench_AVG"]
    ),
}
comparison_path = Path(sys.argv[1]).with_name(
    Path(sys.argv[1]).name.replace(".summary.json", ".comparison.json")
)
comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
print(json.dumps(comparison, indent=2))
PY
