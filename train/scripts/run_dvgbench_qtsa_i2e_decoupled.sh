#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/CoVT-fork_i2e_6c766d5}"
ENV_PY="${ENV_PY:-/root/autodl-tmp/envs/covt-v8-py310/bin/python}"
DVG_ROOT="${DVG_ROOT:-/root/autodl-tmp/datasets/DVGBench}"
IMAGE_ROOT="${IMAGE_ROOT:-$DVG_ROOT/images}"
GEN_ROOT="${GEN_ROOT:-$DVG_ROOT/generative_qwen_i2e/decoupled_v1}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886}"
QTSA_ROOT="${QTSA_ROOT:-$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1}"
TAG="${TAG:-decoupled_i2e_bbox_only_v1}"
EXPLICITIZER_CKPT="$RUN_ROOT/checkpoints/dvgbench_qtsa_$TAG"
PRED_ROOT="$RUN_ROOT/predictions"
LOG_ROOT="$RUN_ROOT/logs"
TRAIN_JSON="$GEN_ROOT/train_explicitizer_plain.json"
TRAIN_MANIFEST="$GEN_ROOT/train_explicitizer_plain.manifest.json"
TEST_INDEX="$DVG_ROOT/generative_qwen_i2e/dvg_test_question_oracle_free_eval.jsonl"
EXPLICIT_SIDECAR="$PRED_ROOT/dvgbench_qtsa_$TAG.explicit.jsonl"
EXPLICIT_MANIFEST="$PRED_ROOT/dvgbench_qtsa_$TAG.explicit.manifest.json"
GENERATED_INDEX="$GEN_ROOT/test_generated_explicit.jsonl"
GENERATED_MANIFEST="$GEN_ROOT/test_generated_explicit.manifest.json"
FINAL_PRED="$PRED_ROOT/dvgbench_qtsa_$TAG.jsonl"
FINAL_SUMMARY="$PRED_ROOT/dvgbench_qtsa_$TAG.summary.json"
COMPARISON="$PRED_ROOT/dvgbench_qtsa_$TAG.comparison.json"
GPU_ID="${GPU_ID:-0}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-802816}"

mkdir -p "$GEN_ROOT" "$EXPLICITIZER_CKPT" "$PRED_ROOT" "$LOG_ROOT"
cd "$REPO"

QTSA_CKPT="$QTSA_ROOT"
if [[ ! -f "$QTSA_CKPT/adapter_model.safetensors" ]]; then
  QTSA_CKPT="$(find "$QTSA_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi
if [[ -z "$QTSA_CKPT" || ! -f "$QTSA_CKPT/adapter_model.safetensors" ]]; then
  echo "Original QTSA checkpoint not found under $QTSA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$TEST_INDEX" ]]; then
  echo "Oracle-free test index not found: $TEST_INDEX" >&2
  exit 2
fi

"$ENV_PY" train/src/tools/build_dvgbench_explicitizer_plain_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_train.jsonl" \
  --output "$TRAIN_JSON" \
  --manifest-output "$TRAIN_MANIFEST" \
  --image-root "$IMAGE_ROOT"

"$ENV_PY" - "$TRAIN_JSON" "$TEST_INDEX" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
test = [json.loads(x) for x in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if x.strip()]
assert len(train) == 1990
assert len(test) == 873 and len({str(x["sample_id"]) for x in test}) == 873
assert all("question_e" not in x and "question_e_cn" not in x for x in test)
assert all(x.get("oracle_fields_present") is False for x in test)
assert all("<answer>" not in x["conversations"][1]["value"] for x in train)
assert all("<explicit>" not in x["conversations"][1]["value"] for x in train)
print({"train": len(train), "test": len(test), "test_oracle_free": True})
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
  --image_folder "$IMAGE_ROOT" \
  --image_min_pixels "$MIN_PIXELS" \
  --image_max_pixels "$MAX_PIXELS" \
  --output_dir "$EXPLICITIZER_CKPT" \
  --lora_weight_path "$QTSA_CKPT" \
  --anchor_prompt_mode query_tail \
  --anchor_response_mode none \
  --train_anchor_adapters False \
  --compact_non_lora_checkpoint True \
  --i2e_answer_token_weight 1.0 \
  --i2e_format_token_weight 1.0 \
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
  2>&1 | tee "$LOG_ROOT/train_dvgbench_qtsa_$TAG.log"

if [[ ! -f "$EXPLICITIZER_CKPT/adapter_model.safetensors" ]]; then
  echo "Explicitizer checkpoint missing: $EXPLICITIZER_CKPT" >&2
  exit 3
fi

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_PY" train/src/tools/generate_dvgbench_explicit_queries.py \
  --index "$TEST_INDEX" \
  --image-root "$IMAGE_ROOT" \
  --output "$EXPLICIT_SIDECAR" \
  --manifest-output "$EXPLICIT_MANIFEST" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$EXPLICITIZER_CKPT" \
  --anchor-model-id "['sam','dino']" \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --anchor-prompt-mode query_tail \
  --image-min-pixels "$MIN_PIXELS" \
  --image-max-pixels "$MAX_PIXELS" \
  --max-new-tokens 64 \
  2>&1 | tee "$LOG_ROOT/explicit_dvgbench_qtsa_$TAG.log"

"$ENV_PY" train/src/tools/build_dvgbench_generated_explicit_index.py \
  --index "$TEST_INDEX" \
  --sidecar "$EXPLICIT_SIDECAR" \
  --output "$GENERATED_INDEX" \
  --manifest-output "$GENERATED_MANIFEST"

# The final localization stage is the untouched original QTSA. Its raw model
# response and the public prediction field contain only a bbox.
PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_PY" train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$GENERATED_INDEX" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$QTSA_CKPT" \
  --output "$FINAL_PRED" \
  --summary-output "$FINAL_SUMMARY" \
  --image-min-pixels "$MIN_PIXELS" \
  --image-max-pixels "$MAX_PIXELS" \
  --query-field query \
  --prompt-mode answer_only \
  --require-oracle-free-index \
  --anchor-model-id "['sam','dino']" \
  --anchor-prompt-mode query_tail \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --max-new-tokens 64 \
  --temperature 0 \
  --batch-size 1 \
  2>&1 | tee "$LOG_ROOT/eval_dvgbench_qtsa_$TAG.log"

"$ENV_PY" - "$FINAL_PRED" "$FINAL_SUMMARY" "$EXPLICIT_MANIFEST" "$COMPARISON" <<'PY'
import json
import re
import sys
from pathlib import Path

pred_path, summary_path, explicit_path, comparison_path = map(Path, sys.argv[1:])
rows = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
explicit = json.loads(explicit_path.read_text(encoding="utf-8"))
assert len(rows) == 873 and len({str(x["sample_id"]) for x in rows}) == 873
assert all(x["protocol"].get("question_e_used") is False for x in rows)
assert all(x["protocol"].get("gt_visible_during_inference") is False for x in rows)
raw_outputs = [str(x.get("raw_output", "")) for x in rows]
assert all("<explicit>" not in x and "<think>" not in x for x in raw_outputs)
assert all(re.search(r"\{?\s*<\s*-?\d", x) for x in raw_outputs)
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
    "protocol": "decoupled_i2e_intermediate_final_bbox_only",
    "rows": len(rows),
    "question_e_train_only": True,
    "question_e_used_at_inference": False,
    "final_raw_output_bbox_only": True,
    "explicitizer": explicit,
    "metrics": {k: summary[k] for k in keys},
    "baselines": baselines,
    "delta": {
        name: {k: summary[k] - values[k] for k in keys}
        for name, values in baselines.items()
    },
}
comparison_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
