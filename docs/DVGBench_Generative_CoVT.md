# DVGBench Generative CoVT Warm-Start Query-Tail Protocol

This is the current handoff for the DVGBench paper-aligned experiment. The final
comparison is DVGBench region-level implicit-query visual grounding, evaluated by
generative bbox output and `Acc@0.5`, matching the DVGBench baseline table.

## Current Decision

Use DVGBench train only for SFT and DVGBench test only for final evaluation. Do
not tune on `dvg_test.jsonl`.

The main claim path is:

1. Train or reuse a plain generative LoRA checkpoint that already learns the
   DVGBench bbox-output format.
2. Warm-start the CoVT/Qwen Seg+DINO query-tail run from that plain LoRA.
3. Enable real Seg+DINO anchor tokens in both training and evaluation.

This keeps the useful bbox formatting ability from SFT and then tests whether
query-conditioned Seg+DINO visual anchor tokens improve the result.

## Why This Replaces The Old Seg+DINO Runs

Observed results on DVGBench test:

- Plain generative LoRA, no anchors: `mIoU 0.2850`, `Acc@0.5 0.2612`, `DVGBench_AVG 0.2800`.
- Legacy Seg+DINO auxiliary-only runs were lower, around `Acc@0.5 0.239-0.243`.

The old Seg+DINO path was not a fair test of the proposed mechanism:

1. Legacy CoVT inserted anchor tokens immediately after `<|vision_end|>`, before
   the natural-language query. In a causal decoder, those anchor hidden states
   cannot attend to the later query.
2. Training and evaluation were mismatched. Training used legacy random stage
   logic or auxiliary anchor responses, while evaluation used a bbox-only prompt
   without matching anchor tokens.
3. On only 1,990 DVGBench train rows, auxiliary reconstruction can act as noise
   unless the final answer path explicitly consumes the query-conditioned anchors.
4. A cold Seg+DINO run has to learn both bbox formatting and anchor usage at the
   same time. Warm-starting from the plain LoRA removes that avoidable burden.

The fix is `query_tail` plus LoRA warm-start: put Seg+DINO anchor tokens after
query text and before `<|im_end|>`, keep the assistant target bbox-only, and load
plain LoRA weights before continuing Seg+DINO training.

## Code Files To Pull On Server

The AutoDL server directory is usually not a git repo. Pull raw files with a
GitHub mirror.

```bash
export REPO=/root/autodl-tmp/CoVT-fork_v8_3
export ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python
cd "$REPO"

for file in \
  train/src/training/train.py \
  train/src/training/data.py \
  train/src/training/params.py \
  train/src/tools/eval_dvgbench_generative_grounding.py \
  docs/DVGBench_Generative_CoVT.md
do
  mkdir -p "$(dirname "$file")"
  curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 \
    -o "$file" \
    "https://ghproxy.net/https://raw.githubusercontent.com/daimaomaolq/CoVT-fork/main/$file"
done

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
$ENV_PY -m py_compile \
  train/src/training/train.py \
  train/src/training/data.py \
  train/src/training/params.py \
  train/src/tools/eval_dvgbench_generative_grounding.py
```

## Build SFT Data

Use `question`, not `question_e`, for the DVGBench paper-aligned implicit-query
protocol.

```bash
export ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python
export REPO=/root/autodl-tmp/CoVT-fork_v8_3
export DVG_ROOT=/root/autodl-tmp/datasets/DVGBench
export DVG_IMAGE_ROOT=$DVG_ROOT/images
export DVG_GEN_ROOT=$DVG_ROOT/generative_qwen

mkdir -p "$DVG_GEN_ROOT"
cd "$REPO"

$ENV_PY train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_train.jsonl" \
  --image-root "$DVG_IMAGE_ROOT" \
  --image-folder "$DVG_IMAGE_ROOT" \
  --query-field question \
  --mode answer_only \
  --shuffle \
  --validation-split 0.15 \
  --output "$DVG_GEN_ROOT/dvg_train_question_sft.json" \
  --val-output "$DVG_GEN_ROOT/dvg_val_question_sft.json" \
  --write-eval-index "$DVG_GEN_ROOT/dvg_train_question_eval.jsonl"

$ENV_PY train/src/tools/build_dvgbench_generative_sft.py \
  --input-jsonl "$DVG_ROOT/dvg_test.jsonl" \
  --image-root "$DVG_IMAGE_ROOT" \
  --image-folder "$DVG_IMAGE_ROOT" \
  --query-field question \
  --mode answer_only \
  --output "$DVG_GEN_ROOT/dvg_test_question_sft.json" \
  --write-eval-index "$DVG_GEN_ROOT/dvg_test_question_eval.jsonl"
```

Expected row counts:

- Train SFT JSON: about 1,691 rows after `--validation-split 0.15`.
- Held-out train validation JSON: about 299 rows.
- Final test eval index: 873 rows.

## Required Checkpoints

```bash
export MODEL_PATH=/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886
if [ ! -d "$MODEL_PATH" ]; then export MODEL_PATH=Wakals/CoVT-7B-seg_depth_dino; fi

export SAM_DIR=$REPO/train/src/anchors/segment_anything/ckpt
mkdir -p "$SAM_DIR"
if [ ! -f "$SAM_DIR/sam_vit_h_4b8939.pth" ]; then
  mkdir -p /root/autodl-fs/models
  wget -c --tries=10 --timeout=30 \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /root/autodl-fs/models/sam_vit_h_4b8939.pth
  ln -sf /root/autodl-fs/models/sam_vit_h_4b8939.pth "$SAM_DIR/sam_vit_h_4b8939.pth"
fi
ls -lh "$SAM_DIR/sam_vit_h_4b8939.pth"
```

## Warm-Start Source

Use the best plain generative LoRA as the warm-start. This should be a directory
containing `adapter_model.safetensors`; if the root only has checkpoints, choose
the latest checkpoint.

```bash
export RUN_ROOT=/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527
export PLAIN_ROOT=$RUN_ROOT/checkpoints/dvgbench_generative_covt_question_lora_v1_projlr
export PLAIN_WARMSTART=$PLAIN_ROOT
if [ ! -f "$PLAIN_WARMSTART/adapter_model.safetensors" ]; then
  export PLAIN_WARMSTART=$(ls -d "$PLAIN_ROOT"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
fi

echo "PLAIN_WARMSTART=$PLAIN_WARMSTART"
ls -lh "$PLAIN_WARMSTART"/adapter_model.safetensors
ls -lh "$PLAIN_WARMSTART"/non_lora_state_dict.bin 2>/dev/null || true
```

If this directory is missing, run the plain generative LoRA first, or omit
`--lora_weight_path` only as a diagnostic fallback. The final paper run should
use warm-start.

## Final Warm-Started Seg+DINO Query-Tail Training

Use one GPU for the first clean run. The dataset is small, and the SAM forward is
the bottleneck; one GPU avoids DDP/SAM edge cases while preserving the exact
mechanism.

```bash
export REPO=/root/autodl-tmp/CoVT-fork_v8_3
export ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python
export DVG_ROOT=/root/autodl-tmp/datasets/DVGBench
export DVG_IMAGE_ROOT=$DVG_ROOT/images
export DVG_GEN_ROOT=$DVG_ROOT/generative_qwen
export RUN_ROOT=/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527
export LOG_ROOT=$RUN_ROOT/logs
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export TORCH_HOME=/root/.cache/torch
export OUT_DIR=$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1
mkdir -p "$OUT_DIR" "$LOG_ROOT"
cd "$REPO"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES=0 $ENV_PY -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "['sam','dino']" \
  --anchor_loss_weight "[0.05, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --data_path "$DVG_GEN_ROOT/dvg_train_question_sft.json" \
  --image_folder "$DVG_IMAGE_ROOT" \
  --output_dir "$OUT_DIR" \
  --lora_weight_path "$PLAIN_WARMSTART" \
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
  --use_liger False \
  --disable_flash_attn2 True \
  --report_to none \
  2>&1 | tee "$LOG_ROOT/train_dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1.log"
```

Why these hyperparameters:

- `--lora_weight_path`: starts from the plain LoRA that already learned DVGBench
  bbox syntax and implicit-query formatting.
- `anchor_prompt_mode query_tail`: makes the anchor tokens query-conditioned.
- `anchor_response_mode none`: keeps the output target bbox-only, matching the
  DVGBench metric and evaluation prompt.
- `anchor_loss_weight [0.05, 0.2, ...]`: keeps SAM as light structure
  regularization and DINO as the stronger semantic auxiliary signal, without
  letting reconstruction dominate 1,990 SFT rows.
- `freeze_llm True` and `freeze_vision_tower True`: preserves the CoVT/Qwen base
  and tests the extra visual-token mechanism through LoRA and projection layers.
- `save_steps 10000`: avoids partial checkpoints filling `/root/autodl-tmp`; use
  the final output directory, or the latest checkpoint if the trainer creates one.

The training script loads PEFT LoRA weights with `PeftModel.from_pretrained(...,
is_trainable=True)` and then shape-safely loads `non_lora_state_dict.bin` if it
exists. Mismatched keys are skipped and reported instead of crashing.

## Final Test Eval For Warm-Started Seg+DINO Query-Tail

```bash
export SEG_DINO_ROOT=$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1
export SEG_DINO_CKPT=$SEG_DINO_ROOT
if [ ! -f "$SEG_DINO_CKPT/adapter_model.safetensors" ]; then
  export SEG_DINO_CKPT=$(ls -d "$SEG_DINO_ROOT"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
fi

echo "SEG_DINO_CKPT=$SEG_DINO_CKPT"
ls -lh "$SEG_DINO_CKPT"/adapter_model.safetensors "$SEG_DINO_CKPT"/tokenizer_config.json

mkdir -p "$RUN_ROOT/predictions"
export PRED=$RUN_ROOT/predictions/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1.jsonl

CUDA_VISIBLE_DEVICES=0 $ENV_PY train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$DVG_GEN_ROOT/dvg_test_question_eval.jsonl" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$SEG_DINO_CKPT" \
  --output "$PRED" \
  --prompt-mode answer_only \
  --anchor-model-id "['sam','dino']" \
  --anchor-prompt-mode query_tail \
  --max-new-tokens 64 \
  --batch-size 1 \
  2>&1 | tee "$LOG_ROOT/eval_dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1.log"
```

The success condition is not just that `seg_loss` and `dino_loss` print during
training. Evaluation must also pass `--anchor-model-id "['sam','dino']"` and
`--anchor-prompt-mode query_tail`; otherwise the anchor path is disabled at
inference.

## Decision Rule

Use this run for the paper only if it improves over the plain generative LoRA
sanity result (`Acc@0.5 0.2612`). If it does not, do not claim the auxiliary
anchor mechanism improves DVGBench; use the plain LoRA as the main result and
keep Seg+DINO as an ablation/diagnostic until the anchor weighting or trainable
scope is revised.
