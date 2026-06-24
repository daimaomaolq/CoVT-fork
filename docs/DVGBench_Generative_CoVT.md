# DVGBench Generative CoVT Protocol

This note fixes the benchmark-alignment route for DVGBench region-level visual
grounding. The main result must use the same implicit-query, generative bbox
protocol as the DVGBench paper baseline, not the standalone UAV adapter head.

## Positioning

- Baseline to cite: DVGBench paper `Qwen2.5-VL 7B` and `Qwen2.5-VL 7B SFT`.
- Our main model: the CoVT/Qwen Seg+DINO checkpoint, fine-tuned to output
  DVGBench bbox tokens. Keep two tracks separate: plain LoRA with
  `--anchor_model_id "[]"` is only a protocol sanity check, while Seg+DINO
  auxiliary LoRA with `--anchor_model_id "['sam','dino']"` is the actual
  innovation track.
- Adapter-head results are diagnostic only. They are useful for cached-token
  efficiency and fast failure analysis, but they are not comparable to the
  generative Qwen2.5-VL baseline table.

## Build SFT Data

Use `question`, not `question_e`, for the implicit-query protocol.

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

- train split: about 1691 rows when using `--validation-split 0.15`
- val split: about 299 rows
- final test eval index: 873 rows

## Fine-Tune CoVT/Qwen Plain LoRA Baseline

This is the plain protocol-alignment baseline. It does not use Seg/DINO auxiliary losses and should not be used as the final innovation claim.

```bash
export MODEL_PATH=/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886
if [ ! -d "$MODEL_PATH" ]; then export MODEL_PATH=Wakals/CoVT-7B-seg_depth_dino; fi

export OUT_DIR=/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527/checkpoints/dvgbench_generative_covt_question_lora_v1
mkdir -p "$OUT_DIR"

PYTHONPATH="$REPO/train/src" CUDA_VISIBLE_DEVICES=0 $ENV_PY -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "[]" \
  --data_path "$DVG_GEN_ROOT/dvg_train_question_sft.json" \
  --image_folder "$DVG_IMAGE_ROOT" \
  --output_dir "$OUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_steps 200 \
  --save_total_limit 2 \
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
  --report_to none
```

For two GPUs, use the environment's Python, not base `torchrun`:

```bash
PYTHONPATH="$REPO/train/src" CUDA_VISIBLE_DEVICES=0,1 $ENV_PY -m torch.distributed.run --nproc_per_node=2 \
  train/src/training/train.py \
  ...same arguments...
```

If DDP/NCCL fails, stay with single-GPU LoRA first. The dataset is small enough.


Observed plain-LoRA v1 result on DVGBench test:

- `mIoU`: 0.2850
- `Acc@0.5`: 0.2612
- `DVGBench_AVG`: 0.2800
- `parse_failed`: 0

This result confirms the generative bbox pipeline, but it does not validate the Seg/DINO contribution because `--anchor_model_id "[]"` disables anchor supervision.

## Fine-Tune Seg+DINO CoVT/Qwen

This is the innovation track. Training enables SAM and DINO auxiliary reconstruction through CoVT anchor tokens while preserving the same DVGBench generative bbox target.

Before launching, verify that the SAM checkpoint exists and that DINO can be loaded through the server's cached/mirrored torch hub setup:

```bash
export REPO=/root/autodl-tmp/CoVT-fork_v8_3
export ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python
cd "$REPO"

ls -lh train/src/anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth
PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
$ENV_PY - <<'PY'
import torch
print('torch', torch.__version__)
print('hub dir', torch.hub.get_dir())
PY
```

Use the same SFT JSON built above with `--mode answer_only`. The CoVT dataloader wraps responses with Seg/DINO anchor reasoning when `stage_0/1/2` are all zero and `anchor_model_id` is non-empty.

```bash
export MODEL_PATH=/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886
if [ ! -d "$MODEL_PATH" ]; then export MODEL_PATH=Wakals/CoVT-7B-seg_depth_dino; fi

export OUT_DIR=$RUN_ROOT/checkpoints/dvgbench_generative_covt_segdino_question_lora_v1
mkdir -p "$OUT_DIR" "$LOG_ROOT"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES=0 $ENV_PY -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "['sam','dino']" \
  --data_path "$DVG_GEN_ROOT/dvg_train_question_sft.json" \
  --image_folder "$DVG_IMAGE_ROOT" \
  --output_dir "$OUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --projection_layer_lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_steps 200 \
  --save_total_limit 2 \
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
  2>&1 | tee "$LOG_ROOT/train_dvgbench_generative_covt_segdino_question_lora_v1.log"
```

Expected signs that Seg/DINO is active:

- training log prints anchor model loading instead of using `anchor_model_id []`;
- trainable parameters include `sam_projection`, `sam_cross_attention`, `sam_query_vectors`, `dino_projection`, `dino_cross_attention`, `dino_query_vectors`;
- logs contain `seg_loss` and `dino_loss` values during forward passes.

## Final Test Eval For Seg+DINO

```bash
export PRED=$RUN_ROOT/predictions/dvgbench_generative_covt_segdino_question_lora_v1.jsonl

CUDA_VISIBLE_DEVICES=0 $ENV_PY train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$DVG_GEN_ROOT/dvg_test_question_eval.jsonl" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OUT_DIR" \
  --output "$PRED" \
  --prompt-mode answer_only \
  --max-new-tokens 64 \
  2>&1 | tee "$LOG_ROOT/eval_dvgbench_generative_covt_segdino_question_lora_v1.log"
```

Compare this result against the plain-LoRA v1 result above. Only the Seg+DINO track supports the paper claim that structural segmentation and DINO priors improve UAV implicit grounding.
## Final Test Eval For Plain LoRA

Use this only to reproduce the plain-LoRA v1 sanity check. It is not the Seg+DINO innovation result.

```bash
export PRED=/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527/predictions/dvgbench_generative_covt_question_lora_v1.jsonl

CUDA_VISIBLE_DEVICES=0 $ENV_PY train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$DVG_GEN_ROOT/dvg_test_question_eval.jsonl" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OUT_DIR" \
  --output "$PRED" \
  --prompt-mode answer_only \
  --max-new-tokens 64 \
  2>&1 | tee "$OUT_DIR/eval_dvgbench_generative_test.log"
```

The script reports:

- `Acc@0.5`
- six-class macro `DVGBench_AVG`
- per-class `Acc@0.5`
- parse failures

## Review Rules

- Do not use `question_e` for the main DVGBench baseline comparison.
- Do not tune on `dvg_test.jsonl`.
- Do not report the adapter-head score as a direct competitor to the DVGBench
  Qwen2.5-VL baseline.
- It is acceptable to cite DVGBench paper baselines and run only our CoVT variant,
  as long as protocol, query field, and metric match.
