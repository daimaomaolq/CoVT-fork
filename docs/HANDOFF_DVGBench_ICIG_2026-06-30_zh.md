# DVGBench / ICIG 论文实验交接 2026-06-30

## 0. 当前一句话结论

当前主线不是去复现或挑战 DVGBench 论文官方排行榜，而是做一个可控的 local protocol：

```text
同一 CoVT-7B-seg_depth_dino backbone
+ 同一 DVGBench implicit-query 数据
+ 同一 generative bbox SFT/eval pipeline
对比 plain SFT、DINO-only、SAM-only、SAM+DINO query-aligned anchors。
```

最稳妥的论文叙事是：

> 在 UAV 隐式 grounding 中，单纯对 CoVT/Qwen backbone 做生成式 SFT 已经能工作，但没有显式利用区域级结构与语义专家信号。我们加入 query-aligned SAM/DINO visual anchor tokens，并在同一训练与评测协议下提升 implicit UAV grounding。

不要写成：

```text
We outperform the official DVGBench Qwen2.5-VL-7B SFT baseline.
```

因为当前 protocol 和 DVGBench 论文官方 SFT/GRPO 设置不完全一致。可以引用 DVGBench 作为数据集与任务来源，但主表 baseline 应该是本地 plain CoVT SFT。

## 1. 当前数据和协议状态

服务器路径：

```text
REPO=/root/autodl-tmp/CoVT-fork_v8_3
ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python
DVG_ROOT=/root/autodl-tmp/datasets/DVGBench
DVG_IMAGE_ROOT=/root/autodl-tmp/datasets/DVGBench/images
DVG_GEN_ROOT=/root/autodl-tmp/datasets/DVGBench/generative_qwen
RUN_ROOT=/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527
LOG_ROOT=$RUN_ROOT/logs
```

已确认数据：

```text
dvg_train.jsonl: 1990 rows
dvg_test.jsonl: 873 rows
images/era: 759 valid images
images/visdrone: 1483 valid images
```

生成式 SFT/eval 文件：

```text
$DVG_GEN_ROOT/dvg_train_question_sft.json
$DVG_GEN_ROOT/dvg_test_question_eval.jsonl
```

关键语义：

- 当前训练和测试都使用 `question` 字段。
- `question` 是隐式 query，例如需要根据描述、关系、上下文定位目标。
- `question_e` 是更显式的短定位语句，不是当前主线。
- 所以当前实验已经是 implicit-query setting，不需要另开一套 implicit 实验。

已抽样验证：

```text
GEN query: A person in black sitting in a toy car
RAW question: A person in black sitting in a toy car
RAW question_e: A person in black
```

## 2. 当前代码状态

本地仓库：

```text
F:\research\CoVT-fork
```

服务器不是 git 仓库，仍然通过 GitHub raw 镜像拉单文件。

已推送的关键提交：

```text
aa47edc Expose anchor loss weights for CoVT training
ecf157e Add query-tail SegDINO DVGBench protocol
2c15e03 Add LoRA warm-start for DVGBench SegDINO
df93f89 Support configurable anchor token counts
4daa4ff Honor anchor loss weights
```

关键文件：

```text
train/src/training/params.py
train/src/training/data.py
train/src/training/train.py
train/src/training/covt_qwen2_5_vl.py
train/src/tools/eval_dvgbench_generative_grounding.py
```

服务器拉取最新文件命令：

```bash
export REPO=/root/autodl-tmp/CoVT-fork_v8_3
export ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python
cd "$REPO"

for file in \
  train/src/training/params.py \
  train/src/training/data.py \
  train/src/training/train.py \
  train/src/training/covt_qwen2_5_vl.py \
  train/src/tools/eval_dvgbench_generative_grounding.py
do
  mkdir -p "$(dirname "$file")"
  curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 \
    -o "$file" \
    "https://ghproxy.net/https://raw.githubusercontent.com/daimaomaolq/CoVT-fork/main/$file"
done

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
$ENV_PY -m py_compile \
  train/src/training/params.py \
  train/src/training/data.py \
  train/src/training/train.py \
  train/src/training/covt_qwen2_5_vl.py \
  train/src/tools/eval_dvgbench_generative_grounding.py
```

## 3. 已完成实验结果

### 3.1 Plain CoVT generative SFT baseline

设置：

```text
Backbone: Wakals/CoVT-7B-seg_depth_dino
Training data: dvg_train_question_sft.json
Eval data: dvg_test_question_eval.jsonl
Prompt: answer_only
No query-aligned anchor insertion
LoRA rank 64
3 epochs
```

结果：

```text
samples: 873
mIoU: 0.28503611294142367
Acc@0.5: 0.2611683848797251
DVGBench_AVG: 0.2800414737248296
parse_failed: 0
predictions:
/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527/predictions/dvgbench_generative_covt_question_lora_v1_projlr.jsonl
```

### 3.2 SAM+DINO query-tail warm-start main model

设置：

```text
Backbone: same as baseline
Warm-start: plain CoVT LoRA
anchor_model_id: ['sam','dino']
anchor_prompt_mode: query_tail
SAM tokens: 8 fixed
DINO tokens: 4 in best current setting
3 epochs
```

结果：

```text
samples: 873
mIoU: 0.33379627010060053
Acc@0.5: 0.32989690721649484
DVGBench_AVG: 0.36270966618744965
parse_failed: 0
predictions:
/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527/predictions/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v1.jsonl
```

Per-class Acc@0.5:

```text
disaster: 0.4716981132075472
productive: 0.5206611570247934
security: 0.47619047619047616
social: 0.2967032967032967
sport: 0.21656050955414013
traffic: 0.19444444444444445
```

### 3.3 SAM8 + DINO8 high-token sensitivity

结果：

```text
mIoU: 0.33362411544594694
Acc@0.5: 0.32531500572737687
DVGBench_AVG: 0.3550456107255511
predictions:
/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527/predictions/dvgbench_generative_covt_segdino_querytail_warmstart_lora_v2_tok8_8.jsonl
```

结论：

- DINO token 从 4 提到 8 没有提升主指标。
- 当前主线继续使用 SAM8 + DINO4。
- 不要尝试 SAM16：当前 SAM branch 内部固定输出 8 个 mask/token embedding，强行设 16 会触发 reshape 错误。

### 3.4 DINO-only ablation

结果：

```text
samples: 873
mIoU: 0.33636360258845777
Acc@0.5: 0.3230240549828179
DVGBench_AVG: 0.34879924385614397
parse_failed: 0
predictions:
/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527/predictions/dvgbench_ablate_dino_only_querytail_warmstart_lora_v1.jsonl
```

Per-class Acc@0.5:

```text
disaster: 0.4339622641509434
productive: 0.5289256198347108
security: 0.42857142857142855
social: 0.3076923076923077
sport: 0.21656050955414013
traffic: 0.17083333333333334
```

解释口径：

- DINO-only 的 mIoU 略高，说明 semantic region cue 对平均重叠有帮助。
- SAM+DINO 的 Acc@0.5 和 DVGBench_AVG 更高，说明 SAM 的 object-mask prior 能帮助更多样本跨过 0.5 阈值。
- 主表优先按 DVGBench_AVG 或 Acc@0.5 排，SAM+DINO 是主线最佳。

## 4. 当前必须补的实验

只差一个核心 ablation：

```text
SAM-only query-tail warm-start LoRA
```

跑完它，主表就完整：

```text
Plain CoVT SFT
DINO-only
SAM-only
SAM+DINO
```

## 5. SAM-only 训练命令

```bash
export REPO=/root/autodl-tmp/CoVT-fork_v8_3
export ENV_PY=/root/autodl-tmp/envs/covt-v8-py310/bin/python

export DVG_ROOT=/root/autodl-tmp/datasets/DVGBench
export DVG_IMAGE_ROOT=$DVG_ROOT/images
export DVG_GEN_ROOT=$DVG_ROOT/generative_qwen
export RUN_ROOT=/root/autodl-tmp/outputs/covt_uav_refpg_v8_cleanenv_20260527
export LOG_ROOT=$RUN_ROOT/logs
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/predictions" "$LOG_ROOT"

export MODEL_PATH=/root/autodl-tmp/hf_cache/hub/models--Wakals--CoVT-7B-seg_depth_dino/snapshots/154b974eb0d071160a4bc5b287f242bc2875b886
if [ ! -d "$MODEL_PATH" ]; then export MODEL_PATH=Wakals/CoVT-7B-seg_depth_dino; fi

export PLAIN_ROOT=$RUN_ROOT/checkpoints/dvgbench_generative_covt_question_lora_v1_projlr
export PLAIN_WARMSTART=$PLAIN_ROOT
if [ ! -f "$PLAIN_WARMSTART/adapter_model.safetensors" ]; then
  export PLAIN_WARMSTART=$(ls -d "$PLAIN_ROOT"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
fi

export OUT_DIR=$RUN_ROOT/checkpoints/dvgbench_ablate_sam_only_querytail_warmstart_lora_v1
cd "$REPO"

PYTHONPATH="$REPO/train:$REPO/train/src:$REPO/train/src/anchors" \
CUDA_VISIBLE_DEVICES=0 $ENV_PY -m training.train \
  --model_id "$MODEL_PATH" \
  --model_path "$MODEL_PATH" \
  --anchor_model_id "['sam']" \
  --anchor_loss_weight "[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" \
  --anchor_token_counts "[8,4,4,4,4,4,4,4]" \
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
  2>&1 | tee "$LOG_ROOT/train_dvgbench_ablate_sam_only_querytail_warmstart_lora_v1.log"
```

## 6. SAM-only 评测命令

```bash
export CKPT=$OUT_DIR
if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
  export CKPT=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
fi

export PRED=$RUN_ROOT/predictions/dvgbench_ablate_sam_only_querytail_warmstart_lora_v1.jsonl

CUDA_VISIBLE_DEVICES=0 $ENV_PY train/src/tools/eval_dvgbench_generative_grounding.py \
  --index "$DVG_GEN_ROOT/dvg_test_question_eval.jsonl" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$CKPT" \
  --output "$PRED" \
  --prompt-mode answer_only \
  --anchor-model-id "['sam']" \
  --anchor-prompt-mode query_tail \
  --anchor-token-counts "[8,4,4,4,4,4,4,4]" \
  --max-new-tokens 64 \
  --batch-size 1 \
  2>&1 | tee "$LOG_ROOT/eval_dvgbench_ablate_sam_only_querytail_warmstart_lora_v1.log"
```

## 7. 参数解释

当前参数不是随便设的：

- `num_train_epochs=3`：1990 train samples，global batch 16，约 315 optimizer steps。对 LoRA + frozen LLM/vision tower 是合理的快速收敛设置。
- `per_device_train_batch_size=1`：CoVT + SAM/DINO 显存压力大，单卡稳定。
- `gradient_accumulation_steps=16`：有效 batch size 16。
- `learning_rate=2e-5`：LoRA 微调常用稳定范围。
- `projection_layer_lr=2e-5`：避免 projection 过快漂移。
- `lora_rank=64, alpha=128`：给小数据任务足够适配容量。
- `freeze_llm=True, freeze_vision_tower=True`：降低过拟合和显存风险，论文叙事也更清晰：改的是 anchor adaptation，不是全模型重训。
- `anchor_prompt_mode=query_tail`：将 anchor token 插到 query 后面，避免破坏视觉前缀和原始 Qwen/CoVT 的图像理解路径。
- `lora_weight_path=$PLAIN_WARMSTART`：先学会 bbox 生成格式，再加 anchor signal，减少 anchor 冷启动干扰。

## 8. 已知坑和不要再踩的点

1. 服务器不是 git 仓库，不要 `git pull`。
2. 如果 GitHub raw 拉不动，继续用 `ghproxy.net`，不要换成不可解析镜像。
3. `SAM token count` 只能用 8，不能设 16。
4. `question_e` 不是当前主线，别混进主表。
5. 旧 adapter-head 指标不能和生成式 DVGBench 指标混表。
6. `anchor_loss_weight` 在 `4daa4ff` 前没有真正生效。已完成结果仍可用，但写作时不强调权重数值，强调 query-tail warm-start + anchor composition。
7. 若磁盘满导致 `PytorchStreamReader failed reading zip archive`，先删坏 checkpoint 或扩容后重跑 eval。

## 9. 给论文线程的最小实验表

当前可先写：

| Method | Anchors | mIoU | Acc@0.5 | DVGBench_AVG |
|---|---:|---:|---:|---:|
| Plain CoVT SFT | none | 0.2850 | 0.2612 | 0.2800 |
| DINO-only | DINO | 0.3364 | 0.3230 | 0.3488 |
| SAM-only | SAM | TODO | TODO | TODO |
| Ours | SAM+DINO | 0.3338 | 0.3299 | 0.3627 |

写法：

- 主指标用 `DVGBench_AVG` 和 `Acc@0.5`。
- `mIoU` 作为辅助连续指标。
- 如果 SAM-only 结果比预期复杂，只写 complementary analysis，不要改主结论。

## 10. 论文题目候选

1. Query-Aligned Visual Anchor Tokens for Implicit UAV Visual Grounding
2. Enhancing CoVT for Implicit UAV Grounding with SAM and DINO Visual Anchors
3. Structure- and Semantics-Aware Anchor Adaptation for UAV Visual Grounding

推荐第 1 个，最贴当前实验。

