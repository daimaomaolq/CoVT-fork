# QTSA + I2E-SFT：摘要优先实验协议

## 目标

在现有最佳 QTSA checkpoint 上继续一次高效 LoRA 微调，验证显式的
Implicit-to-Explicit（I2E）文本中间表示是否提升 DVGBench 隐式查询定位。
本实验先回答“是否涨点”，不运行 Agent、GRPO 或消融矩阵。

当前固定基线：

| Method | mIoU | Acc@0.5 | DVGBench_AVG |
|---|---:|---:|---:|
| QTSA one-pass | 0.333796 | 0.329897 | 0.362710 |

成功判据以 `Acc@0.5` 和 `DVGBench_AVG` 为主；mIoU 为辅。只有全量 873
条测试结果超过基线，摘要才写 I2E 带来提升。

## 模型与数据协议

I2E-SFT 输入仅包含图像和隐式查询：

```text
<image>
Locate the region described by: {question}
First convert the implicit request into one brief explicit visual description ...
```

监督输出：

```text
<explicit>{question_e}</explicit>
<answer>{<x1><y1><x2><y2>}</answer>
```

其中 `question_e` 只用于 1,990 条训练数据的监督。最终测试索引会物理删除
`question_e` 和 `question_e_cn`，推理脚本还会使用
`--require-oracle-free-index` 二次校验，因此测试阶段不存在显式查询 oracle。

为防止一轮 I2E 微调削弱原有 bbox 输出能力，训练数据额外加入固定随机种子
选择的 50% bbox-only 副本。副本与对应 I2E 样本按 source group 划分，若以后
启用验证集，不会跨 train/validation 泄漏。

## 高效微调设置

- 初始化：最佳 QTSA SAM8+DINO4 query-tail LoRA。
- 训练：1 epoch，LoRA rank 64，学习率 `5e-6`。
- 基座 LLM 与视觉塔冻结。
- 保留 SAM+DINO query-tail anchor token 路径。
- `vqa_only_stage=-1` 且 anchor loss 全零：不重复运行昂贵的 SAM/DINO
  teacher forward；已训练的 QTSA 投影与 LoRA 继续由 I2E 语言/定位损失更新。
- 单卡 BF16，batch size 1，gradient accumulation 16。

这不是 DroneVG-R1 的复现，也不使用 GRPO；论文中应称为
`QTSA with supervised implicit-to-explicit grounding` 或 `QTSA-I2E-SFT`。

## 一键执行

```bash
cd /root/autodl-tmp/CoVT-fork_v8_3
bash train/scripts/run_dvgbench_qtsa_i2e_sft.sh
```

可用环境变量覆盖：

```bash
GPU_ID=0 \
QTSA_ROOT=/path/to/best/qtsa/adapter \
OUT_DIR=/path/to/new/i2e/adapter \
bash train/scripts/run_dvgbench_qtsa_i2e_sft.sh
```

流程依次执行：

1. 构建 I2E + 50% bbox-only 训练数据；
2. 构建不含任何 `question_e` 字段的 873 条测试索引；
3. 校验 1,990 个训练 source id、873 个唯一测试 id 和 oracle-free 条件；
4. 从最佳 QTSA LoRA warm-start 训练一轮；
5. 以 I2E prompt 完成 873 条全量推理；
6. 写出 summary 与相对 QTSA 基线的 delta。

输出：

```text
$RUN_ROOT/checkpoints/dvgbench_qtsa_i2e_sft_v1/
$RUN_ROOT/predictions/dvgbench_qtsa_i2e_sft_v1.jsonl
$RUN_ROOT/predictions/dvgbench_qtsa_i2e_sft_v1.summary.json
$RUN_ROOT/predictions/dvgbench_qtsa_i2e_sft_v1.comparison.json
$RUN_ROOT/logs/train_dvgbench_qtsa_i2e_sft_v1.log
$RUN_ROOT/logs/eval_dvgbench_qtsa_i2e_sft_v1.log
```

summary 同时报告：

- mIoU、Acc@0.5、DVGBench_AVG；
- 六类 Acc@0.5；
- bbox parse failure；
- explicit 格式成功率与平均显式描述长度；
- 每样本平均推理延迟；
- `question_e_used=false` 与 `gt_visible_during_inference=false`。

## 摘要决策

- 若 `DVGBench_AVG` 和 `Acc@0.5` 均提高：摘要可将 I2E 作为 QTSA 的文本推理
  补强模块，并保留 Agent 为下一阶段。
- 若只有一个主指标提高：如实写对应指标，暂不宣称全面提升。
- 若没有提高：回退现有 QTSA，摘要不写 I2E 贡献；I2E 仅保留为后续分析。
