# 冻结候选验证与保守重选实验

## 目的

当前 v4.1 的 Candidate Oracle Acc@0.5 比 one-pass 高约 4.24 个百分点，
说明候选池存在可恢复样本，但现有融合器无法稳定选中它们。该实验固定已经
生成的候选框，不重新运行 Target/Context/Relation/Zoom 搜索，只评测候选
验证与选择，避免每改一次评分规则就重跑 873 条 grounding。

## 两阶段协议

1. `verify_dvgbench_frozen_candidates.py`
   - 输入已有 hierarchical trace；
   - 将互斥候选框用 H0/H1/... 标在原始全图上；
   - 关闭 grounding LoRA，调用冻结基座做视觉候选验证；
   - 可选择一个候选或输出 `ABSTAIN`；
   - 不读取 GT、`bbox_norm` 或 `question_e`；
   - 输出独立 verifier sidecar JSONL。

2. `eval_dvgbench_frozen_candidate_selection.py`
   - 先仅根据 inference trace 与 verifier sidecar 确定 final bbox；
   - 高置信度、足够 margin 且综合证据提升时才允许替换 initial bbox；
   - verifier 弃权、格式错误、置信度不足或 margin 不足时保留 initial bbox；
   - final bbox 固定以后才从 `--index` 读取 GT 并计算指标。

## 正式选择器

`conservative_visual` 的默认门控为：

```text
verifier confidence >= 0.70
verifier top-2 margin >= 0.20
composite score gain >= 0.05
```

综合分数只使用无监督推理信号：

```text
0.55 visual verifier
+ 0.20 bbox token confidence
+ 0.10 relation consistency
+ 0.10 global constraint score
+ 0.05 box plausibility
```

这些默认值必须预先固定，或仅在训练/开发集上校准，不能使用 DVGBench test
GT 调参。

## 选择器对照

- `initial`：始终使用 one-pass bbox。
- `stored_fusion`：复现候选 trace 中已有选择结果。
- `visual_only`：不加保守门控，用于分析 verifier 的原始能力。
- `conservative_visual`：正式方法。

## 新增指标

除 mIoU、Acc@0.5、DVGBench_AVG、Recovery、Regression、Avg Calls、Latency
外，还输出：

- verifier Coverage / Abstain Rate / Mean Confidence；
- Visual-supported Replacements；
- Useful Call Rate 与 Wasted Call Rate；
- Mean Action Regret；
- 每个功能单元产生有效候选的比例；
- candidate oracle 与 alternative selection success。

所有 action quality 指标均在推理终止后使用 GT 进行事后评测，不参与选择。

## 运行

```bash
export CANDIDATE_TRACE=/path/to/main_hierarchical.jsonl
export DVBENCH_INDEX=/path/to/dvgbench_test_index.jsonl
export MODEL_PATH=/path/to/CoVT-7B-seg_depth_dino
export ADAPTER_PATH=/path/to/querytail_lora
export OUTPUT_DIR=/path/to/frozen_selection
export CODE_REVISION=$(git rev-parse HEAD)
bash train/scripts/run_dvgbench_frozen_selection.sh
```

正式报告首先比较四种 selector。只有当 `conservative_visual` 在冻结候选上
实现正的 Net Recovery、Regression 受控，并显著提高 Alternative Selection
Success，才启动新的 873 条在线候选生成实验。
