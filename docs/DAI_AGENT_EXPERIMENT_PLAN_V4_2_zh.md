# DAI Agent 实验计划 v4.2：先选择器，后在线全量

## 旧 15 项矩阵状态

旧 15 项实验停止排队但不删除配置。已经完成的 one-pass、旧 hierarchical
结果保留为历史对照；未启动的 confidence-gated、parent-only、static-all、
预算和逐单元消融不再按旧顺序自动执行。

原因是 v4.1 已证明主要瓶颈是候选选择，而不是 bbox 解析或候选数量。继续
运行旧矩阵只会重复测量尚未修复的选择器。

## 阶段 A：冻结候选选择器（现在执行）

固定 v4.1 的 873 条候选 trace，不重新生成 bbox，运行：

1. `initial`：one-pass baseline；
2. `stored_fusion`：复现 v4.1 选择结果；
3. `visual_only`：测量 LoRA-disabled 全图候选验证器原始能力；
4. `conservative_visual`：正式保守选择器。

这一阶段只需要对具有至少两个互斥 hypothesis 的样本调用一次视觉验证器。
其余样本直接跳过，因此显存需求与单次 CoVT 推理相同，不需要多卡。

## 阶段 A 的通过条件

以下条件在运行前固定，不能根据 DVGBench test GT 反复调节：

- `Net Recovery Count > 0`；
- `Regression@0.5 <= 0.5%`；
- `Alternative Selection Success` 明显高于 v4.1 的约 8.1%；
- `conservative_visual` 优于 `stored_fusion`；
- 所有行 `question_e_used=false`；
- verifier 与 selector 的 `gt_visible=false`；
- 873 行、873 个唯一 sample_id。

若希望 Acc@0.5 提升至少 2 个百分点，需要净增加至少 18 个正确样本。v4.1
约有 37 个 oracle-recoverable 样本，在无回退情况下至少需要正确利用其中
约 48.6%。这是目标值，不作为伪造或筛选测试结果的依据。

## 阶段 B：正式在线实验

只有阶段 A 通过后，才重新运行一次 873 条完整 hierarchical inference，
并将保守验证器接在候选生成之后。随后完成下列精简正式配置：

1. One-pass QTSA（已有结果，不重跑）；
2. Same-query rerun（等额外计算量对照）；
3. Static zoom-all；
4. Confidence-gated rerun；
5. Parent-only verification；
6. Full diagnosis-driven hierarchical agent；
7. Full w/o candidate verifier；
8. Candidate Oracle（仅离线上限，不作为可部署方法）。

不再默认运行 15 项。Target、Context、Relation、Zoom 的逐项消融在主方法
获得正向结果后再补充。

## 最终表格

### 主结果

```text
Method | mIoU | Acc@0.5 | DVGBench_AVG | Recovery@0.5 |
Regression@0.5 | Avg Calls | Latency | Dispatch Rate
```

### 候选选择

```text
Selector | Candidate Oracle | Alternative Recall |
Alternative Selection Success | Replacements | Net Recovery
```

### Agent 决策质量

```text
Method | Failure Precision | Failure Recall | False Dispatch |
Useful Call Rate | Mean Action Regret | Stop/Abstain Rate
```

### 成本收益

```text
Budget | Acc@0.5 | DVGBench_AVG | Avg Calls | Latency
```

## 服务器执行顺序

1. 确认没有旧矩阵进程；
2. 拉取固定 commit；
3. 运行 `run_dvgbench_frozen_selection_v2.sh`；
4. 验证 sidecar 与四个 selector 都是 873 个唯一 sample；
5. 读取 `selector_comparison.csv/json`；
6. 未达到阶段 A 条件时停止，不启动新的全量 873；
7. 达到条件后，只启动阶段 B 的 Full hierarchical，完成后再次停下汇报。
