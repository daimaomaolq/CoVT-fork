# DVGBench 层级主动感知推理（正式实现）

本目录是 `agent` 分支的正式实现，主入口为：

```text
train/src/tools/eval_dvgbench_agentic_v3.py
```

旧的 `eval_dvgbench_agentic_inference.py` 与 `eval_dvgbench_hierarchical_*.py` 是早期原型，不用于正式实验。

## 1. 最终层级感知 Agent 与专用功能单元

系统是一个负责诊断、路由与决策的层级感知 Agent，按需调用四类专用功能单元；不是多个自治 Agent 协作，也不包含 Query Rewrite 单元：

- `TargetAgent`：先用 target clause 进行全图竞争探测；当候选重复 initial 且仍存在风险时，在不包含当前候选的重叠分区 transformed views 中搜索多个根假设，局部框统一映射回原图。
- `ContextAgent`：定位 context 区域。context bbox 只作为关系证据，不会混入最终 target bbox 候选。
- `RelationAgent`：对 target/context 配对、全局位置和顺序约束进行排序；它是无需额外视觉模型调用的推理 Agent。
- `ZoomAgent`：不负责从错误 initial 中搜索逃逸，只对父控制器选出的候选假设执行 crop-based zoom-in 验证，并将局部 bbox 映射回原图形成独立支持证据。

层级感知 Agent 负责 query constraint graph、无监督可靠性检查、依赖感知路由、全局竞争、候选加权融合、停止和升级。Target、Context、Zoom 是二次感知单元，Relation 是结构化关系推理单元；它们没有独立目标、长期状态或自治决策权，最终 bbox 只能由控制器选择。

## 2. 按需派发与候选权重

`routing.py` 使用以下无监督信号生成调用计划：

- parse validity；
- bbox token confidence；
- box area 和 shape plausibility；
- query 是否含有 context、relation、全局方位、顺序、朝向或时序语言；
- 最大 specialized-unit perception call 预算；
- 消融实验禁用的 Agent。

`hierarchical` 默认对每条样本执行一次最小全图 Target competition probe，因为单个高置信预测无法暴露“自信但定位错误”。随后才按分歧和风险动态使用剩余预算：分离的替代假设优先交给 Zoom 验证；Target probe 只重复 initial 时，小目标、粗框、关系、全局位置、顺序或时序风险触发重叠分区搜索。关系 query 在预算允许时保留 Context 证据。`competition_probe_mode=off|risk|always` 用于成本分析，默认 `always`。

预算 K 按 Target、Context、Zoom 的真实视觉模型调用逐次扣减，而不是按动作类型计数；Relation 是无视觉模型调用的结构化推理。运行时存在硬断言，任何路径超过 K 都会失败而不是静默超预算。

候选融合不是 bbox 平均或无条件投票。默认权重为：

```text
0.30 * bbox token confidence
+ 0.10 * box plausibility
+ 0.20 * target consistency
+ 0.15 * relation consistency
+ 0.15 * global constraint consistency
+ 0.10 * cross-observation agreement
- ambiguity penalty
```

正权重自动归一化。融合对象是 identity hypothesis cluster，而不是平坦候选列表：Base 和不同 Target 根框是互相竞争的假设，彼此 IoU 只表示冲突，不能相互提供 target consistency；只有以某个根候选为父节点的 Zoom transformed view 才能给该假设提供跨观察稳定性。每个假设选择 root 或 Zoom refined box 作为 representative，再在假设级进行 relation/global/ambiguity 竞争。

有效 initial bbox 默认受 evidence-supported replacement guard 保护。非初始候选只有同时满足可比分数增益，并获得下列至少一种独立证据时，才允许成为 final bbox：

- bbox token confidence 达到 `replacement_confidence_threshold`，且相对 initial 的增益达到 `replacement_confidence_gain_threshold`；
- Target 与其 crop-based Zoom 结果满足跨视图 IoU 和置信度确认；
- relation consistency 相对 initial 获得显著增益；
- global constraint consistency 相对 initial 获得显著增益。

trace 在 `verification_evidence.fusion` 中记录 `comparable_initial_score`、`replacement_confidence_gain`、`replacement_support_evidence`、`cross_view_partner_id` 与 guard 原因。确定性 query decomposition 保留关系词之前的完整目标子句，并仅从目标子句提取属性，防止 context 属性泄漏；这不是 Query Rewrite 单元。

权重和阈值只能在验证集确定，不能用 DVGBench test GT 调参。

## 3. Zoom 的坐标与语义安全

Zoom 只是 single-image transformed observation view，不是真实多视角或多 UAV：

1. object-relative query 优先使用 target/context 的 union crop，保留双方证据；
2. crop grounding 只接收去除全局方向、顺序和 target-context 关系后的 target clause；
3. 左上、右下、全局方位和顺序约束始终由父控制器在局部框映射回原图后重新计算，绝不按 crop 坐标解释；
4. Target 分区搜索使用带重叠的确定性 transformed observation views，并优先观察不包含 initial 中心的区域；
5. 局部框映射回原图后执行 identity IoU/center、relation drop、global-position drop 三类 guard；
6. Zoom 的 `parent_candidate_id` 与 `hypothesis_id` 明确绑定被验证根候选；
7. trace 同时记录 `local_bbox`、`global_bbox`、crop region 和全局约束重应用标记。

因此 crop 的局部置信度不能覆盖全局语义约束。

## 4. GT 与 question_e 隔离

`io.inference_input_from_row()` 是进入 Agent 系统前的唯一数据白名单，只传递：

```text
sample_id, image, query, class
```

主实验默认并由矩阵脚本固定使用 `--query-field question`。显式使用 `question_e` 或 `question_e_cn` 会报错；通用 `query` 若 provenance 指向 `question_e` 也会报错。`bbox_norm` 只传给 `evaluation.attach_evaluation()`，且该函数在 `parent.run()` 完全结束后才执行。

`--initial-predictions` 必须来自真实模型预测。缓存行必须包含 `raw_output`、`parse_ok`、`pred_bbox`、`final_bbox`、`inference` 等 prediction provenance 之一。仅包含 `answer/bbox/bbox_norm/question_e` 的数据索引会被硬拒绝，防止把 GT bbox 误当 one-pass prediction。0-1000 token 坐标会自动归一化到 0-1。

## 5. 输出 JSONL

每条记录包含：

- `inference.final_bbox`；
- `inference.diagnosis`；
- `inference.action_trace`；
- `inference.confidence`；
- `inference.unit_calls`；
- `inference.child_calls` 仅作为兼容早期实验协议的同值别名；
- constraint graph、routing plan、实际 budget 使用、target/context candidates、`hypothesis_clusters` 和 verification evidence；
- `decision = accept | refine | escalate` 与 `stop_reason`；
- 预算耗尽或观测不足时的 `human_feedback`；
- 独立的 `evaluation`，其中才允许出现 GT 和 IoU；
- `cost` 中的逻辑/实际 perception calls、specialized unit calls、初始/增量/端到端 latency 和 dispatch。

`--feedback-mode base` 会在停止后调用去除 LoRA adapter 的同一基座生成自然语言反馈。动作由确定性安全策略先选定，基座只负责叙述，不能改变动作或给出数值飞行控制指令；不合规输出自动回退到模板。建议可包括保持上下文的降低高度/高分辨率观察、更宽上下文、侧向或斜视重观察、时序观察和人工复核。

## 6. 方法与实验模式

```text
one_pass          原始 CoVT-SegDINO 一次 grounding，无 Agent
confidence_gated  parse/低 token confidence 时做一次 transformed rerun
parent_only       单控制器规则验证加一次通用 transformed rerun，无专用单元
static_all        固定调用全部适用专用单元
hierarchical      诊断和预算驱动的层级主动感知推理
```

PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File train/scripts/run_dvgbench_agentic_v3_matrix.ps1 `
  -Index <dvgbench_question_index.jsonl> `
  -ModelPath <base_model> `
  -AdapterPath <lora_adapter> `
  -InitialPredictions <existing_one_pass_predictions.jsonl> `
  -OutputDir <output_dir>
```

Linux/AutoDL：

```bash
bash train/scripts/run_dvgbench_agentic_v3_matrix.sh \
  <index.jsonl> <base_model> <output_dir> <lora_adapter> <existing_one_pass_predictions.jsonl>
```

使用 `InitialPredictions` 可保证各方法共享完全相同的 one-pass 初始框。矩阵会用 `--require-initial-confidence` 拒绝缺少实测 bbox-token confidence 的旧缓存；省略第五个参数时，脚本会先生成一次正式 one-pass 缓存并供后续全部方法复用。

矩阵脚本自动运行主对比、四个专用单元消融、约束图/语义坐标保护/防误修复三项机制消融，以及 K=0/1/2 成本点；主方法本身即 K=3，避免重复全量推理。

矩阵脚本默认使用 `AnchorModelId=['sam','dino']` 和 `AnchorPromptMode=query_tail`，与现有 CoVT-SegDINO one-pass 协议一致；只有训练配置不同才应覆盖这两个参数。

## 7. 已实现指标

- 定位：mIoU、Acc@0.5、DVGBench_AVG、per-class Acc@0.5；
- 恢复：Recovery@0.5、False Repair Rate、Regression@0.5、Net Recovery Count；
- 失败检测：Precision、Recall、Specificity、False Dispatch Rate、AUROC、AUPRC；
- 候选与选择：CandidateRecall@1/2/3、Candidate Oracle Acc@0.5、Alternative Candidate Recall@0.5、Alternative Selection Success、Search Yield@DeltaIoU0.1、Mean Candidate/Hypothesis Count、Candidate Diversity、Root Verification Rate、Oracle Gap；
- 成本：Avg Calls、Avg Executed Calls、Avg Specialized Unit Calls、初始/增量/端到端 mean/P50/P95 latency、Dispatch Rate；
- 选择性预测：Coverage、Selective Acc@0.5、Escalation Rate；
- 其他：confidence ECE/Brier、failure-type recovery、dispatch distribution、每个专用单元的调用和 IoU gain、反馈有效率与回退率。

“召回率”被拆成失败检测 Recall 与 CandidateRecall@K，避免把 Agent 是否发现失败和是否生成正确候选混为同一指标。

## 8. 验证

```powershell
$env:PYTHONPATH = "train/src"
python -m unittest discover -s train/tests -v
python -m compileall train/src/uav_agentic train/src/tools/eval_dvgbench_agentic_v3.py
```

测试覆盖数据隔离、缓存 provenance、按需派发、无 Query Rewrite 单元、关系排序、Zoom 坐标语义、one-pass 缓存与延迟、恢复路径、防误修复、关键机制消融、时序不确定性反馈、正式 CLI 和指标汇总。
