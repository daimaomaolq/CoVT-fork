# ICIG 2026 论文写作指南：DVGBench 隐式 UAV Grounding + Query-Aligned Visual Anchors

## 0. 给下一个论文线程的任务说明

请基于本文档直接起草 ICIG 风格论文初稿，并用本地 Springer LNCS 模板排版。

模板路径：

```text
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP
```

可从以下文件开始：

```text
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP\samplepaper.tex
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP\llncs.cls
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP\splncs04.bst
```

建议新建：

```text
F:\research\icig2026\main.tex
F:\research\icig2026\figures\
F:\research\icig2026\tables\
```

当前论文先写成 8-10 页 conference paper，不追求完整 journal 叙事。

## 1. 论文核心定位

这篇论文的安全主线是：

> We study implicit UAV visual grounding, where the query often requires contextual and relational understanding rather than direct category matching. Starting from a CoVT/Qwen-style multimodal backbone, we introduce query-aligned visual anchor tokens from SAM and DINO to inject object-structure and semantic-region cues into the generative grounding process. Under a controlled DVGBench implicit-query protocol, our anchor-enhanced model improves over the same backbone with plain SFT.

中文理解：

```text
任务：UAV 隐式视觉 grounding
方法：query-aligned SAM/DINO anchor token
基线：同一 CoVT backbone 的 plain SFT
贡献：不是全模型重训，而是让视觉专家 token 在 query-conditioned generation 中真正被用起来
```

不要写：

```text
We beat the official DVGBench Qwen2.5-VL-7B SFT baseline.
```

可以写：

```text
We use DVGBench as an implicit-query UAV grounding benchmark.
```

## 2. 推荐题目

首选：

```text
Query-Aligned Visual Anchor Tokens for Implicit UAV Visual Grounding
```

备选：

```text
Structure- and Semantics-Aware Anchor Adaptation for UAV Visual Grounding
Enhancing CoVT for Implicit UAV Grounding with SAM and DINO Visual Anchors
```

## 3. 摘要草稿

可直接改写：

```text
Implicit visual grounding in unmanned aerial vehicle (UAV) imagery is challenging because user queries often describe targets through contextual, relational, or scene-level cues instead of explicit object names. Although recent multimodal large language models can generate grounding coordinates, plain supervised fine-tuning may not fully exploit region-level structure and semantic visual evidence in UAV scenes. In this paper, we propose a query-aligned visual anchor adaptation strategy for UAV grounding. Built on a CoVT/Qwen-style multimodal backbone, our method injects lightweight visual anchor tokens derived from SAM and DINO near the textual query, enabling the model to condition coordinate generation on object-mask priors and semantic region cues. We evaluate the method on the implicit-query setting of DVGBench under a controlled generative grounding protocol. Compared with the same backbone trained with plain SFT, our SAM-DINO anchor model improves Acc@0.5 from 26.12% to 32.99% and DVGBench average score from 28.00% to 36.27%. Ablation results show that DINO anchors improve semantic localization, while SAM anchors help more predictions cross the strict IoU threshold. These results suggest that query-aligned visual anchors are an effective and parameter-efficient mechanism for UAV grounding.
```

摘要里的数字若 SAM-only 补完后主表微调，保持：

```text
Plain: Acc@0.5 26.12, AVG 28.00
Ours: Acc@0.5 32.99, AVG 36.27
```

## 4. 关键词

```text
UAV visual grounding; implicit query; multimodal large language model; visual anchor tokens; SAM; DINO
```

## 5. 论文结构

### 1 Introduction

要点：

1. UAV 图像不同于自然图像：高空视角、小目标、密集场景、遮挡和尺度变化。
2. 真实查询常常不是显式类别，例如“中间偏左、风机更大的白色飞机”，需要隐式推理。
3. MLLM 可以生成 bbox，但 plain SFT 容易只学文本到坐标格式，未充分利用区域级视觉专家。
4. SAM 提供 object-mask/shape prior，DINO 提供 semantic region prior。
5. 本文提出 query-aligned visual anchor tokens：把 SAM/DINO anchors 插入 query 附近，使坐标生成受 query 约束。

贡献写三条：

```text
1. We formulate a controlled implicit UAV grounding protocol on DVGBench for CoVT/Qwen-style generative grounding.
2. We propose query-aligned SAM/DINO visual anchor tokens, which inject object-structure and semantic-region cues into the generation process.
3. We provide ablations showing that DINO and SAM offer complementary benefits, and their combination improves the plain SFT baseline.
```

### 2 Related Work

建议分三小节：

```text
2.1 UAV Visual Grounding and Referring Expression Understanding
2.2 Multimodal Large Language Models for Grounding
2.3 Visual Expert Tokens and Region-Level Adaptation
```

可引用方向：

- DVGBench / DroneVG-R1：UAV implicit visual grounding benchmark and reasoning-oriented grounding。
- Qwen2.5-VL / LLaVA / InternVL：MLLM grounding。
- CoVT：visual thought/visual expert token backbone。
- SAM：object mask prior。
- DINO / DINOv2：semantic visual representation。

不要在 Related Work 里暗示本文和 DVGBench 官方 leaderboard 直接同协议竞争。

### 3 Method

建议标题：

```text
3 Query-Aligned Visual Anchor Adaptation
```

子节：

```text
3.1 Generative UAV Grounding Formulation
3.2 SAM and DINO Anchor Extraction
3.3 Query-Tail Anchor Injection
3.4 Warm-Started LoRA Adaptation
```

#### 3.1 任务定义

输入：

```text
I: UAV image
q: implicit natural-language query
```

输出：

```text
{<x1><y1><x2><y2>}
```

坐标归一到 0-1000。

#### 3.2 SAM/DINO anchors

写法：

```text
SAM anchors encode object-level mask and boundary priors, which are useful for small and irregular UAV targets.
DINO anchors encode semantic region-level evidence, helping the model associate implicit textual descriptions with visually meaningful regions.
```

当前最佳：

```text
SAM tokens: 8 fixed by the CoVT SAM branch
DINO tokens: 4
```

解释：

- SAM 分支内部固定产生 8 个 mask token，不能随意设 16。
- DINO token 数量增加到 8 没有提升，可能因为隐式 query 下过多 semantic anchors 引入冗余候选。

#### 3.3 Query-tail injection

核心动机：

```text
Putting visual anchors near the query preserves the original image prefix while making the auxiliary visual evidence directly condition the language query and subsequent coordinate generation.
```

可以配图：

```text
Image tokens -> original MLLM encoder/decoder
Text query -> [query] [SAM anchors] [DINO anchors] -> bbox generation
```

#### 3.4 Warm-start LoRA

写法：

```text
We first train a plain generative grounding LoRA to learn the coordinate output format, and then initialize anchor-enhanced training from this LoRA. This avoids cold-start instability caused by newly introduced anchor tokens.
```

## 6. Experimental Setup

### Dataset

写：

```text
We evaluate on the implicit-query split constructed from DVGBench. The training set contains 1,990 samples and the test set contains 873 samples. We use the original `question` field as implicit query and do not use the more explicit `question_e` field in the main experiments.
```

### Metrics

主指标：

```text
mIoU
Acc@0.5
DVGBench_AVG
class-wise Acc@0.5
parse_failed
```

说明：

```text
Acc@0.5 measures strict localization success.
DVGBench_AVG is the macro average over task categories/classes, reducing dominance from large classes.
mIoU reflects continuous overlap quality.
```

### Implementation Details

写：

```text
Backbone: CoVT-7B-seg_depth_dino
Training: LoRA SFT
LoRA rank: 64
LoRA alpha: 128
Learning rate: 2e-5
Epochs: 3
Batch size: 1
Gradient accumulation: 16
Warmup ratio: 0.03
Scheduler: cosine
Frozen modules: LLM and vision tower
Prompt mode: answer_only
Anchor prompt mode: query_tail
```

## 7. Main Results 表格

先预留：

```latex
\begin{table}[t]
\centering
\caption{Results on the DVGBench implicit-query UAV grounding setting.}
\label{tab:main_results}
\begin{tabular}{lccc}
\hline
Method & mIoU & Acc@0.5 & DVGBench Avg. \\
\hline
Plain CoVT SFT & 28.50 & 26.12 & 28.00 \\
DINO-only & 33.64 & 32.30 & 34.88 \\
SAM-only & -- & -- & -- \\
SAM+DINO (Ours) & 33.38 & 32.99 & 36.27 \\
\hline
\end{tabular}
\end{table}
```

注意：

- 表里建议用百分数，保留两位小数。
- SAM-only 补出来后替换 `--`。
- 如果会议篇幅紧，主表和消融表可以合并为同一张。

## 8. Per-Class 表格或柱状图

主线 SAM+DINO per-class Acc@0.5：

```text
disaster: 47.17
productive: 52.07
security: 47.62
social: 29.67
sport: 21.66
traffic: 19.44
```

DINO-only：

```text
disaster: 43.40
productive: 52.89
security: 42.86
social: 30.77
sport: 21.66
traffic: 17.08
```

Plain baseline class numbers没有完整整理在当前交接里；如果要画图，可以先只画 DINO-only vs Ours，或从 prediction/log 里重新统计 plain。

可写分析：

```text
The gain is more pronounced on disaster, security, and productive scenarios, where object-level structure and semantic context are important. Traffic remains challenging due to dense small objects and ambiguous spatial descriptions.
```

## 9. Ablation 写法

推荐消融问题：

```text
Q1: Does visual anchor insertion help over plain SFT?
Q2: Are semantic and structural anchors complementary?
Q3: Does increasing DINO token count help?
```

已有结论：

```text
Plain -> Ours:
Acc@0.5: 26.12 -> 32.99
DVGBench_AVG: 28.00 -> 36.27

DINO-only:
mIoU slightly higher than full model, but Acc@0.5 and AVG lower than full model.

SAM8+DINO8:
Acc@0.5 32.53, AVG 35.50; lower than SAM8+DINO4.
```

写法：

```text
DINO anchors contribute strong semantic localization and improve average IoU. However, the full SAM+DINO model obtains the best strict localization and macro average, indicating that SAM's object-mask prior complements DINO's semantic cues. Increasing DINO tokens from 4 to 8 does not further improve performance, suggesting that excessive semantic anchors may introduce redundant candidates in small-object UAV scenes.
```

## 10. Figure 占位

### Fig. 1 Method Overview

画法：

```text
UAV Image
  -> CoVT Visual Encoder
  -> original visual tokens

Image also goes to:
  -> SAM anchor extractor -> 8 structural anchors
  -> DINO anchor extractor -> 4 semantic anchors

Implicit Query
  -> query-tail insertion: query + SAM/DINO anchors
  -> CoVT/Qwen decoder
  -> bbox token output {<x1><y1><x2><y2>}
```

LaTeX 占位：

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/method_overview.pdf}
  \caption{Overview of the proposed query-aligned visual anchor adaptation.}
  \label{fig:method}
\end{figure}
```

### Fig. 2 Qualitative Results

建议选 4-6 个样例：

- Plain 预测偏离，Ours 更接近。
- 小目标。
- 隐式关系描述。
- 交通密集场景中仍失败的例子。

图例：

```text
Green: ground truth
Red: prediction
```

标题：

```text
Qualitative comparison on implicit UAV grounding.
```

### Fig. 3 Optional Loss or Per-Class Bar

如果时间紧，优先做 per-class Acc@0.5 bar chart；loss curve 不是必须。

## 11. Discussion / Limitations

必须诚实但不要自毁：

```text
Although the proposed anchors improve the controlled baseline, performance on dense traffic scenes remains limited. This suggests that implicit UAV grounding still requires stronger small-object disambiguation and possibly reasoning-aware supervision. We also do not claim direct comparability with DVGBench's official reasoning-tuned models, since our focus is controlled anchor adaptation on a CoVT/Qwen-style backbone.
```

如果需要写 “future work”：

```text
Future work will explore reasoning-annotated supervision, larger UAV-specific instruction data, and more efficient anchor selection mechanisms.
```

## 12. Related Baselines 的措辞

安全写法：

```text
We compare against a plain SFT baseline built from the same CoVT backbone, which isolates the effect of visual anchor adaptation.
```

不安全写法：

```text
We compare with Qwen2.5-VL-7B in DVGBench and outperform it.
```

可以在 Related Work 里引用 DVGBench 论文表格，但只说：

```text
DVGBench reports that reasoning-oriented training is important for implicit UAV grounding. Our work is complementary: instead of optimizing reasoning traces, we study whether query-aligned visual expert anchors can improve a CoVT/Qwen-style generative grounding pipeline under a controlled SFT setting.
```

## 13. 实验表述模板

Results paragraph:

```text
Table~\ref{tab:main_results} shows the results on the DVGBench implicit-query setting. Plain CoVT SFT obtains 26.12 Acc@0.5 and 28.00 DVGBench Avg., indicating that the backbone can learn the coordinate generation format but still struggles with implicit UAV queries. Adding query-aligned SAM and DINO anchors improves Acc@0.5 to 32.99 and DVGBench Avg. to 36.27. The improvement suggests that object-level structural priors and semantic region cues provide complementary evidence for query-conditioned grounding.
```

Ablation paragraph:

```text
The DINO-only model improves mIoU to 33.64 and Acc@0.5 to 32.30, showing that semantic visual anchors are effective for implicit localization. The full SAM+DINO model further improves the macro average and strict localization accuracy, supporting the complementarity of semantic and structural anchors. Increasing the number of DINO tokens from 4 to 8 does not improve the final score, suggesting that compact anchor sets are preferable for this setting.
```

## 14. 推荐最终结论

```text
This paper presents query-aligned visual anchor adaptation for implicit UAV visual grounding. By injecting SAM and DINO anchor tokens near the textual query, the model better exploits object-level structure and semantic visual cues during bbox generation. Experiments under a controlled DVGBench implicit-query protocol show consistent improvements over a plain CoVT SFT baseline, with complementary gains from SAM and DINO anchors. The results highlight a practical direction for adapting multimodal grounding models to UAV scenes without full-model retraining.
```

## 15. 最小可交稿清单

必须有：

- [ ] `main.tex` 能编译。
- [ ] Abstract / Introduction / Method / Experiments / Conclusion 完整。
- [ ] Table 1: main + ablation results。
- [ ] Figure 1: method overview。
- [ ] Figure 2: qualitative examples。
- [ ] Related Work 至少覆盖 DVGBench、CoVT、SAM、DINO、Qwen/MLLM grounding。
- [ ] Limitations 写清楚 controlled protocol，不直接碰官方 leaderboard claim。

可选：

- [ ] Per-class bar chart。
- [ ] Loss curve。
- [ ] Token sensitivity table。

## 16. 给另一个线程的第一步指令

可以直接对另一个线程说：

```text
请阅读 F:\research\CoVT-fork\docs\ICIG2026_Paper_Writing_Guide_DVGBench_SegDINO_zh.md 和 F:\research\CoVT-fork\docs\HANDOFF_DVGBench_ICIG_2026-06-30_zh.md，然后基于 F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP 的 LNCS 模板，在 F:\research\icig2026\main.tex 起草 ICIG 论文初稿。先保留 figures/method_overview.pdf 和 figures/qualitative.pdf 占位，不要虚构官方 baseline 对比。
```

