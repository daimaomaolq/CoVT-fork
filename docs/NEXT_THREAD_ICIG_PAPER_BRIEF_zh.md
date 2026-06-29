# 给论文写作线程的 ICIG 初稿任务说明

这份文档只服务于论文写作线程。不要在这个线程继续跑实验、调代码或改训练脚本；它的任务是基于已有实验结论，先写出结构清晰、能本地编译、后续能快速打包上传 Overleaf 的 ICIG/LNCS 初稿。

## 1. 工作目录与模板

论文工作目录：

```text
F:\research\icig2026
```

LNCS 模板目录：

```text
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP
```

模板关键文件：

```text
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP\samplepaper.tex
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP\llncs.cls
F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP\splncs04.bst
```

建议新建论文结构：

```text
F:\research\icig2026\main.tex
F:\research\icig2026\references.bib
F:\research\icig2026\figures\
F:\research\icig2026\tables\
F:\research\icig2026\sections\
```

为了后续上传 Overleaf，`main.tex`、`llncs.cls`、`splncs04.bst`、`references.bib`、`figures/`、`tables/` 最终应在同一个可打包目录下。不要把图片引用写成绝对路径。

推荐本地打包目录：

```text
F:\research\icig2026\overleaf_package\
```

打包时复制这些文件进去：

```text
main.tex
references.bib
llncs.cls
splncs04.bst
figures\
tables\
```

## 2. 必须先读的实验背景文档

如果需要更完整实验背景，读：

```text
F:\research\CoVT-fork\docs\HANDOFF_DVGBench_ICIG_2026-06-30_zh.md
F:\research\CoVT-fork\docs\ICIG2026_Paper_Writing_Guide_DVGBench_SegDINO_zh.md
F:\research\icig2026\ICIG2026_Paper_Writing_Guide_DVGBench_SegDINO_zh.md
```

但论文线程不要重新发散实验路线。当前安全主线已经定好：

```text
DVGBench implicit UAV visual grounding
+ CoVT/Qwen-style generative grounding backbone
+ query-tail SAM/DINO visual anchor tokens
+ LoRA SFT under the same local protocol
```

## 3. 论文定位

推荐标题：

```text
Query-Aligned Visual Anchor Tokens for Implicit UAV Visual Grounding
```

论文核心故事：

```text
UAV grounding 中很多 query 不是直接类别匹配，而是隐式、关系式、上下文式描述。
普通 CoVT/Qwen-style backbone 经过 SFT 可以输出 bbox，但缺少显式区域结构和语义区域提示。
我们把 SAM 的 object-mask prior 和 DINO 的 semantic-region prior 作为 query-aligned visual anchor tokens 注入生成式 grounding。
在同一 DVGBench implicit-query local protocol 下，相比 plain CoVT SFT，SAM+DINO anchors 提升严格定位 Acc@0.5 和综合指标。
```

重要边界：

```text
不要写成 outperform official DVGBench Qwen2.5-VL-7B SFT baseline。
不要把本文主表直接和 DVGBench 论文排行榜混成同一 protocol。
可以引用 DVGBench 作为数据集和任务来源。
可以在 Related Work 里提 DVGBench 论文，但主实验 baseline 是我们本地同一协议下的 Plain CoVT SFT。
```

## 4. 推荐论文框架

### Abstract

写 150-200 words。必须包含：

- implicit UAV visual grounding 的难点；
- query-aligned visual anchors；
- SAM+DINO 分别提供结构和语义 cue；
- controlled DVGBench protocol；
- 主要数值：Plain CoVT SFT `Acc@0.5 26.12% / AVG 28.00%`，Ours `Acc@0.5 32.99% / AVG 36.27%`。

### 1 Introduction

建议 4 段：

1. UAV images have small objects, clutter, viewpoint variation, and implicit textual references.
2. Existing MLLM grounding often relies on global image tokens and may miss object-level structure.
3. SAM and DINO provide complementary visual priors, but they need to be aligned with the query rather than simply appended.
4. Contributions.

贡献写三点：

```text
1. We formulate a controlled implicit UAV grounding protocol based on DVGBench for generative bbox prediction.
2. We introduce query-aligned visual anchor tokens that combine SAM object-mask cues and DINO semantic-region cues.
3. We provide ablations showing that SAM+DINO improves strict localization over the same CoVT backbone and plain SFT baseline.
```

### 2 Related Work

三小节即可：

```text
2.1 UAV visual grounding and remote-sensing MLLMs
2.2 Visual grounding with MLLMs
2.3 Visual prompts, region tokens, and anchor-based adaptation
```

不要写太长。ICIG 初稿优先完整成文。

### 3 Method

建议标题：

```text
Query-Aligned Visual Anchor Adaptation
```

推荐小节：

```text
3.1 Generative Formulation for UAV Grounding
3.2 SAM and DINO Visual Anchors
3.3 Query-Tail Anchor Injection
3.4 Warm-Started LoRA Adaptation
```

方法要讲清楚：

- 输入是 image + implicit query；
- 输出是 `<x1><y1><x2><y2>` 格式 bbox；
- SAM tokens 表示候选对象/区域结构，当前稳定配置是 8 tokens；
- DINO tokens 表示语义区域，当前主线是 4 tokens；
- query-tail 表示把 anchor token 放在 query 后侧，让 grounding 生成阶段更容易用到；
- warm-start 表示从 plain generative SFT 能力出发，再学习视觉 anchor 的使用方式，避免从零开始破坏 bbox 输出格式。

### 4 Experiments

推荐小节：

```text
4.1 Experimental Setup
4.2 Main Results
4.3 Ablation Study
4.4 Qualitative Analysis
```

实验设置必须写：

```text
Dataset: DVGBench train/test local split
Train: dvg_train.jsonl, 1990 samples
Test: dvg_test.jsonl, 873 samples
Query field: question, not question_e
Backbone: CoVT-7B-seg_depth_dino / Qwen-style generative grounding backbone
Training: LoRA SFT, rank 64, alpha 128, dropout 0.05
Epochs: 3
Batch: per-device batch size 1, grad accumulation 16
LR: 2e-5, cosine scheduler, warmup ratio 0.03
Frozen: LLM and vision tower frozen
Prompt mode: answer_only bbox generation
Metrics: mIoU, Acc@0.5, DVGBench_AVG
```

### 5 Conclusion

一段即可。强调 visual anchor tokens 对隐式 UAV grounding 有帮助，并承认当前是 controlled protocol，后续可以扩展到更多 UAV benchmark 和 reasoning-style supervision。

## 5. 表格安排

### Table 1: Main Results

必须有。放在 Experiments 的 Main Results。

列：

```text
Method | SAM | DINO | mIoU (%) | Acc@0.5 (%) | DVGBench_AVG (%)
```

当前可填：

```text
Plain CoVT SFT | - | - | 28.50 | 26.12 | 28.00
DINO-only | - | yes | 33.64 | 32.30 | 34.88
SAM+DINO (Ours) | yes | yes | 33.38 | 32.99 | 36.27
SAM-only | yes | - | TODO | TODO | TODO
```

写法重点：

- `Ours` 不是 mIoU 最高，但 Acc@0.5 和 AVG 最高；
- 论文主指标优先强调 strict localization Acc@0.5 和综合 AVG；
- DINO-only mIoU 稍高说明语义 cue 有帮助，但 SAM+DINO 在严格定位上更稳。

### Table 2: Per-Class Acc@0.5

建议有，能体现 UAV 场景差异。

列：

```text
Method | Disaster | Productive | Security | Social | Sport | Traffic
```

Ours：

```text
47.17 | 52.07 | 47.62 | 29.67 | 21.66 | 19.44
```

DINO-only：

```text
43.40 | 52.89 | 42.86 | 30.77 | 21.66 | 17.08
```

Plain 如果暂时没有逐类表，可以先留 TODO，或者从已有 predictions/log 补。

### Table 3: Ablation Study

如果 Table 1 已经有 SAM/DINO 消融，Table 3 可以放训练策略：

```text
Setting | mIoU | Acc@0.5 | AVG
Plain SFT | 28.50 | 26.12 | 28.00
SAM+DINO without warm-start | TODO/optional
SAM+DINO with warm-start | 33.38 | 32.99 | 36.27
High-token SAM8+DINO8 | 33.36 | 32.53 | 35.50
```

如果篇幅紧，Table 3 可以省略，把 ablation 合并进 Table 1。

## 6. 图片安排

### Figure 1: Method Overview

必须预留。建议画成横向 pipeline：

```text
Image -> CoVT/Qwen Vision Encoder -> visual tokens
Image -> SAM -> object mask anchors -> projection
Image -> DINO -> semantic anchors -> projection
Query -> query-tail fusion with anchors -> Qwen decoder -> bbox tokens
```

图注强调：

```text
Overview of query-aligned visual anchor adaptation. SAM and DINO provide complementary anchor tokens, which are injected near the query to guide generative bbox prediction.
```

### Figure 2: Qualitative Results

必须预留。至少 3-4 个例子：

- good case: Ours bbox closer than plain;
- small object case;
- cluttered traffic/social case;
- failure case, especially traffic or sport.

图中建议用：

```text
Green: ground truth
Red: prediction
```

### Figure 3: Per-Class Bar Chart or Training Curve

可选。如果来得及：

- per-class Acc@0.5 bar chart 更有论文价值；
- loss curve 只能作为补充，不是必须，因为核心指标来自 eval。

如果没有时间，不要硬画 loss 图；优先 Table 1 + Table 2 + Figure 1 + Figure 2。

## 7. LaTeX 文件组织建议

`main.tex` 可以采用：

```latex
\documentclass[runningheads]{llncs}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}

\begin{document}
\title{Query-Aligned Visual Anchor Tokens for Implicit UAV Visual Grounding}
\author{Anonymous Authors}
\institute{Anonymous Institute}
\maketitle

\begin{abstract}
...
\keywords{UAV visual grounding \and Multimodal large language models \and Visual anchors \and SAM \and DINO}
\end{abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_method}
\input{sections/04_experiments}
\input{sections/05_conclusion}

\bibliographystyle{splncs04}
\bibliography{references}
\end{document}
```

如果赶时间，也可以先不拆 `sections/`，直接写在 `main.tex`。但为了协作和 Overleaf，拆分更清楚。

## 8. Overleaf 打包规则

不要上传整个 `F:\research`，只上传论文包。

建议生成：

```text
F:\research\icig2026\overleaf_package.zip
```

包内结构：

```text
main.tex
references.bib
llncs.cls
splncs04.bst
sections\01_introduction.tex
sections\02_related_work.tex
sections\03_method.tex
sections\04_experiments.tex
sections\05_conclusion.tex
figures\method_overview.pdf
figures\qualitative_examples.pdf
figures\per_class_acc.pdf
tables\
```

所有图片引用用相对路径：

```latex
\includegraphics[width=\linewidth]{figures/method_overview.pdf}
```

## 9. 论文线程不要做的事

- 不要重新讨论是否用 DVGBench 官方 Qwen2.5-VL-7B SFT 当主 baseline。
- 不要把 adapter-head 旧实验和 generative bbox 主线混表。
- 不要把 `question_e` 写成主线实验。
- 不要声称已经超过 official DVGBench leaderboard。
- 不要等待所有消融都完成才开始写；SAM-only 可以先留 TODO。

## 10. 可以直接交给论文线程的第一句话

```text
请只做 ICIG 论文初稿写作和 LaTeX 排版，不要继续跑实验。请先阅读 F:\research\icig2026\NEXT_THREAD_ICIG_PAPER_BRIEF_zh.md 和 F:\research\CoVT-fork\docs\ICIG2026_Paper_Writing_Guide_DVGBench_SegDINO_zh.md，然后基于 F:\research\icig2026\LaTeX2e+Proceedings+Template+ZIP 的 LNCS 模板，在 F:\research\icig2026 下建立 main.tex、sections、figures、tables、references.bib。论文需要预留 Table 1 main results、Table 2 per-class results、Figure 1 method overview、Figure 2 qualitative examples，并保证后续能快速打包上传 Overleaf。
```
