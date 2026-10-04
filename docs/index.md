# 动手读懂前沿 LLM

从一个 token 的预测，走到一次可审计的训练更新。
{ .book-lead }

这不是 API 调用教程。你将配着真实 PyTorch 源码，理解模型结构、数据边界、
后训练目标、缓存与服务，再用小实验判断自己的理解是否正确。
课程使用中文讲解，所有章节都有 CPU 原理实验，不要求租 GPU 或调用付费模型。

[开始第一步](getting-started.md){ .md-button .md-button--primary }
[查看 18 章目录](course/README.md){ .md-button }

## 选择一条阅读路线

<div class="book-paths" markdown>

<div class="book-card" markdown>

### 从基础出发

数据与标签 → Decoder → Attention → 本地生成。

先能说清 `[B,T,V]` 的每个维度，以及一个 logits 位置在预测谁。

[读第 01 章](course/01-foundations.md)

</div>

<div class="book-card" markdown>

### 研究后训练

SFT/DPO → PPO/GRPO → 在线蒸馏 → Agent 轨迹。

亲手区分 old、reference、teacher，以及动作和环境反馈。

[读第 05 章](course/05-alignment.md)

</div>

<div class="book-card" markdown>

### 追踪前沿机制

KDA → 深度路由 → 稀疏索引 → 多教师与部署。

每次只替换一个机制，用因果性、梯度和缓存测试约束实现。

[读第 11 章](course/11-delta.md)

</div>

</div>

## 每章怎么读

1. 带着“本章问题”阅读概念与张量形状。
2. 在“源码精读”看真实函数：代码在构建时从当前源码提取，有原始行号和固定版本链接。
3. 运行 `python scripts/lesson_examples.py 章节编号`，先预测结果，再看断言。
4. 完成小结与练习：修改一个条件，解释哪个不变量会被破坏。

原始 Markdown 在 GitHub 也能读；文档站额外提供行号代码、全文检索、章节导航与复制按钮。
见[阅读指南](reading-guide.md)与[源码地图](source-map.md)。

## 能运行，不等于已有能力

示例模型随机初始化，极短训练通常输出无意义文本。工具任务的零奖励记录保留在
[验证记录](validation.md)中；代码正确性与模型能力是两种证据。
本地参考实现不是任何上游权重的兼容实现，也不复制闭源实验室系统。

## 公式、实现、实验要互相约束

课程组织参考 [《动手学深度学习》](https://zh.d2l.ai/) 的“概念与可运行代码交织”方式，
以及 [mini-verl](https://github.com/DaoyuanLi2816/mini-verl) 的工作流、源码契约和验证边界。
本书文字与实验围绕本仓库独立编写，不搬运它们的章节内容。

[前沿论文台账](frontier-papers.md)记录具体版本、阅读范围和未覆盖内容。
文档站是免费的静态阅读站，不是托管模型推理的在线聊天服务。
