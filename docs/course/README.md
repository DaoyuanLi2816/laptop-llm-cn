# 中文课程：从小模型到研究工程

前置知识：Python 函数/类、矩阵乘法、概率、链式求导，会调用 PyTorch `backward()`。
不要求先懂 CUDA kernel 或集群。每章先学一个契约，再读实现，再跑能推翻错误实现的测试。

| 章节 | 主问题 | 阅读源码 | 完成标准 |
|---|---|---|---|
| [01 数据与预训练](01-foundations.md) | 模型到底预测什么？ | tokenizer / data / engine | 手写右移标签与 loss mask |
| [02 现代 Decoder](02-decoder.md) | 信息如何穿过每一层？ | model | 标出每个矩阵的 shape |
| [03 Attention 家族](03-attention.md) | 稀疏计算与缓存压缩有何不同？ | sparse / mla | dense oracle 与 cache 等价通过 |
| [04 MoE 与 LoRA](04-experts-adapters.md) | 参数多为什么不一定算得多？ | moe / lora | 路由梯度与适配器合并等价 |
| [05 SFT、CoT、偏好](05-alignment.md) | 数据形式与目标函数如何组合？ | tokenizer / engine / rewards | 区分 CoT、DPO、RLHF、RLVR |
| [06 PPO、GAE、GRPO](06-reinforcement.md) | token 为什么会得到优势？ | rollout / objectives / trainer | 手算 bootstrap 与 clipping |
| [07 在线蒸馏](07-distillation.md) | 学生犯的错，教师怎么教？ | distillation_loss / trainer | KL 方向、冻结与采样对齐正确 |
| [08 系统与训练工程](08-systems.md) | 为什么放大规模会带来新问题？ | engine / ddp_lesson | DDP 对照＋多维并行预算 |
| [09 推理与服务](09-serving.md) | 模型怎样成为可交互服务？ | generation / server | 分清 latency、throughput、KV 内存 |
| [10 实验与研究习惯](10-research.md) | 如何证明有进展而不是碰巧？ | tests / evaluation / scripts | 完成可复现实验报告 |
| [11 KDA 与混合状态](11-delta.md) | 不保存全历史还能预测下一词吗？ | delta / cache / mla | 推导先遗忘再纠错，证明 cache 分支不可变 |
| [12 深度路由与 mHC](12-residual.md) | 残差为什么不必总是相加？ | residual / model | 标出 source softmax 轴，检验 Sinkhorn 行列和 |
| [13 可学习稀疏索引](13-indexing.md) | 离散 top-k 怎样学会选内容？ | indexed / compression | 因果 block max、detach、dense oracle 与共享 storage |
| [14 记忆、MTP、Muon](14-training.md) | 参数、训练信号与优化方向如何解耦？ | engram / model / optim | 标签偏移正确，MTP 真更新，optimizer state 重载 |
| [15 多教师在线蒸馏](15-mopd.md) | 如何在学生自己的前缀上整合专家？ | trainer / objectives | 同 tokenizer、域路由、KL 梯度与教师冻结 |
| [16 Agent RL 与一致性](16-agent-rl.md) | 多轮工具轨迹怎样变成训练样本？ | agent / frontier | TITO、动作 mask、校正分母、GAR 与版本契约 |
| [17 低精度、投机、多模态](17-deployment.md) | 存得少、算得快、看得懂是同一件事吗？ | quantization / speculative / multimodal | nibble、拒绝修正分布与图像 label mask |
| [18 前沿研究与行为审计](18-auditing.md) | 解释可信吗？榜单之外如何测模型？ | 论文台账 / traces / evaluation | 配对反事实、held-out harness、权限/奖励盲区审计 |

## 三条路线

**刚入门：** 01 → 02 → `pipeline smoke` → 09。先不要同时开 MoE 和 RL。

**后训练研究：** 05 → 06 → 07 → `lab_smoke` → 10，再读 verl/TRL 对应源码。

**模型/系统研究：** 02 → 03 → 04 → 08 → 09，再读 DeepSeek/Megatron/SGLang。

**前沿论文实验：** 先完成 01–10 的基本契约，再走 11–14（架构/训练）→ 15–16（后训练）→ 17–18（部署/研究）。运行 [Frontier Lab](../frontier-lab.md)，用[论文台账](../frontier-papers.md)核对实际实现与原文差异。

每次只改变一个因素。最先提交的成果应是一个能复现 bug 的测试，而不是一张好看的 loss 曲线。

## 概念层级

```text
架构：Dense / MoE；GQA / MLA；全局 / 稀疏 attention
数据：原始文本 / 对话 / 可见 CoT / 偏好对 / 可验证题目
目标：CE / DPO / PPO / GRPO / Distillation KL
奖励：人类偏好训练 RM（RLHF）/ 程序验证（RLVR）/ 混合
系统：单机 / DDP / TP / PP / EP / CP；同步 / 异步 rollout
部署：普通 KV / paged KV；串行 / continuous batching
```

这些层能组合，但不是任意组合都有效。配置明确拒绝本教学版不支持的 `MLA + sliding/QK norm`，而不是悄悄忽略开关。
