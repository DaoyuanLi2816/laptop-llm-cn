# 从 tiny_LLM.py 到模块化研究实验室

原始单文件是阅读起点，不在仓库中复制一份会逐渐过期的巨型源码。
这里保留其“分章节、写出中间张量、解释每一步为什么存在”的方式，同时拆开可独立测试的边界。

| 原始学习入口 | 新结构 | 增加的研究问题 |
|---|---|---|
| `CFG` 全局配置 | `config.py`＋`configs/` | 结构开关、阶段参数、可复现实验目录 |
| 手写中文词表与最长匹配 | `tokenizer.py` 的 Byte-level BPE | 任意文本覆盖、规范化、词表版本一致性 |
| RMSNorm / RoPE / 因果 attention | `model.py`＋`architectures/` | GQA、QK Norm、Sparse、MLA、MoE |
| KV Cache 与生成 | `generation.py`＋`server.py` | prefill/decode 等价、SSE、多轮上下文与错误返回 |
| 预训练、SFT 与可见 CoT | `engine.py`＋`data.py` | packing、有效 token 权重、评测、阶段 checkpoint |
| 可验证奖励 / GRPO 演示 | `posttraining/` | 固定 old/reference、EOS mask、PPO critic/GAE、严格 verifier |
| 单文件注释 | `docs/course/`＋`tests/` | 手算、oracle、故障反例、与公开工业源码逐项对照 |

新加入 DPO、成对奖励模型、完整同步 PPO/GRPO 更新、学生 on-policy 蒸馏与 LoRA 合并。

原始教学中的小词表更容易在极窄任务里快速拟合；Byte-level BPE 扩展了输入覆盖，却也让
小模型需要学更多 token。不能只增加词表/模型/算法就保证聊天立刻变好。
