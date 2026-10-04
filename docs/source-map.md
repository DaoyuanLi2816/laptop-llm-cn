# 源码地图：从一个问题定位到函数

先看问题，再找模块，而不是按文件名背算法。每章源码节选使用 AST 符号提取，
完整代码页与本次构建的 commit 一致；本页末尾索引由当前 Python 文件列表生成。

| 你要追踪的问题 | 原文件 | 关键符号 / 章节 |
|---|---|---|
| 文字怎样变成输入和标签 | [tokenizer.py](../laptop_llm/tokenizer.py)、[data.py](../laptop_llm/data.py) | `build_sft_example`、`PackedTokenDataset` · 01 |
| 一个 token 怎样穿过模型 | [model.py](../laptop_llm/model.py) | `TransformerBlock.forward`、`GroupedQueryAttention` · 02–03 |
| 专家与低秩增量怎样更新 | [moe.py](../laptop_llm/architectures/moe.py)、[lora.py](../laptop_llm/architectures/lora.py) | `SparseMoE.forward`、`LoRALinear` · 04 |
| 哪些损失在训练哪个对象 | [engine.py](../laptop_llm/engine.py)、[objectives.py](../laptop_llm/posttraining/objectives.py) | CE、DPO、PPO、GAE、KL · 05–08 |
| 采样 token 怎样变成轨迹 | [rollout.py](../laptop_llm/posttraining/rollout.py)、[trainer.py](../laptop_llm/posttraining/trainer.py) | `collect_rollout`、`run_lab` · 06–07、15 |
| 为什么流式网页不等于生产服务 | [generation.py](../laptop_llm/generation.py)、[server.py](../laptop_llm/server.py) | `generate_tokens`、`stream_completion` · 09 |
| 状态、来源和索引怎样改变架构 | [architectures](../laptop_llm/architectures) | delta / residual / indexed / compression · 11–13 |
| 新监督与优化器怎样接入 | [optim.py](../laptop_llm/optim.py)、[model.py](../laptop_llm/model.py) | Engram / MTP / HybridMuon · 14 |
| 工具反馈与动作怎样区分 | [agent.py](../laptop_llm/posttraining/agent.py)、[frontier.py](../laptop_llm/posttraining/frontier.py) | `parse_action`、TITO、校正、GAR · 16 |
| 存储、采样、视觉接口如何分别验收 | [quantization.py](../laptop_llm/quantization.py)、[speculative.py](../laptop_llm/speculative.py)、[multimodal.py](../laptop_llm/multimodal.py) | pack / correction / prefix · 17 |
| 如何保留失败与固定证据 | [evaluation.py](../laptop_llm/evaluation.py)、[tests](../tests) | verifier、paired cases、持出集 · 10、18 |

## 两条调用链

```text
基础训练：CLI → run_stage → dataset/loader → model → CE/DPO → backward → optimizer → checkpoint
后训练：  CLI → run_lab → rollout → action mask/reward → GAE/GRPO/KL → optimize → 轨迹与 checkpoint
```

比较两条链时，用不同颜色标识“含梯度的参数”和“固定数据”。
rollout 的随机动作不是可反传的连续输入；训练在保存的 token 上重新计算当前 logp。

## 完整源码页

在线站下面的链接自动列出当前源文件；GitHub 阅读者可从上表进入原文件。

<!-- module-index -->
