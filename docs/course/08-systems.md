# 08｜工业系统不是只把参数调大

一个公式正确的训练循环，是研究工程的起点。模型、数据、通信与故障尺度改变后，
有些设计会发生结构变化。这里明确区分已运行实验与需要阅读的生产技术。

## 单设备训练已有的底座

预训练/SFT/DPO 引擎包含 AdamW、warmup+cosine、梯度累积、AMP、梯度裁剪、
activation checkpointing、验证与原子写入 checkpoint。非有限梯度会报错，避免静默污染权重。
`torch.compile` 可配置但未保证所有设备与 MoE 动态路由都获益；先用 eager 建立数值基线。

FP32、FP16、BF16 是不同数值表示。FP16 动态范围较小，常需 GradScaler；BF16
指数范围接近 FP32，但有效精度仍较低。FP8 不是把 `.to()` 参数换个 dtype 就结束，
还涉及 scaling、amax 历史、敏感算子保留高精度、通信和硬件 kernel。本仓库未实现 FP8 训练。

## 先估内存，再挑配置

P 个参数，粗略 FP32 AdamW 的参数+梯度+两份动量约16P字节；实际 AMP、master weights、
优化器实现和临时 buffer 会改变预算。再加激活、attention 工作区、logits 和框架 allocator。
PPO 还同时保存 reference/reward/critic；MoE inactive 参数仍占内存。
不要拿“推理4GB”推断“全参训练4GB”。

## 两进程 DDP：本仓库实际运行的系统课

```bash
python scripts/ddp_lesson.py
```

两份完整模型各看不同样本，反向 all-reduce 平均梯度。脚本比较 DDP 梯度与单进程
完整 batch 梯度，期望在容差内一致。CPU gloo 可在无独显机器上理解通信语义。
这不是给主引擎加了完整分布式训练/容错支持。

等价前提包括：相同初始参数、相同有效 token 权重、无随机 dropout 差异。
不同 rank 的有效 token 数不同时，平均 rank mean 不等于全局 token mean。
需要全局统计分母，再正确缩放本地 numerator。

## 多维并行阅读地图（未集成）

| 技术 | 切什么 | 新问题 |
|---|---|---|
| DP / DDP | batch 数据 | 梯度同步、全局 loss 归一化 |
| ZeRO / FSDP | 优化器状态/梯度/参数 | 参数 all-gather、分片 checkpoint、峰值内存 |
| Tensor Parallel | 单层矩阵 | 列/行并行、collective 布局、通信频率 |
| Pipeline Parallel | 模型层 | microbatch 调度、bubble、激活生命周期 |
| Expert Parallel | 专家 | token all-to-all、负载倾斜、共享专家布局 |
| Context Parallel | 序列 | 跨分片 attention、位置与因果边界 |

在 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 阅读通信与 parallel state，
不只看配置名称。画一张“某个 rank 持有哪段参数、哪个样本、哪一段序列”的表，比背缩写有效。
DeepSeek 的 MTP 是额外多 token 预测目标；它不等于本仓库普通 next-token CE，当前仅作阅读主题。

## 恢复是算法契约，不只是读一个 pt

精确恢复通常需要模型、optimizer、scheduler、AMP scaler、CPU/CUDA RNG、数据 sampler
位置、数据版本、tokenizer、reference、critic、rollout policy version，及执行环境。
只恢复 step 与参数不保证下一批数据和中断前相同。

本仓库当前边界：

- `train --resume` 恢复模型/optimizer/step，DPO 还恢复固定 reference；没有完整 RNG/sampler/scaler 重放。
- 旧 DPO checkpoint 若没有 reference，拒绝伪等价续训。
- `lab` 保存模型与训练诊断状态，但暂无 `--resume`；非空输出目录拒绝覆盖。
- LoRA 产物已合并为普通权重，只适合推理或新阶段初始化，不能直接续接 adapter optimizer。
- `.pt` 采用 pickle，只加载自己生成或明确可信的文件，不能当作安全的任意上传格式。

原子 rename 防止半写文件，但不替代多机一致性、校验和、分片版本和存储容灾。
正式集群还要处理 rank 故障、网络抖动、重复样本、elastic restart 和混合版本。

## 同步与异步 RL

本实验采样结束才更新，下一批使用新 policy。容易理解但训练与生成无法重叠。
工业系统可能 actor 持续生成、learner 持续更新，引入 stale policy、队列、权重广播和
importance correction。观察 [verl](https://github.com/verl-project/verl) 的 worker/rollout 边界，
理解“框架开销”里哪些是在维护正确性，而不是一律删掉。

练习：设计一个包含 data hash、policy version、sample seed 的 rollout schema；
说明如何拒绝一批 tokenizer 已变化或 policy 版本过旧的轨迹。再写一份中断恢复故障注入计划。
