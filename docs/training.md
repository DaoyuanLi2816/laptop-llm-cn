# 训练：从 smoke 到笔记本实验

这是入门三阶段说明。v0.2 的完整后训练路线请读[中文课程](course/README.md)，
特别是 [PPO/GRPO](course/06-reinforcement.md) 与[在线蒸馏](course/07-distillation.md)。

## 三阶段分别学什么

预训练优化所有文本的 next-token loss，建立语言与知识的基础表示。SFT 沿用预训练权重，用对话数据学习角色格式与回答行为。DPO 从 SFT 开始，同时冻结一份 reference，优化 chosen 相对 rejected 的概率差。

```text
DPO logit = beta × [(logπ(chosen)-logπ(rejected))
                    -(logπref(chosen)-logπref(rejected))]
loss = -log sigmoid(DPO logit)
```

reference 提供相对概率基线；它不是保证模型不会偏移或作弊的硬约束。

## 有效 batch

```text
effective batch = batch_size × gradient_accumulation_steps
```

单卡放不下更大 batch 时，先减少 `batch_size`，再增加累积步数。累积期间 loss 会除以累积步数，避免梯度尺度随累积次数线性放大。

## 混合精度

- RTX 30/40 系通常优先 `bfloat16`，动态范围比 fp16 大；
- 不支持 bf16 的 CUDA 可用 `float16 + GradScaler`；
- CPU 默认 `float32`，先保证正确性；
- 遇到 NaN 时记录首次异常 step，临时切回 float32 排查。

## 梯度检查点

`gradient_checkpointing: true` 不保存每层全部中间激活，反向传播时重算。它通常能明显降低激活显存，但会增加训练时间。模型参数、优化器状态和参考模型仍然占内存，因此 DPO 比 SFT 更吃显存。

## checkpoint 语义

- `--init-from`：只继承模型权重，创建当前阶段的新 optimizer；
- `--resume`：恢复同一阶段的模型、optimizer 和 step；
- 两者不能同时使用；
- checkpoint 内嵌 tokenizer，推理时不会误用另一份词表。

`--resume` 不保证 RNG、sampler、AMP scaler 的精确轨迹重放。DPO 恢复还必须有原始固定
reference；旧格式缺失时会明确报错。`lab` 当前只支持新实验与阶段初始化，不提供精确恢复。

保存采用“先写 `.tmp`、再原子替换”的方式，降低进程中断留下半个文件的概率。

## 如何扩大

建议顺序：

1. 用 `smoke` 跑完 tokenizer、三阶段、chat、serve；
2. 换真实数据，但先保持 smoke 模型；
3. 检查 token 数、loss 和固定提示输出；
4. 扩大 `dim/n_layers/max_seq_len`；
5. 用吞吐和峰值显存决定 batch/累积/检查点；
6. 只在 SFT 基础行为稳定后做 DPO。

不要同时替换数据、扩大模型、开启 compile 并改变 dtype。一次只改变一个主要因素，失败时才有可解释性。

## 训练成功不等于能力成功

最低限度需要分别报告：

- train/eval loss 与 perplexity；
- 固定任务集的准确率或偏好胜率；
- 关键失败样例；
- 生成速度与峰值显存；
- 训练数据版本、tokenizer 与 checkpoint 哈希。

“loss 降了”只能证明优化器找到了更好拟合当前数据的参数，不能单独证明事实性、安全性或通用聊天能力。
