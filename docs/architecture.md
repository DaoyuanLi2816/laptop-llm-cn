# 模型结构：从 token 到下一个 token

本章对应 `laptop_llm/model.py`。建议一边阅读文字，一边在代码里跟踪张量形状。

## 总览

输入 `input_ids` 的形状是 `[B, T]`。Embedding 把它变成 `[B, T, D]`，依次经过 N 个 decoder block，最终 `lm_head` 输出 `[B, T, V]`：每个位置对整个词表的 logits。

```text
token ids [B,T]
      ↓ Embedding
x [B,T,D]
      ↓ N × {RMSNorm → GQA → 残差 → RMSNorm → SwiGLU → 残差}
hidden [B,T,D]
      ↓ RMSNorm + lm_head
logits [B,T,V]
```

模型是自回归的：位置 `t` 的 logits 预测位置 `t+1`。训练时一次处理整段；生成时先处理 prompt（prefill），随后每步只输入一个新 token（decode）。

## RMSNorm

RMSNorm 不减均值，只用均方根归一化：

```text
rms(x) = sqrt(mean(x²) + eps)
y = weight * x / rms(x)
```

代码先转 fp32 计算统计量，再转回原 dtype。这一点在 fp16 训练中比公式本身更重要：归约运算更容易积累数值误差。

## RoPE

RoPE 把每个注意力头的维度两两配成二维平面，位置越靠后，旋转角越大。对 Query 与 Key 同时旋转后，它们的点积天然包含相对距离 `m-n`。

RoPE 也适合 KV Cache：Key 写入缓存时已经按自己的绝对位置旋转好，后续不需要修改历史缓存。

## GQA

普通多头注意力让 Q、K、V 都有相同头数。GQA 让较多 Q 头共享较少 KV 头。例如：

```text
Q heads:  0 1 2 3 | 4 5 6 7
KV head:    0 0 0 0 | 1 1 1 1
```

8 个 Q 头、2 个 KV 头会把 KV Cache 降到普通 MHA 的四分之一。当前实现为了兼容较广 PyTorch 版本，在送入 SDPA 前用 `repeat_interleave` 展开 K/V；**缓存本身仍保持压缩形状**。

## SDPA 与 causal mask

`scaled_dot_product_attention` 会根据设备与 dtype 选择合适后端。没有 padding 的整段训练直接使用 `is_causal=True`；存在 padding，或带历史 cache 一次追加多个 token 时，代码会构造显式的带偏移因果 mask。

一个很隐蔽的错误是：decode 时 Query 长度为 1、Key 长度为历史长度加 1，不能套用“左上对齐”的普通非方形 causal mask，否则新 token 可能只能看到最早的 Key。仓库为 cache 场景显式处理了位置偏移，并用单元测试比较“完整重算”与“逐 token cache” logits。

## SwiGLU

前馈网络不是简单的 `ReLU(Wx)`，而是：

```text
SwiGLU(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )
```

默认隐藏维度约为 `8D/3` 并向上对齐到 64。门控分支决定哪些特征通过，up 分支携带内容。

## 权重绑定

输入 Embedding 与输出 `lm_head` 默认共享同一矩阵。它减少约 `V×D` 个参数，并让“读 token 向量”与“写 token 概率”处在同一表示空间。

## 还没有实现什么

- MoE、MLA、稀疏或滑动窗口注意力；
- YaRN 等长上下文外推；
- continuous batching、PagedAttention、投机解码；
- 多 GPU 的 FSDP、ZeRO 或张量并行。

这些都很有价值，但会掩盖单机完整链路的主线。仓库在接口上保留了继续扩展的空间。
