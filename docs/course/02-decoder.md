# 02｜现代 Decoder 的一层

约定 B=batch、T=序列长、D=残差维度、H=Query 头数、K=KV 头数、d=D/H。
`input_ids [B,T] → Embedding → hidden [B,T,D]`，残差块的输入输出维度不变。

```text
x ── RMSNorm ── Attention ── + ── RMSNorm ── SwiGLU/MoE ── +
└───────────────────────────┘   └──────────────────────────┘
```

## Norm 与残差

Pre-Norm 先归一化再进入分支，残差主干保留直接梯度路径。RMSNorm 是
`x / sqrt(mean(x²)+eps) * weight`，不减均值。平方和用 fp32，再转回激活 dtype，
减少半精度数值问题。它与 LayerNorm 的统计量不同，不是完全等价替换。

## 把多头写成矩阵

GQA 的 `Wq: D→H*d`，`Wk/Wv: D→K*d`。
reshape 后 `q [B,H,T,d]`、`k/v [B,K,T,d]`，每组 H/K 个 Query 共享 KV。
本实现为了兼容 SDPA 版本显式 repeat KV 来计算，但 cache 仍保留 K 个头。

attention 是 `softmax(QKᵀ/sqrt(d)+mask)V`。缩放降低大点积使 softmax 过尖的风险。
SDPA 的布尔 mask 中 True 表示可见；不要假定所有框架都是这个约定。

## RoPE 不是简单加编号

RoPE 把向量两维一组旋转，角度随位置变化，使点积编码相对位置关系。
这里用 interleaved 偶/奇配对；另一些实现采用前后半区配对，已有权重不能直接混用。
decode 必须从 `past_len` 起旋转；每一步位置从 0 开始会破坏 cache 等价。

扩大 `max_seq_len` 或 `rope_theta` 不是自动获得长上下文能力。还要考虑位置缩放、
长文训练与评测；扩大 buffer 只避免越界。
可选 QK Norm 在旋转前逐头归一化。它改变模型函数，不能无损加到旧权重上。

## FFN 与 LM head

SwiGLU：`down(silu(gate(x)) * up(x))`。两次 D→F、一次 F→D，约 `3DF` 参数。
F 常取约 8D/3，这里向上对齐到 64；`hidden_dim` 可以显式调整。
门控逐元素乘法不是 attention，也不跨 token 混合信息。

LM head 把 `[B,T,D]` 变成 `[B,T,V]`，默认和 embedding 共享权重。
统计唯一参数用 `num_parameters()`；state_dict 的两个键可能指同一份参数，直接求和会重复计数。

## 梯度与重算

CE → head → 每层 Attention/FFN → embedding。checkpointing 少存中间激活，反向时重算。
Python 闭包必须绑定当前层，否则反向可能错误地重算最后一层。
MoE 辅助 loss 作为返回值穿过重算边界，不依赖模块可变字段。

练习：dim=32、H=4、K=2，手画所有投影 shape。运行 `tests/test_model.py`，
故意写错 RoPE 偏移，预期 cache 测试失败；只检查输出 shape 无法发现这个错误。
