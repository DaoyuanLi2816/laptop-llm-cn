# 02｜现代 Decoder 的一层

本章问题：保持 `[B,T,D]` 不变的一层，内部到底做了哪些变换？
前置：[01 数据](01-foundations.md)。实验入口：`python scripts/lesson_examples.py 02`。

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

## 源码精读：先读一层，不要先追完整 CLI

打开 [model.py](../../laptop_llm/model.py)，从这个小函数进入主干：

<!-- source: laptop_llm/model.py::TransformerBlock.forward -->

`attend` 返回分支输出、cache、索引辅助项，第一次残差相加保持 hidden 形状；
`feed_forward` 返回 FFN/MoE 输出和路由辅助项，第二次相加同样保持 `[B,T,D]`。
辅助项作为返回值传出，重算时不依赖模块中容易过期的可变统计字段。

再读 RMSNorm：

<!-- source: laptop_llm/model.py::RMSNorm.forward -->

只沿最后的 D 轴求均方，batch 与时间位置不会混合。`float()` 为敏感统计量保留精度，
`to(x.dtype)` 再回到激活类型。若错把 mean 的轴写成时间轴，会引入未来信息，shape 却可能仍正确。

| 张量 | dim=32、H=4、K=2 时 | 它的含义 |
|---|---|---|
| hidden | `[B,T,32]` | 每位置残差向量 |
| Q | `[B,4,T,8]` | 四个查询头 |
| 原始 K/V | `[B,2,T,8]` | 每两个 Q 头共享一组 KV |
| logits | `[B,T,64]` | 实验词表的未归一化分数 |

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_02 -->

预期 `logits_shape=[2,7,64]`，loss 有限，首层 Q 投影有梯度。
这证明维度与基本计算图连通，不证明学到了语言。

## 小结与练习

小结：残差维度固定，不代表中间投影维度固定；归一化和门控都有特定的轴。

1. 将 `tie_embeddings=False`，预测独立参数量和 logits shape 各有什么变化。
2. 为什么只运行 `model.eval()` 仍能得到梯度？

答案提示：解除共享增加词表矩阵，但不改变输出维度；eval 控制模块行为，不关闭 autograd。
下一章检查 [Attention 与 cache](03-attention.md)。
