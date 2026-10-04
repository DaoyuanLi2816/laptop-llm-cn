# 11｜KDA：用一块状态替代一串 KV

本章问题：怎样把历史压进固定状态，又保留可训练的纠错能力？
实验入口：`python scripts/lesson_examples.py 11`。

前置：第 03 章、矩阵外积。源码：`architectures/delta.py`、`cache.py`、`mla.py`；来源版本见[论文台账](../frontier-papers.md)。本章先研究本地递推器，不宣称复现 Kimi 的训练规模或 kernel。

## 从“保存过去”到“压缩过去”

GQA 的 cache 是过去每个 token 的 K/V；生成第 1000 个 token 时，序列维长度为 1000。
KDA 的 cache 是一块不断改写的矩阵 `S[B,H,Dk,Dv]`，外加因果短卷积的有限历史。
状态更小不等于没有代价：压缩可能丢掉可精确检索的信息，所以配置可以穿插全局 MLA 层。

把 `k` 看成地址、`v` 看成要记住的内容。旧状态估计这个地址的内容为 `kᵀS`。
写入不应总把新内容直接加上去，而应先算“旧内容错了多少”。本地单步公式：

```text
S_decay = diag(alpha) @ S_prev
error   = v - kᵀ @ S_decay
S_new   = S_decay + beta * outer(k, error)
y       = qᵀ @ S_new
```

`alpha[B,H,Dk]` 是逐 key 通道遗忘；`beta[B,H]` 控制这一次写入。先遗忘再求误差，不能交换次序。
本地用 `alpha=exp(-5*sigmoid(exp(log_scale)*logits))`，将 log-decay 限在 (-5,0)；状态累加为 FP32。
Q/K 在短卷积与 SiLU 后做 L2 归一化，输出按头做 RMS 归一化并乘输入生成的 gate。

## 跟一次 tensor 走完整条路

`x[B,T,D] → q/k/v[B,T,H,D/H] → S[B,H,D/H,D/H] → y[B,T,D]`。
时间轴是 Python 循环，头和 batch 并行；没有创建 `[T,T]` 主 attention score。
短卷积窗口只含当前及过去，未来 token 不可见。padding 位置不写状态、不推进卷积历史。
`cache.length` 仍记录位置长度；不要从固定形状的 `state.size(2)` 猜序列长度。

缓存对象不原地更新：从同一个 prefix 采样两条 continuation 时，各分支独立。
这与 `torch.no_grad()` 不同：no-grad 只关闭梯度，不会自动防止数据被覆盖。

## 混合模型不是整体常数 cache

`attention_type: hybrid` 按 `hybrid_global_every: 4` 插入全局层，并保证最后一层全局。
默认四层为 KDA/KDA/KDA/NoPE MLA；全局层仍保留增长的 latent cache。
NoPE 是不加显式位置旋转，不是“模型不知道先后”：前面的递推层已携带时序信息。
`cache_bytes()` 按实际 storage 去重，同时统计短 token 历史；测试量的是这个对象，不是 GPU 总显存。

## 可证伪实验

```bash
python -m pytest tests/test_frontier_architecture.py -k "cache or delta or causal"
laptop-llm pipeline --config configs/frontier_hybrid.yaml --device cpu
```

先手算 H=1、Dk=Dv=2 的一次外积。再比较完整前向与“prefill 三词、追加两词”；误差应在数值容差内。
改动第 8 个 token 后，前 7 个 logits 必须不变。检查纯 KDA cache 随 T 不增大，而 hybrid cache 会增大。
最后比较 alpha 接近 1 与接近 0 的状态衰减，别把输出 gate 和写入 beta 混为一谈。

工业差距：没有 chunkwise UT/Flash Linear Attention、状态量化、并行前缀扫描或超长序列性能验证。
要优化先写等价性 oracle，再替换循环；“线性复杂度”不保证这个 Python 后端比 SDPA 快。

## 源码精读：九行递推，分清矩阵两条轴

<!-- source: laptop_llm/architectures/delta.py::delta_update -->

设状态 S 的 shape 为 `[B,H,Dk,Dv]`。key 查询的是第一条特征轴，输出沿 value 轴。
`alpha[..., :, None]` 对 key 轴衰减，`prediction` 是旧状态对当前 key 的回忆；
`v-prediction` 是需要写入的误差，beta 决定纠错强度。
如果先求误差再衰减，得到的并不是本章推导的同一个更新。

显式矩阵式为 `(I-beta*k*kᵀ) diag(alpha) S + beta*k*vᵀ`。
在真实 attention 前向还要追 short-conv 的历史、q/k 的归一化与 output gate：

<!-- source: laptop_llm/architectures/delta.py::DeltaAttention.forward -->

cache 中不只有 S，还有短卷积尾部和逻辑长度。固定状态不能省略位置计数或卷积历史。
混合模型中的 MLA 全局层仍保存增长的历史，不能用纯 KDA 的结论宣传整个 hybrid cache。

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_11 -->

预期递推与矩阵 oracle 相同；纯 KDA 在长度 4 和 16 时 cache storage 相同。
这证明参考路径的状态大小，不是序列长度增大后实际延迟必然不变。

## 小结与练习

1. 将 beta=0，状态如何变化？将 alpha=1，哪个遗忘机制被关闭？
2. 如果只有每四层中的三层使用 KDA，模型整体 cache 是否固定？

答案提示：beta=0 只保留衰减；alpha=1 关闭该轴衰减但仍可纠错；混合全局层仍增长。
