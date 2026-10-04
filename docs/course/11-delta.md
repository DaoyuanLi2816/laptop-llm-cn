# 11｜KDA：用一块状态替代一串 KV

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
