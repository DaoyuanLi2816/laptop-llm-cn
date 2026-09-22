# 03｜GQA、Sparse Attention 与 MLA

它们解决的是不同问题：GQA 共享头；稀疏 attention 减少可见边；MLA 压缩缓存表示。
不要把 FlashAttention 当作稀疏 attention：前者可以精确计算 dense attention，只是改变内存访问方式。

## KV Cache：存什么，省什么

训练时并行输入整段；生成先 prefill 前缀，再每次输入一个 token。
历史 K/V 不依赖未来输入，所以可以复用。Query 通常不缓存。

GQA KV 元素数约为 `2 * B * layers * T * K * d`。
例：B=1，层数=8，T=512，K=2，d=32，fp16 每元素2字节，KV 约1 MiB。
这里只算 KV，不算权重、临时工作区、激活和框架预留内存。

测试契约：相同模型、相同位置，完整计算和带 cache 分段计算的 logits 应在浮点容差内相同。
同时测试一次追加多个 token；只测逐 token decode 容易漏掉 offset causal mask 的错误。

## 滑窗＋sink：实际稀疏计算

位置 t 允许看最近 W 个 token 与开头 S 个 sink，但永远不允许未来 token。
例如 W=3、S=1 时，位置 5 可看 `[0,3,4,5]`。
sink 和窗口重叠时必须去重，否则 softmax 会把同一个 token 算两次。

[sparse.py](../../laptop_llm/architectures/sparse.py) 构造 `[T,W+S]` 的索引表，
直接收集 K/V，再计算这些边的分数，不物化 T×T 分数矩阵。
内存中仍有 `[B,H,T,W+S,d]` 的收集结果，Python/PyTorch 实现可能比 dense SDPA 慢。
渐近稀疏不等于实测更快，要看常数、kernel 启动和显存带宽。

`dense_every=2` 让每第二层保持全局注意力，扩展信息通路。全局层也带回二次计算。
本实现不驱逐旧 KV，因此它减少 attention 边数，但没有把 KV 内存变成常数。

DeepSeek DSA 使用学习到的索引来选 token；本项目固定位置窗口不等于 DSA。
阅读[官方实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)时，区分 indexer 成本、选中 token 的注意力成本和 kernel 成本。

## MLA：用代数消去展开缓存

核心低秩表示 `c = Norm(x W_downᵀ)`，维度 R。普通展开计算为：

```text
K_content = c W_keyᵀ
V         = c W_valueᵀ
score     = Q_content K_contentᵀ + Q_rope K_ropeᵀ
output    = softmax(score / sqrt(2d)) V
```

利用矩阵乘法结合律：`Q Kᵀ = (Q W_key) cᵀ`，
`attention V = (attention c) W_valueᵀ`。
我们缓存 c 与共享的 RoPE key，不缓存每个头展开后的完整 K/V。
位置分支分开是因为位置相关的旋转不能简单当成固定矩阵无条件吸收。

教学 cache 为 `[B,1,T,R]` 与 `[B,1,T,d]`。R=4、H=4、d=8 时，
每 token 为12元素；相应 MHA 完整 KV 为64元素。选择很大的 R 就未必省空间。

[mla.py](../../laptop_llm/architectures/mla.py) 没有复刻 DeepSeek 的全部投影、维度与 kernel。
它专门展示压缩与吸收，测试用显式展开 K/V 作为 oracle。参考[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)。

## 实验

```bash
python -m pytest tests/test_advanced_architecture.py -q
laptop-llm pipeline --config configs/research_mla.yaml --device cpu
```

先证等价，再测性能。将 window 设为大于序列长且 sinks=0，预期 sparse 与 dense causal 相同。
再把窗口缩小，预期完整序列 logits 与 dense 不同，但该 sparse 模型自己的 cache 仍应等价。
这是“改变架构”与“加速同一函数”的区别。
