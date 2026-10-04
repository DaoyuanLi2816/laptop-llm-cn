# 03｜GQA、Sparse Attention 与 MLA

本章问题：少算一些位置与少存一些历史，是不是同一件事？
前置：[02 Decoder](02-decoder.md)。实验入口：`python scripts/lesson_examples.py 03`。

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

DeepSeek DSA 使用学习到的索引来选 token；本章 `sparse.py` 的固定位置窗口不等于 DSA。v0.3 的另一条可学习 block 索引路线见[第 13 章](13-indexing.md)，也不宣称与 DSA 完全相同。
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

## 源码精读：可见集合先于 attention 权重

打开 [sparse.py](../../laptop_llm/architectures/sparse.py)，按“绝对 query 位置 → 允许 key → gather → softmax”读：

<!-- source: laptop_llm/architectures/sparse.py::sparse_attention -->

这里真实 gather 选中的 K/V；不是先算完整 T×T 分数再把大部分清零。
但 index 张量、重复头与 Python 循环本身也要成本，理论 FLOPs 不是实测吞吐。
有 past 时 query 的绝对位置必须加偏移；不能仅按本次输入长度判断因果性。

MLA 关注另一条轴：存 latent 再通过权重吸收计算输出。

<!-- source: laptop_llm/architectures/mla.py::LatentAttention.forward -->

重点标出 cache 里的 latent 与 RoPE 部分，和临时生成的主分支 K/V。
缓存省了哪些维度，应从驻留张量计算，而不是数 state_dict 的键。

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_03 -->

先完整计算 9 个 token，再分别计算前 4 个与后 5 个；预期 cache 等价。
本例两层 FP32 GQA 保存 K、V，各为 `[1,2,9,8]`，合计 `2×2×1×2×9×8×4=2304` bytes。
这不包含权重、临时 attention workspace 或 allocator。

## 小结与练习

小结：稀疏性改变可见或计算集合，GQA 改共享头数，MLA 改缓存表示；三个维度不要混成一个指标。

1. 将 Query 头数翻倍但 KV 头数和每头维度不变，cache 会翻倍吗？
2. 用完整前向和逐 token decode 比较，哪个测试能发现位置偏移错误？

答案提示：GQA cache 按 KV 头数计，不按 Q 头数计；第二个对照直接检测偏移。
带着参数与计算的区别进入 [04 MoE/LoRA](04-experts-adapters.md)。
