# 13｜学会选内容：MSA 与压缩共享是两条轴

前置：第 03 章 sliding/gather、KL。源码：`indexed.py` 与独立实验 `compression.py`。
固定滑窗知道“最近”；可学习 indexer 尝试找到“相关”。压缩/共享则改变保存多少、几层共用，不是同一个问题。

## 为什么 top-k 需要自己的训练信号

本地 GQA 主干有 Hq 个查询头、Hkv 组 KV。每个 KV 组有一个小 index query，所有组共享 index key。
主路径：先选 k 个 token block，gather 其中 K/V，再计算精确 softmax。它没有构造整个主 attention 的 T×T 分数。
index 路径仍扫描前缀；维度较小不表示总体复杂度已经 O(T)，本地 Python 循环也没有 fused kernel。

```text
index scores → future/padding mask → block max → local block 强制占一个名额
             → top-k blocks → token indices → 主 attention gather
```

必须在 block max **之前**屏蔽未来。否则同块里未来的大分数会改变过去位置的选块，主路径随后做 mask 也救不回来。
local block 在预算内，不是额外送一个。最早位置可用块不足预算时，越界槽位有显式 allowed mask。
全 padding 行的 scores 先安全处理，再把权重归零，避免 softmax(-inf,…,-inf) 产生 NaN。

### 硬路由放大数值误差：一次真实的 Windows CI 失败

首轮 v0.3 CI 中，两个历史块含相同 token，理论索引分数相同。完整前向与分段
解码使用不同形状的 GEMM，舍入差异约 `1e-9`；硬 Top-k 却选了不同块，后续
logits 最大误差约 `0.0285`。**小数值误差不保证离散决策的小误差**。

本地参考实现按 token 使用同样的 `[B,D]` 投影形状，并以固定特征维归约计算索引
点积；稳定排序在同分时优先更早的块。未来 masked 块数不能改变过去的路由。
这增加 Python 循环，不是生产优化。原缓存等价性容差不放宽，CI 额外覆盖
`MKL_CBWR=COMPATIBLE`、重复 token、精确同分和不同 prefill 切分。
真实大规模 kernel 仍需单独检查精度、路由一致性与吞吐，不能套用这份参考代码的结论。

离散 indices 不能通过主 CE 给 index projection 普通梯度。本地辅助 KL 的教师为同组 Q 头**概率的均值**，不是平均 logits 后 softmax。
教师 detach，index hidden 输入也 detach：这个辅助项只更新 index Q/K，不影响主干、主 Q/K/V。
模型返回的 `auxiliary_loss` 包含索引、MoE，以及有 labels 时的 MTP；不能把总辅助项统一叫“router loss”。

`indexer.warmup=True` 显式运行 dense 选择并训练索引；默认 sparse。
引擎没有自动 warmup 调度或已有 dense checkpoint 的结构转换器。要做完整转换实验，先明确哪些新增权重随机初始化，再安排 warmup、稀疏化和消融。

## CSA2/CED：不要把另一份实验当成 MSA

`compression.py` 将完整 token 块的 KV 均值压缩；块的最后位置必须不晚于查询位置。
Full 创建 compressed memory 和 indices；Reindex **复用同一个 memory 对象**，按新 query 重选；Reuse 连 indices 也共享。
测试检查 storage/object identity，不能用“张量数值相同但复制了一份”冒充节省存储。

在 CED 设计中，全局 KV 的源是 encoder 输出的投影，局部 attention 是另一条支路。
本地实验由调用者显式传 source KV，**没有主模型 CED、局部补偿、学习式压缩器、层次候选池或 Bounded Replay**。
第一个压缩块尚未完成时，全局分支输出为零；完整模型应该由 local 分支提供信息。

```bash
python -m pytest tests/test_frontier_architecture.py -k "index or sparse"
python -m pytest tests/test_frontier_mechanisms.py -k csa2
laptop-llm pipeline --config configs/frontier_indexed.yaml --device cpu
```

实验任务：把 top-k 预算增到覆盖所有块，对照 dense oracle；只 backward index KL，验证主干 grad 为零/None。
修改未来的 K **和** V，验证过去不变。分别量计算预算、KV 字节和跨层共享；不要把其中一项改善写成三项都改善。
