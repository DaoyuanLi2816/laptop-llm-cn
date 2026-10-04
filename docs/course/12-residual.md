# 12｜AttnRes 与 mHC：信息怎样穿过深度

前置：第 02 章 residual、softmax。源码：`architectures/residual.py` 与 `model.py`。
KDA/MSA 改的是时间轴；这里改的是**深度/残差流轴**。不应把它们都画成 token attention。

## AttnRes：让每个位置选择深度来源

普通残差不断累加 `x←x+F(x)`。本地 AttnRes 维护 embedding 和各 attention/FFN 的**原始输出**。
输入 `sources[S,B,T,D]`：S 是来源个数，T 是 token 位置。每个子层有独立可学习 pseudo-query `[D]`。

```text
keys_s = RMSNormalize(values_s)
score_s = dot(keys_s, learned_query)
weight = softmax(score, axis=sources)
mixed = sum_s(weight_s * values_s)
```

keys 归一化，而 values 保留原始幅度。错误地沿 T 做 softmax 会引入另一种操作，甚至泄露未来。
query 从零初始化，因此初始权重均匀；仍能得到 query 的非零梯度。

`attnres_block_size: 0` 存全部子层输出；正数表示每若干个**子层**将输出加成一个 block source。
默认 4 不是四个 Transformer 层，而是 attention/FFN 各算一次。
未完成 block 的部分和仍可作为 source；embedding 始终单独保留，最终 mixer 输出再进入 LM head。

深度来源不是持久 token KV：解码每个新 token 的 sources 在这一次 forward 内构建。
激活 checkpoint 重算必须是纯函数：本地将 immutable source stack 作为输入，并绑定当前 layer。
如果闭包读取之后被追加的 Python list，backward 会在错误的来源集合上重算，前向正常也可能梯度错误。

## mHC：多条残差流的受约束混合

本地有 `X[B,T,N,D]`，N 为残差流数。归一化后的全部流生成动态 A/B/C：

```text
base = sum_i A_i * X_i                  # A=sigmoid，读入子网络
update = TransformerBlock(base) - base  # 本地把整块定义为 F
X_new_i = sum_j B_ij * X_j + C_i * update
```

A/C 与 B 的约束不同：A 在 (0,1)、C 在 (0,2)，B 用 log-domain Sinkhorn 交替行列归一化。
双随机表示 B 的行和、列和近似 1；有限迭代有误差。它约束的是线性残差混合，不保证带 F 的整个网络非扩张。
本地最后平均残差流，进入 norm/head。此布局是整块教学包装，不是 V4 每个子层的完全一致布局；更不是 V4.1 SinglePass 的跨层系数复用。

## 动手检查

```bash
python -m pytest tests/test_frontier_architecture.py -k "residual or checkpoint or sinkhorn"
laptop-llm pipeline --config configs/frontier_mhc.yaml --device cpu
```

给 AttnRes 两个 source `[1,0]` 与 `[0,2]`，零 query 时输出应是均值，不是总和。
把 query 改成偏向第二个来源，观察其权重增加。对 B 检查行列和、非负性与谱范数容差。
比较开启/关闭 activation checkpoint 的所有参数梯度，而不只比较 loss。

代价：full AttnRes 存更多深度来源，mHC 存更多残差流；本地仍有许多小算子和矩阵展开。
优化需要融合、分布式激活管理和实测，不能把数学稳定性等同于速度提升。
