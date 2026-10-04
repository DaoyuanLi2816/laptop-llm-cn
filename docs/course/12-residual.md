# 12｜AttnRes 与 mHC：信息怎样穿过深度

本章问题：注意力能否沿网络深度选择信息，而不只沿 token 时间轴？
实验入口：`python scripts/lesson_examples.py 12`。

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

## 源码精读：softmax 的轴决定你写的是哪种机制

<!-- source: laptop_llm/architectures/residual.py::AttentionResidual.forward -->

输入 sources 为 `[S,B,T,D]`，S 是深度来源数量。keys 归一化，values 保留原幅值，
query 与 keys 内积得到 `[S,B,T]`，`softmax(0)` 对来源归一化。
如果写成 `softmax(-1)`，就变成在时间位置之间混合，不再是本章的 depth attention。
初始零 query 让来源权重均匀，因此得到 mean，不是 sum。

mHC 对多残差流的混合使用近似双随机矩阵：

<!-- source: laptop_llm/architectures/residual.py::sinkhorn -->
<!-- source: laptop_llm/architectures/residual.py::ManifoldConnection.write -->

Sinkhorn 在 log 域交替归一化行和列；有限迭代只有近似约束。
write 先用 B 混合旧流，再通过 C 写入块更新。B 的约束不能自动约束非线性 F 或 C。
本地 F 包括整个 block，并非上游逐子层或 SinglePass 布局。

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_12 -->

预期初始 AttnRes 等于来源 mean；50 次迭代后的矩阵非负、行列和约为 1。
`whole_network_nonexpansive=not_proved` 提醒你不要把局部矩阵性质扩大到整个网络。

## 小结与练习

1. 将 sources 的所有值乘 2，归一化 keys 与输出 values 分别如何变化？
2. 把 Sinkhorn 迭代数从 50 改成 1，哪个误差可能增大？

答案提示：忽略 eps 时 keys 大致不变而输出幅值翻倍；有限轮次的行列归一化误差可能增大。
