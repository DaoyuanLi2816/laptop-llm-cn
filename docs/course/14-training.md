# 14｜Engram、MTP、Muon：三种不同的训练杠杆

前置：第 04 章 MoE，第 01 章标签偏移。源码：`engram.py`、`model.MultiTokenHead`、`optim.py`。
分别讨论“存什么知识”“提供什么监督”“怎样移动参数”。它们不是同义的模型增大方案。

## 条件记忆与条件计算

MoE 根据 hidden 选择专家网络；Engram 风格模块根据短 token n-gram 查可学习表，再与 hidden 门控融合。
本地用两种 rolling hash，将 2…N-gram 映射到多张表。hash 相同可能碰撞；这是有损地址空间，不是数据库的精确事实查询。
key/value 由表项投影生成，hidden 与 key 的归一化相似度产生 gate，value 加到 token embedding。

解码 cache 只需最近 N-1 个 token；padding 不更新历史，两个 continuation 不共用可变历史。
整数 hash 每步模表长，避免长乘法溢出；表长/embedding 是配置项，不把参数大小冒充激活计算量。
本地没有 tokenizer 压缩、主机/RDMA 预取，也没有模型事实质量证据。增加记忆表仍需同数据预算消融。

## MTP 的关键不是多接几个 head，而是标签位置

基础 head：`hidden[t] → x[t+1]`。
第一个辅助模块：`hidden[t] + embed(x[t+1]) → x[t+2]`。
第二个辅助模块递归读取前一个表示与 `embed(x[t+2])`，预测 `x[t+3]`。
每个模块有 norm、融合投影和自己的 decoder block；token embedding 与 LM head 共享。

```text
输入:         A B C D E
主 head 标签: B C D E -
MTP1 标签:    C D E - -
MTP2 标签:    D E - - -
```

teacher forcing 中读取未来 token 合法，因为这是训练辅助模块，不是主 logits 看到未来。
推理主路径不调用这些辅助头；它们也没有自动接入本地投机解码器。
不能把训练时的真实未来 embedding 当作推理可获得的信息。

模型 `labels` 是与输入同位置的未移位标签；预训练 dataset 的 `next_token_labels` 已经移位。
引擎为 MTP 单独恢复同位置标签，主 CE 仍直接匹配已移位标签；否则不是漏掉 MTP 训练，就是主 CE 移两次。
`mtp_loss` 是各有效深度 CE 均值，乘 `mtp_loss_coef` 后加到辅助项；验证语言 loss 不包含辅助项。
SFT 中 user/image/padding 标签仍是 -100，不能为了辅助头偷偷恢复它们的监督。

## Muon：矩阵的更新方向与其余参数分开

AdamW 按元素历史统计缩放；本地 Muon 对二维矩阵的动量做近似正交化。
Newton–Schulz 多项式迭代只依赖矩阵乘法，不调用完整 SVD；转置矩阵形状后迭代、再还原。
迭代系数、次数、shape 缩放、动量与 weight decay 都属于优化器约定。

```text
二维 hidden 投影/专家权重 → momentum → Newton–Schulz → shape scaling → update
embedding/词表/table/router/一维参数 → AdamW 的一阶与二阶 state → update
```

并不是所有 ndim=2 参数都应该走 Muon：词表与查表嵌入有不同含义，router 也有离散路径敏感性。
checkpoint 保存两类 optimizer state；换 optimizer 后不能把旧 state 无检查载入。
本地是单设备教学变体，不等价于 PerHeadMuon、分布式梯度收集或生产超参数配方。

```bash
python -m pytest tests/test_frontier_mechanisms.py -k mtp
python -m pytest tests/test_frontier_architecture.py -k "mtp or memory"
python -m pytest tests/test_frontier_objectives.py -k "muon or router"
```

完成标准：证明预训练中 MTP 参数真有梯度；手算每个 label 的位置；重载 optimizer 后继续一步与未中断对照相同。
参数变多、辅助 loss 下降、方向更正交，都不是独立能力提升证据。
