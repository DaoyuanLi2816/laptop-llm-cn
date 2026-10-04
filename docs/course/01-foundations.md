# 01｜数据、token 与预训练

本章问题：一段文字怎样变成可反传的监督？读完应能区分 token、位置、目标与两个 mask。
前置：整数索引、交叉熵。实验入口：`python scripts/lesson_examples.py 01`。

## 模型预测什么

给定前缀 `x_0 ... x_t`，模型输出词表分布 `p(x_{t+1}|x_≤t)`。
它不是从数据库取一句回复。生成是在自己的输出后面继续预测，错误会改变后续条件。

```text
输入位置       0       1       2       3
输入 token    BOS     你好     世界    EOS
监督目标      你好     世界     EOS     忽略
```

`model.forward(labels=...)` 内部把 logits 去尾、labels 去头。`PackedTokenDataset`
已经返回 `next_token_labels`，引擎在这个分支不再次右移。混用会变成跳一个词预测。

## Byte-level BPE 与中文

从全部 256 个字节建立基础字母表，再合并高频相邻片段。新汉字能退回 UTF-8 字节，
代价是词表小时一个汉字消耗多个 token。本项目有 NFKC 规范化，因此不是所有原始
Unicode 字节逐字节无损。训练与推理必须共享规范化规则。

两个 tokenizer 就算都叫“8000 词”，ID 42 的含义也可能不同。跨阶段、教师与学生、
奖励模型之间必须核验完整 tokenizer。项目将 JSON 嵌入 checkpoint，并逐字节比较。
不要中途重训 tokenizer 后继续加载旧权重。

## attention mask 不等于 loss mask

交叉熵 `L = -Σ log p(target) / N_valid`。SFT 只训练 assistant 正文与结束符，
用户输入、system、padding 的标签都是 `-100`。用户问题虽然没有 loss，回答仍必须能看见它。

把长度 5 与长度 50 的回答补齐到 50，不能对全部 100 个位置直接平均，否则 padding
会改变权重。评测按有效 token 聚合；DPO 正确率按样本聚合。
目前训练累积采用 microbatch mean 的平均；不同 microbatch 的有效 token 数悬殊时，
它不等于全局 token mean。要实现后者，先统计总分母，再缩放各 microbatch loss。

## 数据质量不是一个下载命令

生产流程至少包括：来源/许可 → 规范化 → 文档去重 → 质量筛选 → 隐私治理 →
领域混合 → 数据切分 → tokenizer → packing → 版本记录。
仅用 SHA256 去重抓不到改标题、翻译和近重复。合成数据也可能带来教师评测污染。

demo 只检查 I/O。课程算术按问题切分；DPO 的内部 eval 不一定从所有前序阶段隔离，
不能当作终极 holdout。`lab` 保存输入哈希，但哈希只能证明文件相同，不证明标签正确。

## Packing 的明确边界

当前预训练以 EOS 连接文档后切固定块，没有文档级 block-diagonal mask。
模型可能关注块内之前文档的文本。若要求硬隔离，需要 document ID、attention mask
和 position ID 配套修改；只插 EOS 不等于隔离。

## 自检

运行 `python -m pytest` 中的 tokenizer/data 测试。手写两条消息的 SFT 样本，打印
`(token_id, decoded_piece, label)`。预期 assistant 角色位置的 logits 预测第一段正文，
而不是正文位置自己预测自己。

再故意把所有 labels 设为 `-100`，解释为什么应在数据阶段拒绝样本，而不是继续接受 NaN loss。

## 源码精读：同位置标签与右移目标

先打开 [tokenizer.py](../../laptop_llm/tokenizer.py)，观察角色、正文和结束符分别怎样进入列表。
这里 `labels` 与原始 ids 等长；角色标记不产生 loss，但 assistant 正文与结束符产生监督。

<!-- source: laptop_llm/tokenizer.py::LLMTokenizer.build_sft_example -->

逐步阅读：`segment` 定义消息的物理布局；`labels.extend` 决定监督范围；
最后 `[:max_length]` 是截断边界。假如截断恰好删掉全部 assistant 标签，数据集必须拒绝该样本。
不要把 `labels == -100` 转成不可见 attention mask，否则回答看不到用户问题。

预训练数据已经提供下一位置目标，因此引擎走另一个 CE 分支：

<!-- source: laptop_llm/engine.py::language_model_batch_loss -->

读这个函数时先区分主 CE 与辅助 MTP：主 CE 使用 `next_token_labels`，
辅助头使用重新对齐的同位置 labels。两个分支不能套一个“统一 shift”再移动一次。

## 可运行小实验

以下是 [lesson_01](../../scripts/lesson_examples.py) 的实际代码，toy IDs 不是实际中文 BPE：

<!-- source: scripts/lesson_examples.py::lesson_01 -->

预期输入为 `[1,11,12]`，目标为 `[11,12,2]`；loss mask 为 `[False,True,True]`。
位置 0 的 logits 预测 ID 11，而不是重构 ID 1。这就是自回归训练与复制输入的区别。

## 小结与练习

小结：tokenizer 决定 ID 的意义，数据决定监督范围，CE 决定聚合方式。
词表大小相同不意味着教师和学生能互换 token。

1. 将第二个目标改成 `-100`，预测哪个 logits 行的梯度变零。
2. 一个 microbatch 有 2 个有效 token，另一个有 20 个；两批 mean 再平均会给谁更大单 token 权重？

答案提示：忽略目标的那一行不产生 CE 梯度；等权 microbatch 下短批中每个 token 的权重更大。
不要据此修改 attention 可见性。继续阅读 [02 Decoder](02-decoder.md)。
