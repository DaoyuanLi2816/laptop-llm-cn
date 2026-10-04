# 术语与符号

读公式前先确定同一个字母在这章表示什么。论文之间常用不同符号，不要只靠缩写匹配实现。

## 张量符号

| 符号 | 本书常用含义 | 典型张量 |
|---|---|---|
| B | batch 中回答或样本数 | input_ids `[B,T]` |
| T / S | 当前 query 长度 / 可用 key 总长度 | cached decode 中 T 可为 1，S 更长 |
| D / d | 残差维度 / 每头维度 | hidden `[B,T,D]`，Q `[B,H,T,d]` |
| H / K / G | Query 头数 / KV 头数 / KV 组数 | 本地 GQA 里 G=K，H/K 个 Q 共享 KV |
| V | 词表大小 | logits `[B,T,V]` |
| S（AttnRes） | 深度来源数，不是 key 长度 | sources `[S,B,T,D]` |
| N / E | 样本量 / 专家数，依章节定义 | router `[tokens,E]` |

## 最易混淆的成对概念

| 概念 | 不是一回事的原因 | 阅读 |
|---|---|---|
| logits / logp | 前者未归一化，后者来自 log_softmax | 02、06 |
| attention mask / loss mask | 前者控制看见谁，后者控制谁提供目标 | 01 |
| old / reference | old 固定当前 rollout 的行为分布，reference 是长期锚点 | 06 |
| eval / no_grad | eval 不关闭反向传播，no_grad 才关闭记录计算图 | 02、07 |
| terminal / truncated | EOS 不 bootstrap，有限窗口通常仍可 bootstrap | 06 |
| forward KL / reverse KL | 加权分布不同，不是教师是否冻结 | 07 |
| on-policy / offline | 取决于状态分布与当前策略关系，不取决于“是否蒸馏”这个名字 | 07、15 |
| 稀疏计算 / 压缩缓存 | 少算位置与改变保存表示是两条轴 | 03、13 |
| 激活参数 / 总参数 | 未激活的专家仍驻留并可能有 optimizer 状态 | 04 |
| QAT / 打包推理 | QAT 修改训练前向；打包推理未必能训练 | 17 |
| SSE / continuous batching | SSE 是传输格式，不是模型调度机制 | 09 |

## 缩写不是路线图

RLHF/RLVR 主要说明奖励来源；PPO/GRPO 说明策略优化方式；CoT 说明可见数据与输出形式。
GQA/MLA/KDA 描述注意力或状态设计；MoE 描述专家结构；TP/EP 描述如何跨设备切分。
不要把这些不同层次排成“技术含量越来越高”的单一排行榜。
