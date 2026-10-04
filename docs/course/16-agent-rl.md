# 16｜Agent RL：轨迹是动作与观察的交替，不是一串答案

前置：第 06 章 PPO/GRPO、old/reference。源码：`agent.py`、`frontier.py`、`trainer.py`。
本地训练循环是**同步单设备**。双侧校正和版本 mask 是学习异步系统所需的数学单元，不是异步平台已经实现。

## 最小但有真实边界的工具环境

动作只能是一个严格 JSON 对象：calculator(a,b,op) 或最终 answer；拒绝重复键、布尔“整数”、过大数字、额外字段和超长文本。
工具没有 `eval`、shell、文件或网络权限。标准答案只在独立 verifier，成功要求最终答案正确，且在指定任务中实际调用匹配工具。
工具不信任模型声称“我已调用”。模型可以调用错误参数、输出非法 JSON、提前结束或耗尽预算，这些都保留在 traces。

```text
prompt/context (不训练) → model action (训练) → tool observation (不训练)
                       → model action (训练) → terminal verifier reward
```

沿用旧 tokenizer，工具反馈用带“工具反馈”前缀的 user role 承载；不是生产 tool chat template。
采样 ID、行为 logp 原样记录，不 decode 后重新 tokenize。`action_mask[t-1]` 表示 logits[t-1] 预测的 token[t] 是否为动作。
这就是本地 TITO 契约。`non_action_context_tokens` 包含原 prompt 与环境观察，不能误称“工具反馈 token 数”。
只有 final answer 是终止；token/context/turn limit 是截断。观测 JSON 自带正确算术结果不意味着模型学会使用它。

## PPO clipping 与双侧校正不是同一个目标

`r=exp(new_logp-behavior_logp)`。PPO 用 clipped surrogate；本地 calibrated 分支：

```text
keep = (1-lower < r < 1+upper) & action_mask
c = stopgrad(r) if keep else 0
loss = -sum(c * stopgrad(advantage) * new_logp) / original_action_count
```

越界 token 完全屏蔽，不把 ratio clamp 后留在目标里。区间严格，边界点也不保留。
分母不改成 surviving count，否则漂移越大、筛掉越多，剩余 token 权重反而越来越大。
`calibration_dropped_fraction` 需与行为版本差、entropy、reward 一起看；采样分布不一致时校正也可能失真。

## GAR：成功样本的质量不相同

有效二值奖励先将已证实 hack 的成功改为失败，再计算组内 `A=R-mean(R)`。
正优势用独立质量因子 f∈(0,1] 加权，以共同 lambda 归还被移走的正质量；lambda 有上限，最后全组中心化。
全成功/全失败组仍为零，GAR 不凭空制造正确性信号。
本地 f=1/轨迹事件数，仅是工具效率诊断，**不是 MiMo 的代码审查 agentic grader**。

`--freeze-router` 固定 MoE router，专家和其余网络仍能更新；它防止某条参数路径漂移，不保证完整路由完全不变。
因为 hidden 可变，固定 router 下的离散选择仍可能改变。我们未实现 R3，不能将“冻结”写成“重放原路由”。

## 训练/推理一致性与异步契约

本地 RL 用温度 1 的原始 softmax；serving top-p 是另一条路径。
若采样用 top-p/top-k，训练 logp 必须在**采样时实际候选集合**内重归一化；不能重新计算一个新集合冒充历史集合。
`replay_log_probs()` 接受记录的 bitmap，选中 token 不在集合内会报错。它是独立函数，当前 collector 不采集 top-p bitmap。

异步系统中每个动作应携带行为 policy version；`version_mask()` 过滤未来/过旧版本。
跨版本复用 KV 会混合不同时刻的模型状态，不能靠 logp 校正恢复一致性。
本地只记录一个轨迹版本并验证数学 mask；没有 queue、权重广播、部分 rollout 接续、故障恢复或长度偏差校正器。

```bash
python -m pytest tests/test_frontier_objectives.py tests/test_lab_integration.py -k "agent or calibrated or replay or gar or router"
```

成功协议 oracle 只证明 mask/工具接口能工作，不是训练模型的能力。
先用 protocol oracle 验证边界，再看随机初始化小模型的零奖励；奖励全零时训练 loss 仍可因 KL/辅助项变化，不能称为 RL 学会解题。
