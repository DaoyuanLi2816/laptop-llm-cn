# 06｜从 token 到 PPO / GRPO 更新

本章最值得配着 debugger 看。先读 `rollout.py`，再读 `objectives.py`，最后读 `trainer.py`。
算法参考 [PPO 原论文](https://arxiv.org/abs/1707.06347) 与 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)；
工程阅读映射在[源码索引](../references.md)。

## 一个 token 是一个动作

状态 s_t 是 prompt 加已生成前缀，动作 a_t 是下一个 token。
reward 可以每步提供，也可只在最后给整个回答评分。
`Rollout` 存 ids、完整 attention mask、回答 action mask、terminal、old_logp、原始文本。

```text
ids:         BOS  USER  问题  END  ASSISTANT  答  案  EOS  PAD
action位置:    0     1     2    3       4       5   6    7
action_mask:   0     0     0    0       1       1   1    0
terminal:      0     0     0    0       0       0   1    0
```

动作位置4的 logits 预测“答”。EOS 也是策略选择，应该有 logp 与训练信号。
padding 不是动作。长度用完不是 EOS；要区分是否需要 value bootstrap。

## 同一个实验中的四个模型

| 对象 | 作用 | 更新吗 |
|---|---|---|
| policy | 当前要改进的生成模型 | 是 |
| old policy 的 logp | 生成这批数据时的行为分布 | 这批更新期间固定 |
| reference | 最初 SFT 的锚点，约束漂移 | 否 |
| reward model | 给结果评分 | PPO 时冻结 |
| critic | 估计状态的后续回报 | 是，独立 backbone＋value head |

old 与 reference 不是同一个概念：old 每批 rollout 刷新，reference 不刷新。
我们只保存 old logp，不额外保留 old 模型副本。PPO 的内存仍远多于单模型推理。

## 采样分布一定要对齐

课程 RL 直接从 `softmax(logits)` 采样，temperature=1，不截 top-k/top-p。
训练时重新计算同一分布的 logp。若采样用了温度或截断却用原始 logits 算 PPO ratio，
第一轮更新的 ratio 就可能不为1。
模型始终 `eval()` 关闭 dropout；这不阻止 `backward()`。采样函数单独 `no_grad()`。

## GAE 手算一次

`delta_t = r_t + gamma * V(s_{t+1}) * (1-terminal_t) - V(s_t)`。
`A_t = delta_t + gamma*lambda*(1-terminal_t)*A_{t+1}`，倒序递推。

两步轨迹：r=[0,1]，V=[0.2,0.3]，gamma=lambda=1。
真正 EOS 时 A_1=1-0.3=0.7，A_0=0+0.3-0.2+0.7=0.8，returns=[1,1]。
若最后只是截断且 V(next)=0.5，则 returns=[1.5,1.5]。
测试把这两个结果分别锁定，避免把所有长度截断误当真实结束。

本实验在最后一次采样动作上附加终点 reward（即使是部分回答），长度截断仍允许 critic bootstrap。
这是一种明确的有限 rollout surrogate；严肃任务需要定义截断评分、最长 episode 与终止语义，
不能不加分析地沿用某一框架默认值。

## PPO clipping 不是截 loss

`ratio = exp(logp_new - logp_old)`，最大化
`min(ratio*A, clip(ratio,1-eps,1+eps)*A)`。
正优势动作不希望概率无限上升，负优势动作不希望概率无限下降；这是局部稳定机制，
不是保证每次更新 KL 都不超过某阈值的硬约束。

value 也用相对 old_values 的 clipped MSE；本实现采用未截断和截断误差的较大者。
PPO reward shaping 在采样概率上加 `-kl_coef*(old_logp-ref_logp)`，只加在有效回答位置。
同一 rollout 更新 `epochs` 次时 old_logp、优势、returns、old_values 都不能跟着重算。

## GRPO 不用 critic，代价是什么

每个问题生成 G 条回答，以同题奖励组内均值作为 baseline：
`A_i=(r_i-mean(group))/max(std(group),eps)`，std 使用 population 版本。
每条回答所有 token 共享该优势；仍用 PPO 风格 clipped ratio，再加 reference KL surrogate。
少了一套 critic，但多条生成有真实计算与 KV 成本，不是免费节省所有资源。

全对或全错的组都可能方差0，此时优势精确为0。`zero_variance_groups` 是重要诊断：
若长期100%，先改善 SFT、问题难度与采样探索；不能把 loss 微小变化说成学到了新能力。

目标 reduction 也重要。本项目先每回答按有效 token 平均，再平均回答。
这与全局 token 平均、固定长度归一化、其他 GRPO 变体不同，会影响长度偏好。
不要只比较算法名字，要比较完整公式、mask、normalization 和 KL 估计方式。

## RLVR 反作弊边界

verifier 仅接受可选单段 `<think>` 后跟唯一 `<answer>整数</answer>`，完整匹配。
不执行输出，不在任意位置搜索正确数字，重复答案、尾随控制 token、格式作弊都不给分。
这验证最终算术答案，不验证 CoT 的每一步。
程序执行类奖励需要隔离沙箱、超时与资源限额；本仓库没有对模型代码使用 `eval`。

## 运行与验收

```bash
python -m pytest tests/test_posttraining.py tests/test_lab_integration.py -q
python scripts/lab_smoke.py --output artifacts/rl-course --device cpu
```

验收不仅看 final.pt：检查 old log-ratio 第一次更新接近0、冻结参数不变、critic 有更新、
prompt/pad 梯度不进策略目标、零奖励 dense GRPO 不产生伪更新。MoE 辅助项是另一个梯度来源。
这些测试通过仍不等于有用的 RL 收益；收益需要独立持出集和多个随机种子。
