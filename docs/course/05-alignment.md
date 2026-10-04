# 05｜SFT、CoT、DPO 与奖励模型

本章问题：监督答案、排序偏好与验证奖励分别告诉模型什么？
实验入口：`python scripts/lesson_examples.py 05`。

## 先区分四个层次

| 名称 | 描述什么 | 本仓库如何演示 |
|---|---|---|
| SFT | 用示范回答做监督学习 | assistant-only 交叉熵 |
| CoT | 可见的分步推理文本/数据形式 | `<think>步骤</think><answer>答案</answer>` |
| RLHF | 反馈来自人类偏好或其训练的奖励模型 | pairwise reward model → PPO；demo 使用合成标签 |
| RLVR | 反馈来自可程序验证的结果 | 算术 verifier → GRPO |

GRPO 也能使用学习型奖励，PPO 也能使用规则奖励；当前 CLI 选择最容易理解的两条组合。
“GRPO=RLVR”或“PPO=RLHF”的等号不成立。

## SFT 与 CoT

把一段 reasoning 放进 assistant 消息，就能用同一个 SFT 目标训练。
本项目的 `<think>` / `<answer>` 是普通文本片段，不是必须占一个 ID 的魔法控制 token。
只有角色控制 token 是保留符号。加入几个标签不可能自动获得推理能力。

示例数据：

```json
{"messages":[{"role":"user","content":"2+3=?"},{"role":"assistant","content":"<think>2+3=5.</think><answer>5</answer>"}]}
```

这段 CoT 只是短算式示范。真实任务还需要步骤质量、可验证性、难度分布、长度预算与测试时计算评估。
可见 CoT 可能错误、合理化或与决定答案的内部计算不一致，不能把文本当作忠实心理记录。
本项目网页显示原始文本，没有把它宣传为“读取模型真实思维”。

## DPO 的相对偏好

同一 prompt x，有 preferred `y+` 和 rejected `y-`。定义：

```text
margin = beta * [(log pi(y+|x)-log ref(y+|x))
               - (log pi(y-|x)-log ref(y-|x))]
loss = -log sigmoid(margin)
```

序列 log probability 是**回答 token 的和**，包括监督的结束符，不包含 prompt。
这里没有显式 reward head；通过固定 reference 的相对概率学习偏好。
chosen/rejected 必须共享完全相同的 prompt token，不能按各自回答长度分别截断问题。
本实现优先固定 prompt，回答超预算则截断；长回答被截断可能损害偏好语义，正式实验应预先过滤。

`beta` 改变相对 reference 的尺度，不等于学习率。DPO 数据不是随当前 policy 在线采样的，
所以它与 PPO 的 rollout 更新不同。保存 DPO checkpoint 时必须保存最初固定的 reference；
恢复时把当前 policy 复制成 reference 会改变目标函数，本版本已拒绝这种旧格式伪恢复。

## Reward Model 的标签从哪里来

偏好对格式：

```json
{"prompt":[{"role":"user","content":"2+3=?"}],"chosen":"<answer>5</answer>","rejected":"<answer>6</answer>"}
```

模型最后一个有效 token 的 hidden state 经线性头变成标量 r。
Bradley–Terry 模型假设 `P(y+>y-)=sigmoid(r+ - r-)`，最小化其负对数。
奖励值有相对意义，不天然是正确概率，也不天然跨任务可比较。

训练后冻结 reward backbone 和 head，再给 PPO 的生成轨迹评分。
真人标注还需要清晰 rubric、标注一致性、多样性和数据合规；课程合成的正确/错误算术对
只是展示机械流程，不能称为完成了有代表性的真人 RLHF。

reward hacking：策略可能找到让 RM 高分的乱码、长度特征或模板，而没有真正满足任务。
因此 RM 偏好正确率、策略 reward、独立任务正确率要分别监测。

## 阅读与练习

先看 `tokenizer.build_sft_example`，再看 `engine.dpo_batch_loss`，最后看
`posttraining.rewards.ScalarHead` 与 `preference_reward_loss`。

手算：初始 policy=reference 时 margin=0，DPO task loss 应为 `log(2)`。
把 chosen 的 log probability 增大，loss 应下降。reward loss 对 chosen 分数的梯度为负，
梯度下降会把 chosen 分数往上推。测试检查方向，而不只是检查数值有限。

## 源码精读：一个偏好对如何进入 DPO

<!-- source: laptop_llm/engine.py::sequence_log_probabilities -->
<!-- source: laptop_llm/engine.py::dpo_batch_loss -->

先用回答标签确定 mask，再 gather 每个真实 token 的 log probability 并求和。
这里是 sequence logp，不是每 token 平均；改变长度会改变其量级。
DPO 的 margin 是 policy 的 chosen/rejected 差，减 reference 的同一差，再乘 beta。
`no_grad` 冻结 reference；chosen/rejected 辅助项则仍可能训练 policy 的 router。

初始 policy=reference 时 margin=0，CE 风格偏好 loss 为 `-log sigmoid(0)=log 2`。
它不是“模型回答正确率 50%”的实测结果，只是二元排序目标的初始化值。

奖励模型另外拟合 `sigmoid(r_chosen-r_rejected)`：

<!-- source: laptop_llm/posttraining/objectives.py::preference_reward_loss -->

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_05 -->

预期 chosen 的 loss 梯度为负、rejected 为正，DPO 初始化 loss 约 0.693147。
梯度下降会提升 chosen 分数；不是给每个 chosen token 添加“奖励标签”。

## 小结与练习

小结：CoT 是可见数据/输出形式，DPO 是偏好目标，RLHF/RLVR 是奖励来源维度。

1. 某个回答最终答案正确但解释错误，当前算术 verifier 能发现吗？
2. 若只恢复 policy 却重新初始化 DPO reference，是同一次实验的继续吗？

答案提示：本地最终答案 verifier 不检查每个 CoT 步骤；更换 reference 改变了目标。
继续读 [06 策略优化](06-reinforcement.md)。
