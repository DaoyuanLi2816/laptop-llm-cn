# 07｜On-policy Distillation：在学生会走到的状态上教

本章问题：教师的概率分布怎样教学生，又为什么要让学生自己采样前缀？
实验入口：`python scripts/lesson_examples.py 07`。

传统序列蒸馏可以先让教师生成一批文本，再像 SFT 一样训练学生。
问题是部署时学生会走到自己生成的前缀，和教师轨迹的数据分布不同。
在线蒸馏让学生当前 policy 生成回答，再请教师评价这些前缀上的下一 token 分布。
这是 [GKD / On-Policy Distillation](https://arxiv.org/abs/2306.13649) 的核心出发点。

```text
prompt → student rollout（无梯度）→ 同一整段 token
                                  ├→ student logits（有梯度）
                                  └→ teacher logits（冻结）
                                         ↓
                           回答位置的全词表 KL → student 更新
```

## KL 方向不是命名细节

本实验使用 reverse KL：`Σ_v p_student(v) log[p_student(v)/p_teacher(v)]`。
另一种 forward KL 是 `Σ_v p_teacher(v) log[p_teacher(v)/p_student(v)]`。
权重是谁的分布，决定了惩罚重点。用一个采样 token 的概率差不能冒充完整词表 KL。
纯函数同时支持两种方向；CLI 的 OPD 路线固定 reverse，避免参数太多掩盖主线。

温度 T：先用 logits/T 得到分布，目标乘 T²。采样温度与蒸馏温度是不同旋钮；
当前 CLI 两者都用1。softmax/log-softmax 用 fp32，以减少小概率的数值问题。

## 哪些地方 detach

教师权重全部 `requires_grad=False`，teacher logits detach。
学生生成 token 是离散样本，rollout 不保留采样计算图；更新时对这些前缀重新 forward。
因此这是 stop-gradient trajectory 上的 GKD 风格 surrogate，不是对“学生轨迹分布变化”
也做完整求导的无偏策略梯度目标。需要额外 REINFORCE 项的推导不应偷偷省略后还宣称相同。

只在回答动作位置计算 KL；prompt 参与上下文但不计损失。
师生 tokenizer 必须完全一致；模型宽度/层数可以不同，但 teacher 上下文必须容纳同一序列。
跨词表蒸馏需要额外的分布映射，不能仅判断 vocab_size 相等。

## 与 PPO/RLVR 的关系

PPO 使用标量 reward 与 advantage。OPD 使用教师的每个词表 token 软分布，信号更密集。
教师可能错误，蒸馏可能传递偏差；同等或更差教师不保证改进。
当学生与教师恰好相同时，full KL 应接近0。它不是“开个蒸馏开关自动变聪明”。

`lab_smoke` 用 pretrain 权重作学生、SFT 权重作教师，目的是产生非零分布差异，
而不是声称几步 SFT 就是高质量教师。它不调用收费 API，不下载外部模型。

```bash
laptop-llm lab opd --checkpoint artifacts/my-first-lab/pretrain/final.pt --teacher artifacts/my-first-lab/sft/final.pt --data artifacts/my-first-lab/rl_train.jsonl --output artifacts/distill-course
```

训练内存除师生权重外，还包含 `[B,T,V]` teacher/student logits。
大词表或长 CoT 时可能成为瓶颈。生产实现可做词表分片或分块 KL，但必须保留同一归一化语义。

练习：用 `torch.distributions.Categorical` 的 KL 做 oracle；检查 teacher.grad 为 None，
mask 外 student logits 梯度为0。再对比固定教师轨迹与每步学生 rollout，保持题目和 token 总预算一致。

## 源码精读：KL 方向与梯度方向是两件事

<!-- source: laptop_llm/posttraining/objectives.py::distillation_loss -->

`student_logits[B,T,V]` 与 teacher 必须对应同一 token 语义。
reverse 分支用 student 概率加权 `log p_student-log p_teacher`；forward 分支换成 teacher 加权。
教师无论哪种方向都 detach，不能因为选了 reverse 就让 teacher 也优化。
温度改变分布，T² 改变梯度量级；不是把生成温度和损失温度默认视作同一概念。

本实验只验证损失函数的梯度契约。完整 on-policy 性质还依赖采样发生在当前 student，
轨迹随后固定，并及时更新 policy version；一个纯 KL 函数本身不能保证数据是 on-policy。

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_07 -->

预期 teacher.grad 为 None，mask 外 student 梯度为零，有效回答位置有梯度。
这里输出的维数 V=5 是玩具概率实验，不是实际 tokenizer 的词表。

## 小结与练习

小结：蒸馏需要同时审计采样状态、分布支持集、目标公式和教师质量。

1. `teacher_logits.detach()` 删除后，会影响哪一条计算图？
2. 师生词表 ID 次序不同但 shape 相同，会不会被 shape 检查发现？

答案提示：teacher 会被反向更新；shape 检查发现不了词表意义错位。
多教师与域路由留到 [15 MOPD](15-mopd.md)。
