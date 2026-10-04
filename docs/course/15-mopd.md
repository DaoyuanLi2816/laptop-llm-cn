# 15｜多教师 OPD：在学生会遇到的前缀上教学

前置：第 07 章 KL、冻结教师。源码：`posttraining/trainer.py`、`objectives.py`。
MOPD 不是“加载几个大模型再平均 logits”；前缀来源、教师选择、概率空间和梯度路径都必须明确。

## 本地方法的三个契约

1. 学生先独立生成自己的轨迹；教师不替学生生成一个较容易的前缀。
2. 同一份学生 token prefix 给对应领域/effort 教师，教师冻结且 eval。
3. 只在学生动作位置优化 full-vocabulary `KL(student || teacher)`。

```json
{"domain":"math","effort":"low","prompt":[{"role":"user","content":"3+4=?"}]}
```

教师清单：

```json
{"math:low":"indexed/sft/final.pt","math:max":"hybrid/sft/final.pt"}
```

清单里的相对路径相对**清单文件**，不是 shell。effort 是 low/high/max 之一；缺路由提前报错，不默默用任意教师补位。
按路由分批前向，再写回原始 batch 顺序。每个 sample 使用一个教师，不平均冲突任务的分布。
`run.json` 保存输入权重/教师/数据 SHA256；同词表大小不够，tokenizer JSON 必须逐字节一致。

## 逐位置 KL 是什么

```text
p = softmax(student_logits)
log_p = log_softmax(student_logits)
log_q = log_softmax(teacher_logits).detach()
loss_t = sum_vocab p * (log_p - log_q)
loss = sum_action loss_t / number_of_action_tokens
```

学生的 p 参与求导，教师的 q 不参与。所有位置都在同一个 prefix 条件下比较。
本地轨迹采样本身在 no-grad 下，不对离散生成路径反传；更新是固定这些学生访问到的状态上的逐位置 KL。
如果只取 sampled token 的 `(log_p-log_q)` 然后随意 `.mean().backward()`，一般不等价于这个全分布目标。

Kimi K3 与 Thinking Machines 的公开方法还涉及 sampled-token 概率、stop-gradient 奖励、RL 更新/校正。
本地实现主要复现**学生前缀＋domain/effort 路由机制**，选择更易检查的全词表 KL；不称为逐公式重现 K3 MOPD。
MiMo V2.6 MOPD2 的多 prefix/teacher prefix 也没有实现。

## 一个应当失败的实验

令 student=teacher、相同 eval/no-dropout 路径，KL 应近零，不能从它声称学生变强。
扰动教师某个输出方向，KL 应大于零、学生参数改变、教师参数不变。
再更改 tokenizer JSON 中模板而不更改 vocab size，加载必须拒绝。

```bash
python scripts/frontier_smoke.py --output artifacts/mopd-lesson --device cpu
python -m pytest tests/test_lab_integration.py -k mopd
```

查看 `teacher_routes`、动作数量、objective loss，以及保存的学生文本。教师小/弱时蒸馏可能传递错误。
full-vocab KL 的 logits 内存随 B×T×V 增长；规模化需要分片、top-k 近似或采样估计，不能直接把教学张量放大。
毕业实验：固定学生、数据、生成预算，比较单教师、多教师、off-policy SFT；评估每个域与域外遗忘，而不只比较训练 KL。
