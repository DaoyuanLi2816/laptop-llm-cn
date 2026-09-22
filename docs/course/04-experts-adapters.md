# 04｜MoE 与 LoRA：两种不同的参数效率

MoE 增加可用专家容量，每个 token 只用一部分；LoRA 冻结原模型，只训练低秩增量。
一个讨论激活计算，一个讨论可训练参数，不能混为一谈。

## MoE 的五个步骤

输入展平成 `[N,D]`，N=B*T。router 投影产生 `[N,E]` 专家 logits。

1. fp32 softmax 得到专家概率。
2. 每个 token 取 top-k；选中权重重新归一化。
3. 找到属于某专家的 token 索引，只把这些 token 送入该专家。
4. 把专家输出乘路由权重，再用 `index_add` 散回原位置。
5. 额外加入所有 token 共享的专家输出。

E=4、k=2、shared=1 时，每 token 用两个路由专家和一个共享专家，**不是只用两个专家**。
全部专家参数依然驻留内存。因此“激活参数少”不代表笔记本能容纳任意大 MoE。
本实现 dropless，没有容量溢出后丢 token；这提高可读性，也缺少生产调度中的容量控制。

## 为什么 router 需要额外约束

如果所有 token 都去专家0，其他专家学不到东西，专家0还可能成为通信热点。
定义 `f_i` 为专家 i 被选择的频率，`p_i` 为平均路由概率，
本实现 `L_balance = E Σ_i f_i p_i`，f 是离散统计不求梯度，p 可求梯度。
另有 `L_z = mean(logsumexp(router_logits)^2)` 约束 logits 尺度。

最终是 `L_task + aux_coef * L_balance + z_coef * L_z`。
padding 不计入统计；各层辅助项取平均，不因层数变化无意放大系数。
top-k 索引本身不可微，选中概率有梯度；平衡项还给未选中的概率提供梯度。
本实现是带辅助损失的路由，不是 DeepSeek-V3 的 auxiliary-loss-free 路由策略。

常见误判：GRPO 组内全零奖励时 loss 仍非零，就宣称 RL 有进步。
如果模型有 MoE，可能只是 balance/z loss 在更新！先分离任务项与辅助项，再解释权重变化。

## LoRA 的推导

冻结 `W [out,in]`，训练 `A [r,in]`、`B [out,r]`：
`y = Wx + (alpha/r) BAx`。
可训练量从 `out*in` 变成 `r*(out+in)`。B 初始为零，初始函数与原模型一致。
第一步 A 梯度可能是零，因为它前面乘了零 B；这不是 bug。

`inject_lora` 默认只替换 q/v 投影，`lab lora` 用 SFT 数据实际训练。
保存前把 BA 合并回普通 W，沿用已有 inference checkpoint 格式。
optimizer 仍是适配器训练阶段的状态，合并后的权重不能直接配这份 optimizer 做精确续训。

```bash
laptop-llm lab lora --checkpoint artifacts/my-first-lab/sft/final.pt --data artifacts/my-first-lab/sft_train.jsonl --output artifacts/lora-experiment --rank 4
```

## 必须通过的测试

router 有有限、非零梯度；padding 不改变有效 token 的 auxiliary loss；
LoRA 注入前后 logits 相同；更新只影响 adapter；合并前后 logits 相同。
跨卡 Expert Parallel 还需要 all-to-all dispatch、负载均衡与通信重叠，这里留到系统章节。
