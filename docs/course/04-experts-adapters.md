# 04｜MoE 与 LoRA：两种不同的参数效率

本章问题：MoE 省激活计算，LoRA 省更新参数，两者为什么不能互换？
实验入口：`python scripts/lesson_examples.py 04`。

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

## 源码精读：谁被选中，谁收到梯度

<!-- source: laptop_llm/architectures/moe.py::SparseMoE.forward -->

`topk` 的整数索引没有普通梯度，但被选中的权重仍参与计算图。
按专家取 token、执行 FFN、加权散回；一个 token 可分派给多个专家。
router 的辅助项改善负载诊断，不保证最终任务质量。未被激活的专家仍占权重/optimizer 内存。

LoRA 的更新是一个低秩矩阵，不是 token 路由：

<!-- source: laptop_llm/architectures/lora.py::LoRALinear.forward -->
<!-- source: laptop_llm/architectures/lora.py::LoRALinear.merged -->

设输入维 8、输出维 6、rank=2，A 为 `[2,8]`，B 为 `[6,2]`。
`BA` 才与 W 的 `[6,8]` 同形；不要写反乘法。零 B 保证初始化时函数不变，
但第一步 B 能有梯度、A 的梯度可能为零，不应据此判定训练坏了。

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_04 -->

预期初始 LoRA 与 base 相同，改变 B 后合并前后仍相同；MoE router 获得梯度。
完整导入可在 [实验脚本](../../scripts/lesson_examples.py)查看。

## 小结与练习

小结：可训练参数、总参数、每 token 激活参数与通信量是四个不同的量。

1. 为什么 LoRA checkpoint 更小，不代表原始模型推理内存同样下降？
2. 把所有 token 指向同一专家，任务 CE 与负载辅助项分别可能怎样变化？

答案提示：推理仍需要 base W；CE 不直接保证专家负载均匀，应同时记录路由分布。
