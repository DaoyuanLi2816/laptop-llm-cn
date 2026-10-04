# 18｜从“会运行”到“可信研究”：行为审计与反事实

本章问题：怎样区分模型能力、测试器漏洞和漂亮的解释？
实验入口：`python scripts/lesson_examples.py 18`。

前置：第 10 章实验设计、第 16 章轨迹。阅读：[CHIVE](https://alignment.anthropic.com/2026/chive/)、[GPT-6 Astra](https://deploymentsafety.openai.com/gpt-6-astra)、[6.1 Sol 增补](https://deploymentsafety.openai.com/gpt-6-1-sol)、[MiMo 工具重复诊断](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition)。以下是课程实验设计，**不是本仓库已复现这些评测**。

## 研究员需要证据，不只是一个故事

模型说“因为 X 所以我输出 Y”，并不能证明 X 导致 Y。
CHIVE 将解释与反事实行为联系起来：改变一个因素，再检验实际输出如何变化，而不是用语言流畅度评判解释。
本地可以从确定性实验开始：固定 checkpoint、问题、seed、预算，只改变提示中的一个数字/格式/无关线索。

```text
同一问题 ID
  baseline：原提示 → 原始输出、动作轨迹、verifier 结果
  treatment：只改一处 → 原始输出、动作轨迹、verifier 结果
  paired delta：按问题比较，失败和拒绝也保留
```

注意两种问题不同：

- 改算术数字时，正确答案应随之改变，需使用对应独立 verifier。
- 插入“答案一定是 999”的不可信线索时，正确答案不应改变，可观察诱导敏感性。

贪心只是起点；多次随机采样比较分布时，固定抽样协议、预注册干预，不从很多干预里事后挑漂亮的一组。
没有变化也可能是模型根本不会任务，所以同时需要正控制（真正相关输入）和负控制（无关输入）。

## 排名分数之外，完整行为是什么

OpenAI 的公开 system cards 区分模型行为、部署模拟、监控与安全层，并说明评测版本/harness 会影响结果。
CoT-only、action-only、full-context 监控看到的信息不同；看不到口头风险不表示动作安全。
不能把“未观察到失败”推广成“所有设置可靠”，也不能从 system card 推测未公开的参数量或架构。

把这个原则变成自己的实验记录：

```json
{"problem_id":"heldout-007","checkpoint_sha256":"...","harness":"calculator-v1",
 "policy_version":2,"seed":42,"token_budget":64,"turn_budget":3,
 "answer_correct":false,"schema_valid":true,"tool_calls":2,"stop_reason":"turn_limit",
 "scope_violation":false,"raw_trace_file":"..."}
```

这比一个“综合智能分”有用：错误答案、违规动作、格式错误、预算耗尽是不同故障。
本地 JSON parser 的严格工具边界是 deterministic 防线；不代表神经模型已经学会权限意识。
真实执行 code/browser/shell 时，需要可信进程隔离与权限，不应将本地 calculator 直接换成任意执行器。

## MiMo 的负结果给实验设计的提醒

官方重复调用诊断展示了只奖最终正确、只惩罚超过大阈值时的盲区。
同轮 exact JSON 重复可复查，但它不覆盖跨轮、近似重复或隐藏在代码中的调用，也不等于所有重复都是错误。
本地 agent 一轮只有一个动作，所以不要直接套同轮多调用率；可设计跨轮“同参数、同观察、无新信息”的回放指标，并允许合理失败重试。

先审计 reward：正确答案但没按要求使用工具是否获奖？工具结果与自报文字不一致怎么办？
再审计 harness：更换反馈格式、错误信息或预算，小模型还会不会成功？
最后审计推广：将若干 harness 留作 held-out，不能在同一个 harness 调参后声称跨环境通用。

## 毕业实验报告模板

```text
问题：只改变一个机制，想检验什么假设？
基线：固定 checkpoint、tokenizer、数据切分、harness、采样预算
处理：唯一变化；原理预期与可能的反例
计量：按问题配对、随机种子、成功/失败/拒绝、成本、置信范围
审计：答案泄漏、奖励投机、越权动作、数据/harness 重叠
负结果：零奖励/不显著/速度变慢，解释哪些假设被否定
边界：单设备数值实验、特定任务能力、跨域能力，分别说到哪一级
```

推荐先做架构因果/梯度测试，再做固定算术 held-out，最后做合法开放数据任务。
别跳过两个现实：小数据小模型可能不产生有用语言；工业系统的可靠性、成本与安全会改变设计，不只是 scale。
本仓库的“掌握”标准是能复现、解释、反驳和改进一个实验，不是背完最新论文名字。

## 源码精读：答案与实际事件分别验证

<!-- source: laptop_llm/posttraining/agent.py::ArithmeticTask.verify -->

正确数字不足以证明工具调用发生；`used_matching_tool` 来自可信环境事件，而不是模型说
“我调用了 calculator”。标准答案只在 verifier 中计算，不进入可见 prompt。
这防止“口头报告行为”变成行为证据，但仍不能证明模型的口头 CoT 是真实因果过程。

构造一对只改权限或工具可用性的样例，使用相同 checkpoint、预算、seed 和验收器。
保留所有轨迹，包括失败和奖励全零组。另用直接干预隐藏状态的研究工具时，也需报告其假设和局限。
本地没有复制任何实验室内部解释系统，阅读范围见[论文台账](../frontier-papers.md)。

## 可运行小实验

<!-- source: scripts/lesson_examples.py::lesson_18 -->

预期合法答案正例得 1，文本中夹带数字得 0；同一个正确答案，无实际匹配工具事件得 0，有事件得 1。
这是验收器/协议的正例控制，不是模型通过了任务。结果最后明确打印 `learned_agent_capability=not_measured`。

## 小结与毕业练习

1. 增加“模型声称执行但事件缺失”的攻击样例，写出应该返回 0 的原因。
2. 将你的一项架构消融整理为报告：固定来源 → 配对样例 → 机制不变量 → 质量指标 → 成本指标 → 负结果。
3. 挑选一份上游报告，指出本地教材、论文方法和生产系统之间各一个不能只靠 scale 消除的差异。

答案提示：由可信事件验收行为；不要混合成本和质量；缺失的通信、数据治理或工具权限边界不是增大参数即可解决。
完成后回到[源码地图](../source-map.md)，确认每个结论都能定位到代码与实验。
