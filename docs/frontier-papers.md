# 前沿论文台账：读到了哪里，代码实现到哪里

核验日期 **2026-10-03**。只以作者论文、官方技术报告、模型仓库和实验室文章为依据。
本页不是全球模型排名，也不把“最近发布的产品”自动等同于“公开完整训练方案”。
论文版本固定为已阅读版本；发布日期与报告上传日期可能不同，不从 arXiv 编号推断发布日期。

## 本轮阅读与实现映射

| 团队 / 资料 | 版本或发布日期 | 本轮重点阅读 | 本地对应 / 未覆盖 |
|---|---|---|---|
| Kimi — [K3: Open Frontier Intelligence](https://arxiv.org/html/2607.24653v1) | v1，2026-07-27；[发布页](https://www.kimi.ai/blog/kimi-k3) | §2.1–2.2 KDA/AttnRes；§4.1.2–4.1.4 后训练 | `delta.py`、`residual.py`、hybrid；MOPD 仅路由机制，不是论文 sampled-token RL 目标；未实现 Stable LatentMoE/PerHeadMuon/SiTU |
| DeepSeek — [V4 报告](https://arxiv.org/html/2606.19348v1) | 阅读 v1；[摘要页](https://arxiv.org/abs/2606.19348)显示初次提交 2026-04-26 | 架构与 mHC | `ManifoldConnection` 的 A/B/C 与 Sinkhorn；全 Transformer block 为本地 F，非生产逐子层布局 |
| DeepSeek — [V4.1-Flash 报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | [官方发布](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)，2026-09-10 | PDF p1–4、7、9–14、30–31：CED、CSA2、Engram、低精度/训练系统 | `compression.py` 仅均值压缩/共享代数实验；无完整 CED、层次索引、Bounded Replay、SinglePass mHC、DSpark |
| Z.ai — [GLM-5](https://arxiv.org/html/2602.15763v1) | v1，2026-02-17；[当前模型仓库](https://github.com/zai-org/GLM-5)已包含 5.2 | §3.6、§4.1：基础设施、异步 RL、TITO 与双侧校正 | `agent.py` 原始 token/logp；`frontier.py` 校正/版本 mask；**未实现异步执行引擎**。5.2 发布不意味着本地复现 5.2 全训练配方 |
| Xiaomi — [MiMo V2.6 报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf) | [官方发布入口](https://mimo.xiaomi.com/mimo-v2-6)，2026-09；PDF 文件版本校验见下 | PDF p18、23–24、32：GAR、router 冻结、训练/采样一致性 | GAR 数学参考、router 冻结与候选集重归一化函数；无 agentic grader、R3 路由重放、MOPD2 teacher prefix、多模态 RL 集群 |
| Xiaomi — [工具重复调用诊断](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition) | 2026-09-27 | 重复定义、奖励阈值盲区、同样例跨 checkpoint 回放 | 第 18 章研究练习；本地工具质量因子为有界轨迹长度诊断，不冒充论文 grader |
| MiniMax — [Sparse Attention](https://arxiv.org/html/2606.13392v1) | v1，2026-06-11；[M3 发布](https://www.minimax.io/blog/minimax-m3) | §3.1–3.3：block max、GQA 组索引、KL、detach、warmup | `indexed.py` 真实选块/gather；warmup 是显式开关，不是训练引擎自动调度；无 TileLang kernel |
| Google DeepMind — [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770v1) | v1，2026-07-02 | §2 架构、encoder-free 图像入口、QAT 与草稿解码 | `multimodal.py` 图像 patch 前缀实验；未实现共享 KV/K=V、pRoPE、音视频；本地 FP4 不是其 int2/int4 配方 |
| OpenAI — [gpt-oss model card](https://arxiv.org/html/2508.10925v1) | v1，2025-08；故意保留的开放权重架构对照，不称为最新产品 | 架构与开放部署边界；[官方发布](https://openai.com/index/introducing-gpt-oss/) | GQA、MoE、混合局部/全局注意力的比较；不兼容 gpt-oss 权重或 Harmony 模板 |
| Thinking Machines — [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 2025-10-27 | Implementation 与持续学习讨论 | 本地 OPD 是全词表 reverse KL；文章实际实验用 sampled-token 概率与 RL 更新，不宣称逐项复现其结果 |
| Muon 作者 — [Muon is Scalable for LLM Training](https://arxiv.org/html/2502.16982v1) | 阅读 v1，2025-02 | §2 矩阵更新、Newton–Schulz、分布式组织 | `optim.py` 单设备矩阵 Muon＋AdamW 回退；非分布式 Muon/PerHeadMuon |
| Anthropic — [CHIVE](https://alignment.anthropic.com/2026/chive/) | 2026-08-21 | 反事实干预、行为评估与解释局限 | 第 18 章配对干预设计；未实现内部解释工具，不把模型口头解释视为因果证据 |
| OpenAI — [GPT-6 Astra System Card](https://deploymentsafety.openai.com/gpt-6-astra)、[6.1 Sol 增补](https://deploymentsafety.openai.com/gpt-6-1-sol) | 2026-09-03 / 09-29 | Alignment、Monitorability、部署模拟与评测版本差异 | 第 18 章轨迹审计/权限边界/固定 harness；不推测闭源架构，不调用付费模型 |
| Anthropic — [Opus 5.5 官方发布](https://www.anthropic.com/claude-opus-5-5)、[system card 目录](https://www.anthropic.com/system-cards) | 2026-09-22 | 官方发布中的行为审计与评测 caveat | 第 18 章阅读比较；system card 下载端点因体积超限未完整解析，**不声称精读该卡**；官方目录非完整产品全集 |

## 可复查的 PDF 指纹

PDF 在实验目录仅作阅读输入，不随本仓库再分发。页码指 PDF 物理页，从 1 开始。

```text
DeepSeek_V41_Tech_Report.pdf (51 pages)
sha256 ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d
MiMo_V2_6_technical_report.pdf (44 pages)
sha256 fb81e6e083801b3358f084ed6be953dc23b0d2e434690f4541d5eae03e01e7af
```

上游 `main` PDF 可能变动；请对照 hash，不把同文件名当成同版本。
KDA/AttnRes 的前序资料：[Kimi Linear](https://arxiv.org/abs/2510.26692)、[Attention Residuals](https://arxiv.org/abs/2603.15031)。本轮前序条目核验摘要与谱系，主要公式依据 K3 正文。

## 怎样使用台账

先打开本地实现，标出 shape、梯度、cache 生命周期，再阅读对应方法章节。
给每个差异写一个可证伪测试：未来 token 改动是否影响过去？改变采样候选集是否改变 logp？
加入 FP4 后量的是文件、驻留张量，还是峰值显存？每次升级台账必须更新日期与阅读范围。
