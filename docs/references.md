# 公开源码阅读索引

调研日期：2026-09-21。这里列出实际查阅的公开入口及本仓库教学映射。
固定 commit 的链接用于可复查阅读；上游 main 会持续演进。本项目为独立教学实现，
没有把参考仓库整体复制进来，也不宣称兼容它们的 checkpoint 或性能。

| 参考 | 查阅入口 | 应带着什么问题读 | 本地映射 |
|---|---|---|---|
| MiniMind | [README 与训练目录](https://github.com/jingyaogong/minimind/tree/1e6e909f887a442c4df7797b6fcf7464c374a4a2)；[train_grpo.py](https://github.com/jingyaogong/minimind/blob/1e6e909f887a442c4df7797b6fcf7464c374a4a2/trainer/train_grpo.py) | 低门槛入口如何组织？分组奖励怎样进入目标？ | README、lab_smoke、GRPO |
| DeepSeek-V3 | [inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py) | MLA、专家、并行线性层与 RoPE 各承担什么？ | model、mla、moe、系统课 |
| DeepSeek-V3.2-Exp | [inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py) | learned indexer 与实际 sparse attention kernel 的边界？ | sparse 课中的 DSA 差异 |
| verl | [core_algos.py](https://github.com/verl-project/verl/blob/8e03c039f9a70750490c93196b759b332fc029fd/verl/trainer/ppo/core_algos.py) | 优势估计、mask 与 reduction 怎样解耦？ | objectives、rollout、trainer |
| TRL | [GKD trainer](https://github.com/huggingface/trl/blob/5bb945c38f319259eddaad7bb3838e4aaf590c49/trl/experimental/gkd/gkd_trainer.py)；[官方文档](https://huggingface.co/docs/trl/gkd_trainer) | 谁生成轨迹？JSD/KL 与 teacher logits 如何对齐？ | OPD 全词表 KL |
| Qwen3 | [官方仓库](https://github.com/QwenLM/Qwen3) | reasoning 模式与模型/模板的关系；权重系列的不同选择 | Decoder、CoT 与使用边界 |
| Megatron-LM | [官方仓库](https://github.com/NVIDIA/Megatron-LM) | 模型计算如何映射到通信组与并行维度？ | DDP 实验＋多维并行阅读 |
| SGLang | [官方仓库](https://github.com/sgl-project/sglang) | scheduler、cache、attention backend 怎样分层？ | serving 课与未实现项 |

## 论文与官方概念入口

- [PPO](https://arxiv.org/abs/1707.06347)：概率比与 clipped surrogate。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 的组内相对优势。
- [On-Policy Distillation / GKD](https://arxiv.org/abs/2306.13649)：学生生成分布与教师软监督。
- [LoRA](https://arxiv.org/abs/2106.09685)：低秩增量适配。
- [InstructGPT](https://arxiv.org/abs/2203.02155)：示范、偏好与 RLHF 的公开研究路线。

这些是概念来源，不是统一“所有 SOTA 标配清单”。框架支持某个算法，不意味着某个
闭源模型必然采用该算法。也不能从开放推理代码反推完整训练数据与内部平台。

## 建议的阅读方法

先在本仓库画 shape 与梯度，再到固定源码定位同类函数；逐项记录不同的 mask、
归一化、rollout backend、并行通信和数值精度。看到名称相同不要立即认定公式相同。
特别注意本地 sliding 不是 DSA、本地带辅助项 MoE 不是 auxiliary-loss-free 路由、
本地同步 OPD 不是完整 TRL/GKD 的全部模式。

代码、模型权重和数据可能有不同许可证。复用任何上游资产前单独检查其授权，
本项目 MIT 不能替代上游授权，也不能替数据中的隐私内容提供使用许可。
