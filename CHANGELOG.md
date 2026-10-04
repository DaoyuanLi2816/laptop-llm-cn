# 更新记录

## 0.3.0 — Frontier Lab（2026-10-03）

- 模型：KDA 与 NoPE MLA 混合、full/block AttnRes、动态 mHC、learned block sparse indexer、Engram 风格 n-gram 记忆、递归 MTP。
- 训练：矩阵 Muon/AdamW、MTP 预训练标签桥接、三种独立前沿配置。
- 后训练：domain/effort 多教师 OPD、严格工具 agent-GRPO、双侧校正、GAR、router 冻结；候选集合/策略版本数学实验。
- 推理：E2M1/FP32-scale 的 FP4 真打包与重载、精确投机采样；两条参考后端均不宣称生产加速。
- 独立实验：CSA2/CED 共享机制、图像 patch 前缀与梯度，不冒充集成式多模态/百万上下文服务。
- 教材：新增 8 章，合计 18 章；论文版本与阅读范围台账、11 阶段本地 smoke、负结果与实际存储诊断。
- 品牌：原创透明 Logo、README 与本地聊天页更新；旧配置/高精度 checkpoint 保持可加载，FP4 checkpoint 明确只读推理。

## 0.2.0

模块化模型/后训练/系统课程：MLA、MoE、滑窗、PPO/RM、GRPO/RLVR、OPD、LoRA、DDP 与本地 HTTP/SSE；历史验证见 `docs/validation.md`。
