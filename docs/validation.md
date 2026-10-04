# 验证记录：正确性不等于能力

## v0.3 前沿实验室（2026-10-03）

本地 Windows/Python 3.10、PyTorch 2.13.0+cu130，CUDA 为 NVIDIA GeForce RTX 4080。
这是桌面 GPU 路径验证，**不是笔记本实测，也不是独立能力评测**。CPU 不依赖 CUDA。

| 检查 | 观察 / 边界 |
|---|---|
| 本地测试 | 97 项通过，包含 6 项 CUDA；CPU CI 应为 91 项通过、6 项跳过。仅有现有 FastAPI/Starlette testclient 弃用提醒 |
| Frontier CPU | 三种架构各 pretrain/SFT/DPO＋MOPD＋agent-GRPO，共 11 阶段，各 2 步；路由修复后复跑脚本耗时约 20.5s |
| Frontier CUDA FP32 | 同流程约 61.0s；小矩阵/Python 循环下 GPU 更慢，**没有加速结论** |
| 模型规模 | hybrid 354,512、indexed 286,496、mHC 295,820 参数；实际窗口 192，共享 tokenizer hash `68b9bcf99c14029e4d594683a892a23cd0b542e945c8596f3284cec61776a660` |
| FP4 存储 | hybrid 模型唯一 storage：1,418,048 → 366,992 bytes（约 3.86×），不包含 optimizer，不是文件/峰值显存指标 |
| FP4 数值误差 | 固定 prompt 全位置 logits RMSE：CPU 0.05832067，CUDA 0.05832062；不是量化无损或能力保留结论 |
| 投机解码 | 两种架构之间的目标贪心 8 token 与普通目标解码相同；本例 6 个草稿接受、2 次目标前向＋bonus；不证明一般接受率或速度 |
| Agent 负结果 | CPU/CUDA 两步 reward_mean 都为 0，所有组零方差；无任务学习收益证据。KL/辅助项仍可更新网络 |
| 原理实验 | CSA2 共享对象/因果性、视觉前缀 CE backward、FP4 STE 梯度通过；这些实验未接入完整 CED、多模态聊天或生产 QAT |
| 包与网页 | sdist/wheel 构建、从仓库外的独立 `--target` wheel 安装目录加载 FP4 checkpoint 并启动 HTTP 服务；浏览器实测 Logo、发送、SSE 结束/按钮恢复、新对话，无页面 JS error/warn |
| 编码兼容 | 强制 `PYTHONIOENCODING=cp1252`、`PYTHONUTF8=0` 后运行完整 frontier smoke 成功；入口显式配置 UTF-8 |
| 旧课程回归 | v0.3 代码重新跑通 `lab_smoke.py` 的 8 个阶段，PPO/RM/LoRA 等路径仍可用 |

数值测试还覆盖：KDA 单步 oracle、cache 多 token 追加/不可变分支、AttnRes activation-checkpoint 梯度等价、mHC 行列归一化、索引 KL 梯度隔离、MTP 标签偏移与真实预训练梯度、Muon state 重载、双侧校正原分母/GAR/hack 筛除。
后训练日志在 v0.3 将总辅助项改名 `auxiliary_loss`，并保留原始 `input_ids`、`action_mask`、`behavior_logp`、`terminal_mask`、版本与 agent 事件。

首轮远程 v0.3 CI（`8605082`）的 Linux 任务通过，两个 Windows 任务暴露索引
缓存路由不一致：同 token 分数约 `1e-9` 的 GEMM 舍入差异导致 Top-k 换块，
最大 logits 差约 `0.0285`。本地 alternate CPU arithmetic 复现后，改为逐 token
canonical 索引投影、固定维点积和稳定同分排序；保留原严格误差阈值，并增加
回归与 Windows CI 后端检查。原失败记录保留，不将第一轮失败覆盖成成功。

复查：

```bash
python -m pytest
python scripts/frontier_smoke.py --output artifacts/frontier-recheck --device cpu
python scripts/frontier_smoke.py --output artifacts/frontier-recheck-cuda --device cuda
```

每次用新目录。阅读 `report.json`、实际配置与完整失败轨迹；不同 PyTorch/设备/后端产生的采样轨迹可能不同，hash 不能当跨平台必然相同的断言。
原论文阅读范围及差异见[台账](frontier-papers.md)。远程 CI 以当前提交的 Actions 为准，不把过去的绿灯算作本次证据。

## v0.2 历史验证（2026-09-21）

本次 v0.2 本地验证日期：2026-09-21。开发机器为 Windows、Intel i7-13700KF、
RTX 4080 16GB，Python 3.10、PyTorch 2.13.0+cu130。这是桌面机测量，不能冒称笔记本实测。
CPU 路径不要求 NVIDIA；Apple MPS 未在本次实机验证。

本次最终本地测试包含 47 项 CPU/通用测试与 3 项 CUDA 混合精度测试；CPU CI 会明确跳过后者。

## 已执行的验证

| 范围 | 命令/方法 | 观察 |
|---|---|---|
| 数值与集成 | `python -m pytest` | sparse dense-oracle、MLA 吸收/缓存等价、MoE 梯度、LoRA 合并、RL 目标、训练更新、HTTP |
| 基础中文链路 | `pipeline --config configs/smoke.yaml --device cpu` | tokenizer → pretrain → SFT → DPO 完成 |
| MLA 配置 | `pipeline --config configs/research_mla.yaml --device cpu` | 246,400 参数，三阶段与重算训练完成 |
| Hybrid MoE 配置 | `pipeline --config configs/research_moe.yaml --device cpu` | 820,928 参数，三阶段与重算训练完成 |
| 八阶段课程 | `scripts/lab_smoke.py`，分别 CPU/CUDA fp32 | 138,560 参数，所有阶段保存权重并可重新加载 |
| 分布式梯度 | `python scripts/ddp_lesson.py` | 两进程 gloo 与单进程全局 batch 梯度在容差内一致 |
| 网页 | 本机浏览器访问 localhost | 设备信息、发送、实际响应、按钮恢复、新对话通过；无 JS console error |

网页实际生成的是 smoke 随机文本，没有把漂亮 UI 当作聊天质量。已修复旧页面内嵌 Python
字符串转义造成的 JS 换行问题，并加入回归测试；页面文字不通过 innerHTML 执行。

第一次远程 Windows CI 暴露了英文 cp1252 管道无法打印中文日志的问题；CLI 与课程脚本
入口现统一 UTF-8，增加强制 cp1252 环境的 subprocess 回归测试，不只依赖本机中文 locale。

## 必须保留的负结果

CPU 与 CUDA 的两步 GRPO 实验 `reward_mean=0`，`zero_variance_groups=1`。
这表示当前极小学生没有产出合格算术答案，组内相对优势为0，**没有任务学习收益证据**。
MoE 的 auxiliary loss 仍可能改变权重，不能把这些变化归因于 RL 学会了推理。
日志拆分 `objective_loss` 与 `router_auxiliary_loss`，原始输出全部保留。

两步奖励模型的偏好拟合也不构成有用奖励器；PPO 只是验证冻结、采样、GAE、更新与保存。
OPD 只证明不同阶段权重的分布可以进行蒸馏，不证明教师强或学生通用能力提升。

## 如何自己复查

```bash
python -m pytest
python scripts/lab_smoke.py --output artifacts/validation-new --device cpu
python scripts/bandit_lesson.py
python scripts/ddp_lesson.py
python -m build
```

两动作 bandit 是控制良好的目标函数学习正例；不是 LLM 推理结果，不能与零奖励语言
rollout 混在一张“能力提升”表里。准确区分正例/负例是课程的一部分。

默认 seed=7 的 bandit 实测：高奖励动作概率 `0.5000 → 0.9802`。

每次使用全新输出目录。检查 `run.json` 的数据 SHA256、设备、种子与训练参数，
`metrics.jsonl` 的数值与 `rollouts.jsonl` 的失败输出。checkpoint 与训练文件不上传 Git，
避免误提交权重、个人数据或大体积缓存。

CI 从 GitHub 的 [Actions](https://github.com/DaoyuanLi2816/laptop-llm-cn/actions) 查看当前结果，
不要把本地通过或过去某个 commit 的绿色徽章冒充最新提交的验证结果。
