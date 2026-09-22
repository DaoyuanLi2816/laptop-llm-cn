# 验证记录：正确性不等于能力

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
