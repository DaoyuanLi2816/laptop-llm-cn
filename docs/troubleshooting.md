# 常见问题与检查顺序

排错先保留命令、完整 traceback、配置、环境和输出目录，不先扩大模型或重复覆盖产物。

## 命令找不到 / 包不是你刚改的版本

从仓库根目录运行 `python -m pip install -e ".[dev]"`，再运行：

```bash
python -c "import laptop_llm; print(laptop_llm.__file__)"
python -m laptop_llm --help
```

确认模块路径来自当前 checkout。Windows 不激活环境也能用 `.venv/Scripts/python.exe`。

## CUDA 不可用或显存不足

先用 `--device cpu` 跑原理实验。PyTorch 安装是否带 CUDA、驱动是否兼容是独立问题。
不需要为学习公式先购买云 GPU；本书没有覆盖全部笔记本型号，也没有实测 Apple MPS。

减少 batch/序列/层数后重新记录实际配置。PPO 同时有多个模型，不能按单模型推理预算选参数。
不要把 Windows 桌面 RTX 4080 的资格验证写成笔记本性能保证。

## 输出目录已存在

`lab` 和实验脚本拒绝覆盖非空目录，这是为了保护失败证据。
换新目录，例如 `artifacts/rl-seed-8`，不要先删除旧实验。
主训练的 `--resume` 与后训练 `lab` 的支持边界见[08 系统](course/08-systems.md)。

## reward 全零，但 loss 还在变化

检查 task objective、reference KL 与 auxiliary loss 各自数值。
全错 GRPO 组优势为零；KL/路由/MTP 仍可给梯度。不能将这种变化报告为 RL 学会解题。
先用 verifier 正例控制，再改善 SFT、问题难度、输出预算与采样探索。

## teacher tokenizer 不匹配

词表大小相同不足够。加载相同序列化 tokenizer 与 chat template 后，再初始化师生模型。
不能“随便换一个 tokenizer”继续旧训练。多教师实验优先使用 Frontier Lab 的共享 tokenizer 流程。

## 网页打开但生成失败

服务是否仍运行？checkpoint 是否可信且路径正确？查看 `/health` 和终端错误。
量化权重使用 `--dtype float32`；它是只读推理产物，不可当训练 checkpoint。
`127.0.0.1` 只在当前电脑有效，GitHub Pages 文档站也不会替你运行模型。

## 文档代码页或内部链接失效

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python scripts/check_docs.py --site site
```

先修复源码符号/链接再提交。站点构建引用固定 commit，网页与仓库 `main` 可能在发布过程中短暂不同步。
文档截图应包含输入、发送、停止或重置等关键控件，不把窄视口的裁切图当成完整界面。
