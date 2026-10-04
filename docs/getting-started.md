# 安装与第一次实验

目标：不下载上游权重，先验证环境、理解一次前向，再跑自己的训练链路。
以下命令从仓库根目录执行。Python 3.10+；推荐先用 CPU。

## 1. 建立独立环境

```bash
git clone https://github.com/DaoyuanLi2816/laptop-llm-cn.git
cd laptop-llm-cn
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

PowerShell 若阻止激活，不必全局降低执行策略：直接用 `.\.venv\Scripts\python.exe`
替换命令中的 `python`，用 `python -m laptop_llm` 替换 `laptop-llm`。

```bash
python -m pip install -e ".[dev]"
python -m pytest
python scripts/lesson_examples.py 01
python scripts/lesson_examples.py 02
```

第 01 个实验打印右移标签；第 02 个实验显示 logits 形状并检查 loss/梯度有限。
loss 的具体小数不是跨设备承诺。CUDA 不可用时相应测试跳过，不算失败。

## 2. 训练最小中文模型

```bash
python -m laptop_llm pipeline --config configs/smoke.yaml --device cpu
```

它依次建立 BPE、预训练、SFT、DPO。`artifacts/smoke/` 保存 tokenizer、指标与 checkpoint。
先读 [01 数据](course/01-foundations.md)和 [05 SFT/偏好](course/05-alignment.md)，
不要把同一个“loss”误认为每个阶段都在优化同一件事。

## 3. 打开自己的网页

```bash
python -m laptop_llm serve --checkpoint artifacts/smoke/sft/final.pt --device cpu
```

访问 `http://127.0.0.1:8000`，接口说明在 `/docs`。停止终端中的进程后服务也停止。
`127.0.0.1` 是本机地址，不是别人能访问的公开链接。小模型的随机回复不是训练成功的能力证据。

![完整本地聊天界面：输入框、发送和新对话按钮都在截图中](assets/frontier-demo.jpg)

## 4. 按目的升级实验

| 想理解什么 | 入口 | 检查什么 |
|---|---|---|
| 全部后训练角色 | `python scripts/lab_smoke.py --output artifacts/first-lab --device cpu` | 8 阶段、冻结角色、原始 rollout |
| 前沿架构与机制 | `python scripts/frontier_smoke.py --output artifacts/first-frontier --device cpu` | 11 阶段、FP4 存储、投机等价、零奖励 |
| 每章的原理 | `python scripts/lesson_examples.py all` | 18 个小实验的断言，不训练大模型 |

每次实验使用新目录；不要覆盖失败记录。GPU 与笔记本资源边界见[常见问题](troubleshooting.md)。

## 5. 本地读文档站

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve --dev-addr 127.0.0.1:8001
```

访问 `http://127.0.0.1:8001`。文档构建不调用模型，不需要 GPU；网络只用于安装依赖。
离线可直接读仓库 Markdown。静态站包含全文索引；正文、字体与代码高亮不依赖外部 CDN。
