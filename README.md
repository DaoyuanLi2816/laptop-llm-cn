# LaptopLLM-CN

一台普通笔记本也能完整跑通的现代中文小型 LLM：从训练自己的 Byte-level BPE 分词器开始，依次完成预训练、监督微调（SFT）、直接偏好优化（DPO）、评测、KV Cache 聊天，以及 OpenAI 兼容的流式 Serving。

它是 `tiny_LLM.py` 的工程化续篇：仍然坚持“代码就是教材、关键处用中文讲透”，但不再把所有内容塞进一个文件，而是把真实项目需要的数据格式、配置、断点、测试和服务边界补齐。

> [!IMPORTANT]
> 仓库附带的 `data/demo` 只用于验证全链路，无法训练出通用智能。聊天能力主要由**有效数据量 × 模型容量 × 训练计算**决定。`smoke` 跑通代表实现正确，不代表模型聪明；要得到更丰富的聊天能力，请换入有许可的高质量中文预训练与对话数据。

## 它包含什么

```text
纯文本 → Byte-level BPE → 预训练 → SFT → DPO → 评测
                                      ↓
                         KV Cache 推理 → CLI / Web / OpenAI API
```

| 模块 | 这个仓库的实现 | 为什么重要 |
|---|---|---|
| 分词 | NFKC + Byte-level BPE | 任意中英文、数字和符号都可编码，不再受手写词表限制 |
| 模型 | RMSNorm、RoPE、GQA、SwiGLU、权重绑定 | 与现代 LLaMA 系 decoder-only 模型同族 |
| Attention | PyTorch SDPA | 自动选择可用的 Flash / memory-efficient / math 后端 |
| 训练 | AMP、梯度累积、裁剪、余弦调度、断点续训 | 在有限显存上稳定训练，并保留完整实验状态 |
| 数据 | memmap packed tokens、assistant-only mask、偏好对 | 避免预训练 padding 浪费与 SFT 误监督用户文本 |
| 对齐 | DPO + 冻结参考模型 | 让 chosen 相对 rejected 更可能 |
| 推理 | prefill + 每层 KV Cache、top-k/top-p、重复惩罚 | 多轮聊天时只计算新 token |
| 服务 | `/v1/chat/completions`、SSE、健康检查、网页 | 现有 OpenAI 客户端可直接连接 |
| 质量 | 单元测试、端到端 smoke、GitHub Actions | 验证训练、cache 等价性、接口与包安装 |

## 5 分钟跑通

Python 3.10–3.12 均可。若使用 NVIDIA GPU，建议先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装与你驱动匹配的 CUDA wheel，再安装本项目，避免 `pip` 意外换成 CPU 版 PyTorch。

```powershell
git clone https://github.com/DaoyuanLi2816/laptop-llm-cn.git
cd laptop-llm-cn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 从 tokenizer 一直跑到 DPO；smoke 只验证全链路
laptop-llm pipeline --config configs/smoke.yaml

# 加载最终 checkpoint 聊天
laptop-llm chat --checkpoint artifacts/smoke/dpo/final.pt

# 固定贪心解码评测，逐条保存成功与失败
laptop-llm evaluate --checkpoint artifacts/smoke/dpo/final.pt `
  --suite data/demo/eval_suite.jsonl --output artifacts/smoke/evaluation.jsonl
```

Linux/macOS 把激活命令换成 `source .venv/bin/activate` 即可。

## 三档配置

| 配置 | 典型模型 | 用途 | 说明 |
|---|---:|---|---|
| `configs/smoke.yaml` | 小于 1M | CI / 排错 | 每阶段几步，只验证所有接口 |
| `configs/cpu.yaml` | 约 7.7M | 纯 CPU 学习 | 能训练，但完整实验需要耐心 |
| `configs/laptop_16gb.yaml` | 约 42M | 12–16GB NVIDIA GPU | bf16 + 梯度检查点；真实语料通常要数小时到数天 |

参数量会随实际 tokenizer 词表大小改变。先执行 `smoke`，再复制配置并一次只扩大一个维度。16GB 档默认仍引用演示数据，是为了开箱不报路径错误；正式训练前必须替换 `data.*` 与 `tokenizer_corpus`。

## 分阶段运行

```powershell
# 1. 训练分词器
laptop-llm train-tokenizer --config configs/cpu.yaml

# 2. 预训练
laptop-llm train pretrain --config configs/cpu.yaml --device cpu

# 3. SFT：从预训练权重开始，但使用新的 optimizer
laptop-llm train sft --config configs/cpu.yaml `
  --init-from artifacts/cpu/pretrain/final.pt --device cpu

# 4. DPO：冻结一份 SFT 模型作 reference
laptop-llm train dpo --config configs/cpu.yaml `
  --init-from artifacts/cpu/sft/final.pt --device cpu

# 本阶段中断后，连同 optimizer 和 step 恢复
laptop-llm train pretrain --config configs/cpu.yaml `
  --resume artifacts/cpu/pretrain/step_0000500.pt
```

每个 checkpoint 内含模型配置、tokenizer、模型权重、优化器、阶段和步数，因此聊天与服务只需一个 `.pt` 文件。`torch.save` 使用 pickle 容器，**只加载你自己生成或明确可信来源的 checkpoint**。

## 本地 Serving

```powershell
laptop-llm serve --checkpoint artifacts/smoke/dpo/final.pt `
  --host 127.0.0.1 --port 8000
```

- 网页聊天：<http://127.0.0.1:8000>
- 交互 API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

OpenAI Python 客户端可把 `base_url` 指向本机：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="laptop-llm",
    messages=[{"role": "user", "content": "请解释 KV Cache。"}],
)
print(response.choices[0].message.content)
```

如果不只监听 `127.0.0.1`，请通过 `--api-key` 或 `LAPTOP_LLM_API_KEY` 设置 Bearer token，并在真实网络环境加入 TLS、限流和反向代理。

## 数据格式

预训练文件是 UTF-8 纯文本，每个非空行视作一个文档；SFT 使用 `messages` JSONL；DPO 使用 `prompt/chosen/rejected` JSONL。完整约定、数据量建议与许可证检查见 [数据说明](docs/data.md)。

```json
{"messages":[{"role":"user","content":"什么是 RoPE？"},{"role":"assistant","content":"RoPE 用位置相关旋转把相对位置信息注入注意力。"}]}
```

```json
{"prompt":[{"role":"user","content":"你会永远正确吗？"}],"chosen":"不会，我有能力边界。","rejected":"会，我永远正确。"}
```

## 仓库地图

```text
laptop_llm/
  config.py       # dataclass + YAML 配置与约束
  tokenizer.py    # BPE、对话模板、SFT/DPO loss mask
  model.py        # RMSNorm / RoPE / GQA / SwiGLU / SDPA / KV Cache
  data.py         # packed memmap、JSONL dataset、padding collator
  engine.py       # 预训练 / SFT / DPO、AMP、调度、checkpoint
  generation.py   # 采样、增量解码、多轮聊天
  server.py       # FastAPI、SSE、OpenAI 兼容 API、内置网页
  evaluation.py   # 固定提示、逐条结果、准确率、延迟与 tokens/s
  cli.py          # train-tokenizer / train / pipeline / evaluate / chat / serve
configs/          # smoke、CPU、16GB GPU 三档实验
data/demo/        # 只用于验证链路的微型中文数据
docs/             # 原理、数据、训练与部署教程
tests/            # cache 等价性、mask、训练与 API 测试
```

## 推荐阅读顺序

1. [模型结构：从一行文字到 logits](docs/architecture.md)
2. [数据与监督边界](docs/data.md)
3. [训练流水线与显存预算](docs/training.md)
4. [推理与服务](docs/serving.md)

底层组件采用 Hugging Face Tokenizers 的可训练分词管线与 [BPE trainer](https://huggingface.co/docs/tokenizers/main/api/trainers)，Attention 使用 PyTorch 的 [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)，混合精度遵循 [`torch.amp`](https://docs.pytorch.org/docs/stable/amp.html) 的 autocast + GradScaler 模式。

## 开发与验证

```powershell
ruff check .
pytest
python -m build
```

本项目的目标是让完整 LLM 生命周期变得可读、可改、可验证。它不是 vLLM、Transformers 或分布式训练框架的替代品，而是理解这些系统之前的一座足够正规的桥。

## License

[MIT](LICENSE)
