# Frontier Lab：不买云资源的实验路线

从仓库根目录执行。Python 3.10+，安装 `python -m pip install -e ".[dev]"`。
这些程序不请求云 API、不下载上游权重；模型随机初始化，短训练通常产生无意义输出。
先理解链路，再换合法数据、扩大预算、设置 held-out 评测。

## 一条命令：11 个训练阶段与两项推理检查

```bash
python scripts/frontier_smoke.py --output artifacts/my-frontier --device cpu
```

三种架构各有 pretrain/SFT/DPO（9 个阶段），再运行 MOPD 与 agent-GRPO。
每阶段默认 2 步；脚本把训练窗口缩到 192、所有模型共享 768 目标词表的 BPE。
原配置的 512 上下文没有被当作 CPU 性能承诺。可加 `--device cuda` 或 `--steps 10`；仍不是能力评测。

```text
artifacts/my-frontier/
  tokenizer.json                 # 师生逐字节相同的模板与词表
  resolved-configs.json           # 实际运行的配置，不能只引用默认 YAML
  hybrid/{pretrain,sft,dpo}/      # KDA/MLA＋AttnRes＋MoE＋Engram＋MTP
  indexed/{pretrain,sft,dpo}/     # 组级 learned block indexer＋MoE
  mhc/{pretrain,sft,dpo}/         # 多残差流＋Sinkhorn＋MoE
  teachers.json                  # domain:effort → 教师 checkpoint
  mopd/{run,metrics,rollouts}.*   # 不丢弃学生失败前缀
  agent-grpo/                    # 原始动作、工具事件、奖励、版本、checkpoint
  hybrid-fp4.pt                  # 只读推理格式，不能用它 resume 训练
  report.json                    # hash、设备、零奖励、存储字节、RMSE、投机等价
```

MOPD 教师只是小模型的不同 SFT checkpoint；没有证据表明它们是强教师。
agent 生成使用原始温度 1 softmax，不使用 top-p，因此与训练全词表 logp 对齐。
脚本检查投机**贪心结果**与目标模型相同，不把 acceptance rate 当吞吐改善。

## 分别观察三个模型，而不是一次打开全部开关

```bash
laptop-llm pipeline --config configs/frontier_hybrid.yaml --device cpu
laptop-llm pipeline --config configs/frontier_indexed.yaml --device cpu
laptop-llm pipeline --config configs/frontier_mhc.yaml --device cpu
```

三个命令各自训练 tokenizer，故它们的结果**不能直接用于跨模型蒸馏**。
跨模型实验使用上面的共享 tokenizer 脚本；词表大小相等仍不足以保证 token 意义相同。
基础 `configs/smoke.yaml` 是更快的初学入口，`lab_smoke.py` 保留 PPO/RM/GRPO/OPD/LoRA 全课程。

## 后训练与服务

```bash
laptop-llm lab mopd --checkpoint artifacts/my-frontier/hybrid/sft/final.pt --data artifacts/my-frontier/mopd.jsonl --teachers artifacts/my-frontier/teachers.json --output artifacts/another-mopd --steps 2 --max-new-tokens 8
laptop-llm lab agent-grpo --checkpoint artifacts/my-frontier/hybrid/sft/final.pt --data artifacts/my-frontier/agent.jsonl --output artifacts/another-agent --steps 2 --max-new-tokens 8 --freeze-router --policy-objective calibrated --gar
laptop-llm serve --checkpoint artifacts/my-frontier/hybrid-fp4.pt --device cpu --dtype float32
```

打开 `http://127.0.0.1:8000`。Logo、HTML、JS 都由本地服务提供；支持流式、多轮历史和参数调节。
`/health` 显示架构/残差/专家数，`/docs` 提供 API 文档。
网页只接受文本聊天，**不支持图像上传、agent 工具自动执行或投机 serving**；相应模块是独立实验。
FP4 推理按需反量化矩阵，可能比高精度更慢。公开网络部署仍需 TLS、隔离、限流与审计。

## 低精度与投机解码的独立入口

```bash
laptop-llm quantize --checkpoint artifacts/my-frontier/indexed/sft/final.pt --output artifacts/indexed-fp4.pt
laptop-llm speculate --checkpoint artifacts/my-frontier/hybrid/sft/final.pt --draft artifacts/my-frontier/indexed/sft/final.pt --prompt "3+4=?" --max-new-tokens 8 --temperature 0
python -m pytest tests/test_frontier_mechanisms.py tests/test_quantization_speculative.py
```

量化模块包含 STE 函数的 forward/backward 实验，不包含完整大模型 QAT 训练配方。
CSA2/CED 与图像 patch 前缀以测试为入口，不能凭测试通过宣传为百万上下文或视觉聊天能力。

## 资源预算与复现实验

CPU 小配置可验证公式；CUDA 小配置验证设备路径，不代表在每台笔记本上的速度。
训练内存包含权重、梯度、optimizer、激活；PPO 还同时驻留多个模型。
混合注意力仍有全局层，KV 随长度增长，不能宣传整个模型“常数 cache”。
实际资格验证与负结果见[验证记录](validation.md)。输出非空时实验脚本拒绝覆盖；比较结果时保存配置、数据/权重 hash 和所有失败轨迹。
