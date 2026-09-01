# 贡献指南

感谢你愿意改进 LaptopLLM-CN。这个仓库首先是一份可运行教材，因此“更容易读懂且行为可验证”与性能同样重要。

## 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
ruff check .
pytest
```

提交前请至少运行：

1. `ruff check .`；
2. `pytest`；
3. 改动训练管线时运行 `laptop-llm pipeline --config configs/smoke.yaml`；
4. 改动 KV Cache 时确认 cache parity 测试仍通过；
5. 改动 API 时同时检查普通与流式响应。

## 代码与文档风格

- 对关键张量写清形状，对容易错的边界解释“为什么”；
- 中文注释保持简洁，不逐字翻译显而易见的代码；
- 新增配置必须能说明目标硬件和能力边界；
- 不提交 checkpoint、token cache、访问令牌或无许可证的数据；
- 性能改动同时报告正确性检查、硬件、dtype、batch 和测量方法。

如果实验没有提升，也欢迎保留可复现的负面结果；它比只展示成功案例更有价值。
