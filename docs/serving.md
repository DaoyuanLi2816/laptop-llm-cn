# 推理与 Serving

## prefill 与 decode

第一次前向把整个 prompt 送入模型，得到每层所有历史 token 的 K/V，这一步叫 prefill。之后每生成一个 token，只输入这个新 token，并复用 cache，这一步叫 decode。

第 `t` 步若每次完整重算，需要重复生成前缀所有层的投影和 FFN；使用 KV Cache 后，只需为新位置计算这些部分。Attention 仍需让新 Query 读取全部历史 Key/Value，所以长上下文成本不会完全消失。

## 采样旋钮

- `temperature=0`：贪心，最稳定，容易重复；
- `temperature≈0.7–0.9`：聊天常用起点；
- `top_k`：只保留最高的 K 个候选；
- `top_p`：保留累计概率达到 p 的最小集合；
- `repetition_penalty`：降低已经出现 token 的再次概率。

比较模型时必须固定这些参数，否则“模型变化”和“采样变化”无法区分。

## OpenAI 兼容范围

已实现：

- `GET /v1/models`
- `POST /v1/chat/completions`
- 普通 JSON response
- `stream=true` 的 SSE chunk 与 `[DONE]`
- `temperature/top_p/max_tokens/seed`

没有实现：tool calls、logprobs、response format、并行 choices、图像/音频输入。客户端若依赖这些能力，应显式关闭。

## 并发边界

本 server 面向单用户笔记本。生成阶段用锁串行化，避免多个请求同时占满 GPU 显存。生产级系统通常还需要 continuous batching、KV 分页、请求调度、取消与超时；这些是 vLLM/SGLang 一类框架的工作范围。

## 网络安全

默认 `127.0.0.1` 只供本机访问。若改成 `0.0.0.0`：

1. 设置 `--api-key`；
2. 通过反向代理提供 TLS；
3. 配置防火墙、限流、请求体上限与日志脱敏；
4. 不要把训练 checkpoint 或日志目录作为静态文件暴露。

API key 只是最小保护，不是完整公网部署方案。
