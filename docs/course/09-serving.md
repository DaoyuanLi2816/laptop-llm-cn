# 09｜推理、网页与生产服务的距离

## 本地可用路径

```bash
laptop-llm serve --checkpoint artifacts/smoke/sft/final.pt --device cpu
```

浏览器打开 `http://127.0.0.1:8000`。页面纯 HTML/CSS/JavaScript，不依赖 CDN、构建工具
或云模型。显示实际设备、唯一参数量、上下文长度，支持采样温度、最大输出长度与新对话。
启用 `--api-key` 时可在页面输入；key 不写 localStorage。

`/v1/models`、`/v1/chat/completions` 提供兼容风格子集，不是完整外部 API 实现。
没有 tools、图片、logprobs、多候选、生产审核等能力。SSE 以 `data: ...` 帧发送文本，
最后 `[DONE]`；停止原因区分模型 EOS 与长度耗尽。

## Prefill 与 decode

prefill 并行计算 prompt，每层得到历史 KV；decode 每步只前向新 token。
prefill 通常更像矩阵计算，decode 常受权重读取与 KV 带宽影响。
同样 tokens/s，长 prompt 的首字等待可能很差，所以至少区分：

- TTFT：请求至首 token 的时间（含排队、分词、prefill）。
- TPOT/ITL：后续 token 间延迟。
- Throughput：稳定负载下所有请求总吞吐。
- 峰值内存：权重、KV、临时 buffer 与 allocator。

本仓库的生成评测总时长含 prefill，不冒充纯 decode kernel 吞吐。

## 对话历史和字节流

历史是每次请求重新构造的 token 前缀；服务没有跨请求会话 KV 复用。
超长时删除中间最老历史，保留 system 与最新消息；若必需消息本身超长，返回明确错误，
不把用户问题悄悄丢掉再生成。输出 token 上限受剩余上下文约束，max_tokens 是上限不是保证数量。

中文字符可能跨多个 byte token。逐 token decode 直接显示会出现替换字符，
所以累计解码后只推送稳定增量。SSE 网络 chunk 也不等于完整事件，前端必须保留未完成 buffer。
网页使用 textContent 显示模型文本，不能把模型输出直接塞进 innerHTML。

## 当前服务的限制

全局锁串行生成，适合个人本机试聊。一个慢请求会阻塞其他请求；没有有界队列、
完善的取消调度、batch scheduler、速率限制与生产监控。
默认只监听 localhost。不要以“设置 API key”为由直接暴露到公网；还需要 TLS、
网关认证、请求体限制、并发预算、日志隐私与攻击面审计。

## 生产优化阅读路线（未实现）

**Paged KV：** 逻辑 token 到物理 block 的间接映射，减少连续分配碎片；需要 block table、
引用计数、生命周期与 kernel 配合，不只是把 Python list 改成 dict。

**Continuous batching：** 每次 decode 调度可加入新请求、移除完成请求。吞吐改善可能牺牲
部分请求延迟，需要负载与 SLO 指标。本服务的锁不是 continuous batching。

**Prefix caching：** 共享完全一致的前缀计算；必须匹配权重、tokenizer、位置、adapter 等。
模型版本不同但 prompt 字符串相同不够。

**Speculative decoding：** draft 提议、target 验证与校正；正确的采样校正才可能保持目标分布。
直接接受 draft 的 token 不能宣称与 target 无损等价。draft 成本、接受率和 batch 状态决定是否加速。

**量化：** weight-only INT8/INT4 与 KV 量化不是同一件事。需要校准/尺度、算子支持与精度回归，
FP16 推理也不能冒称 INT4。教学 repo 当前未提供量化导出或 GGUF 权重。

从 [SGLang](https://github.com/sgl-project/sglang) 读 scheduler/cache 与后端边界，再回来看
这个单请求循环，就能指出缺了什么。目标是理解差距，而不是掩盖差距。
