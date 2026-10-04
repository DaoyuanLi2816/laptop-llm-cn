# 17｜低精度、精确投机与多模态入口

前置：第 09 章推理、基础概率。源码：`quantization.py`、`speculative.py`、`multimodal.py`。
部署三个独立问题：权重怎样存、输出怎样采样、输入怎样表达。下面三条路径各自有测试，不能互相替代能力证据。

## FP4 不只是一行 `.half()`

本地格式是 E2M1 的 16 个有符号码（含正负零），每 32 元素一组 FP32 scale，两个 4-bit code 存一个 uint8。
scale=amax/6；选择最邻近幅值、记录符号、打包，再按原 shape 反量化。拒绝 NaN/Inf，补齐尾块。

```text
两项 code a,b → byte = a | (b << 4)
byte → low = byte & 15；high = byte >> 4
值 = signed_E2M1[code] * group_scale
```

这是真的 byte packing，不是把整数值放进 float tensor。
但 FP32 scale 与本地扁平分组不同于 MXFP4/NVFP4；也不同于 V4.1 的 KV scale 格式。
`PackedLinear` 前向将整个矩阵反量化后做普通 F.linear；节省驻留编码存储，不承诺峰值显存或吞吐改善。
shared embedding/lm_head 不量化，bias/norm 等仍高精度；模型不能期待理想 8× 整体压缩。

量化 checkpoint 是只读推理产物，载入时先依 metadata 替换模块，再读 state；训练拒绝用它初始化/resume。
比较压缩率应统计两侧**模型 storage**，不能把含 optimizer/reference 的训练文件与只含模型的推理文件比较。
同时记录 logit RMSE、生成变化与独立任务质量，压缩率不证明精度无损。

`fake_quantize_fp4(W) = W + stopgrad(QDQ(W)-W)`：前向量化、反向近似恒等梯度。
这提供 STE 算子，不包含完整模型 QAT 训练、rollout 权重 QDQ 同步或低比特 matmul kernel。

## 精确投机采样：草稿快不等于接受得多

草稿 q 提议一个 token x，目标 p 一次验证多个位置。接受概率 `min(1,p(x)/q(x))`；拒绝后从 `(p-q)_+` 归一化采样替代 token，并丢弃后续草稿。
若全部接受，再从目标最后位置采样 bonus token。

手算 p=[0.6,0.4]、q=[0.2,0.8]：直接接受质量为 [0.2,0.4]，拒绝质量为 0.4，修正分布为 [1,0]，总输出仍为 p。
不要拒绝后从 p 直接重新采样：那会改变输出分布。
相同 p/q 不发生拒绝，`(p-q)_+` 总和为零不是需要“加 epsilon 后继续抽样”的正常路径。

本地使用全前缀重算，便于检查验证位置、预算、EOS 与拒绝分布；无 cache 回滚/分页/专用草稿 kernel。
temperature=0 测试与普通目标贪心结果逐 token 相同；随机采样另有拒绝质量恒等式测试。
这里不将 MTP 辅助头自动作为 drafter，不实现 DFlash/DSpark，也不声称参考 Python 路径更快。

## 不带 ViT 的图像入口：先弄懂标签和位置

独立 `VisionPrefixModel` 把 `[B,3,8,8]` 分成四个 4×4 patch，展平、线性投影、加学习式行列坐标、LayerNorm。
四个视觉 embedding 接到文本 embedding 前，通过 `inputs_embeds` 进入 decoder。
视觉位置标签为 -100，文本保持原 SFT 标签；图像 projector 能通过文本 CE 得到梯度。

```text
RGB → patch projection → 四个视觉 token
                            + 文本 tokens → decoder → 文本标签 CE
```

这只是 encoder-free 输入接口的最小实验，不是 Gemma 4 权重实现，不具备预训练视觉知识。
不支持音频、视频、多图、任意分辨率或网页图像聊天；Engram/MTP 与占位 token IDs 的组合明确拒绝，避免伪造 n-gram 或未来词嵌入。

```bash
python -m pytest tests/test_quantization_speculative.py tests/test_frontier_mechanisms.py
```

完成标准：手动拆一个 byte；证明拒绝修正分布守恒；检查图像标签不参与 CE，而 projector 梯度非零。
然后分别做存储、延迟、能力实验，不用一个指标替所有问题作答。
