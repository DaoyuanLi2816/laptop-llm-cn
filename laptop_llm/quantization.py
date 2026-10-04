"""真实 nibble 打包的 E2M1 FP4 教材：不是把 float tensor 改个名字。

每块使用 FP32 scale，因此既不是 MXFP4 的 E8M0 scale，也不是 NVFP4 的
E4M3 双层 scale。forward 反量化后调用普通 Linear，只承诺存储节省。
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def pack_fp4(weight, block_size=32):
    if block_size < 2 or block_size % 2 or weight.numel() == 0:
        raise ValueError("FP4 block_size 必须是正偶数，输入非空")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("不能量化 NaN/Inf")
    flat = weight.detach().float().flatten()
    padded = F.pad(flat, (0, (-flat.numel()) % block_size)).view(-1, block_size)
    scales = (padded.abs().amax(-1) / 6).clamp_min(1e-12)
    scaled = padded / scales[:, None]
    thresholds = scaled.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
    codes = torch.bucketize(scaled.abs().contiguous(), thresholds).to(torch.uint8)
    codes = (codes | ((scaled < 0).to(torch.uint8) << 3)).flatten()
    packed = codes[0::2] | (codes[1::2] << 4)
    return packed, scales


def unpack_fp4(packed, scales, shape, block_size=32):
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten().long()
    levels = scales.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    values = levels[codes & 7] * torch.where(codes >= 8, -1, 1)
    values = values * scales.repeat_interleave(block_size)
    return values[: math.prod(shape)].reshape(shape)


def fake_quantize_fp4(weight, block_size=32):
    """QAT 的直通梯度估计：前向看量化值，反向近似 dQ/dW=1。"""
    packed, scales = pack_fp4(weight, block_size)
    rounded = unpack_fp4(packed, scales, weight.shape, block_size).to(weight.dtype)
    return weight + (rounded - weight).detach()


class PackedLinear(nn.Module):
    """只读推理层；buffer 中保存 uint8 codes 和 FP32 scales。"""

    def __init__(self, linear, block_size=32):
        super().__init__()
        self.shape = tuple(linear.weight.shape)
        self.block_size = block_size
        packed, scales = pack_fp4(linear.weight, block_size)
        self.register_buffer("codes", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    @property
    def weight(self):
        # MLA 的权重吸收直接读取投影矩阵；仍兼容这条代数参考路径。
        return unpack_fp4(self.codes, self.scales, self.shape, self.block_size)

    def forward(self, x):
        weight = self.weight.to(x.dtype)
        return F.linear(x, weight, None if self.bias is None else self.bias.to(x.dtype))


def replace_linears(model, block_size=32, names=None):
    selected = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue  # 共享词表矩阵保持高精度与 tie_embeddings 语义。
        if names is not None and name not in names:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, PackedLinear(module, block_size))
        selected.append(name)
    if names is not None and set(selected) != set(names):
        raise ValueError("量化清单与模型模块不匹配")
    return selected


def export_quantized(checkpoint, output, block_size=32):
    from laptop_llm.engine import load_inference_bundle, read_checkpoint

    target = Path(output)
    if target.exists():
        raise ValueError("量化输出已存在；请选择新文件")
    model, tokenizer, _ = load_inference_bundle(checkpoint, device_name="cpu", dtype_name="float32")
    payload = read_checkpoint(checkpoint)
    if "quantization" in payload:
        raise ValueError("请从高精度 checkpoint 量化，不重复量化")
    names = replace_linears(model, block_size)
    result = {
        "format_version": 3,
        "stage": "quantized-inference",
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "tokenizer_json": tokenizer.to_str(),
        "quantization": {"format": "e2m1-fp32-scale", "block_size": block_size, "modules": names},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(target)
    return target
