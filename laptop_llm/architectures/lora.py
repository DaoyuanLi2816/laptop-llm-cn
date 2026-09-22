"""LoRA：冻结 W，只训练低秩增量 BA * alpha/r；零初始化 B 保持初始函数不变。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank 必须大于零")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale

    @torch.no_grad()
    def merged(self):
        """返回普通 Linear，便于复用原有 checkpoint 和 serving，无需部署 adapter。"""
        import copy

        result = copy.deepcopy(self.base)
        result.weight.add_((self.b @ self.a) * self.scale)
        return result


def inject_lora(model, rank=4, alpha=8, targets=("q_proj", "v_proj")):
    model.requires_grad_(False)
    count = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                count += 1
    if not count:
        raise ValueError("没有匹配的 LoRA target")
    return count


def merge_lora(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
    return model
