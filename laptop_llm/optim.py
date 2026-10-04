"""Muon/AdamW 混合优化器：矩阵方向正交化，embedding/标量继续用 AdamW。

这是单设备参考实现，不包含分布式 Muon、Per-Head Muon 或 Muown。
Newton–Schulz 是近似极分解；不要期待 5 步后奇异值精确等于 1。
"""

import math

import torch


def orthogonalize(update, steps=5):
    if update.ndim != 2 or steps < 1:
        raise ValueError("Muon 正交化输入须为二维矩阵，steps 为正")
    x = update.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(1e-7)
    for _ in range(steps):
        gram = x @ x.T
        polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + polynomial @ x
    return x.T if transposed else x


class HybridMuon(torch.optim.Optimizer):
    def __init__(self, model, lr=0.001, weight_decay=0.1, momentum=0.95):
        matrix, adaptive = [], []
        for name, parameter in model.named_parameters():
            # 词表矩阵的形状特殊；router 不做矩阵正交化，避免把路由实验混在一起。
            use_muon = parameter.ndim == 2 and not any(
                marker in name for marker in ("embedding", "lm_head", "tables", "router")
            )
            (matrix if use_muon else adaptive).append(parameter)
        super().__init__(
            [{"params": matrix, "kind": "muon"}, {"params": adaptive, "kind": "adamw"}],
            dict(lr=lr, weight_decay=weight_decay, momentum=momentum, betas=(0.9, 0.95), eps=1e-8),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                state = self.state[parameter]
                if group["kind"] == "muon":
                    if not state:
                        state["momentum"] = torch.zeros_like(gradient)
                    velocity = state["momentum"]
                    velocity.lerp_(gradient, 1 - group["momentum"])
                    direction = orthogonalize(gradient.lerp(velocity, group["momentum"]))
                    direction *= math.sqrt(max(1, parameter.size(0) / parameter.size(1)))
                else:
                    if not state:
                        state.update(
                            step=0, m=torch.zeros_like(gradient), v=torch.zeros_like(gradient)
                        )
                    state["step"] += 1
                    b1, b2 = group["betas"]
                    state["m"].lerp_(gradient, 1 - b1)
                    state["v"].lerp_(gradient.square(), 1 - b2)
                    direction = (state["m"] / (1 - b1 ** state["step"])) / (
                        (state["v"] / (1 - b2 ** state["step"])).sqrt() + group["eps"]
                    )
                decay = group["weight_decay"] if parameter.ndim >= 2 else 0
                parameter.mul_(1 - group["lr"] * decay)
                parameter.add_(direction.to(parameter.dtype), alpha=-group["lr"])
        return loss
