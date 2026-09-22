"""稀疏 MoE：top-k 路由 → 按专家收集 token → 专家计算 → 加权散回。

所有专家权重仍驻留内存，只有部分专家参与每个 token 的计算。
无容量截断（dropless）、无跨卡 all-to-all，易读但不是生产调度器。
"""

import torch
from torch import nn
from torch.nn import functional as F


class SparseMoE(nn.Module):
    def __init__(self, config, expert_factory):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList([expert_factory(config) for _ in range(config.num_experts)])
        self.shared = nn.ModuleList([expert_factory(config) for _ in range(config.shared_experts)])

    def forward(self, x, valid_mask=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        valid = torch.ones(flat.size(0), device=x.device, dtype=torch.bool)
        if valid_mask is not None:
            valid = valid_mask.reshape(-1).bool()
        logits = self.router(flat).float()
        probabilities = logits.softmax(-1)
        weights, indices = probabilities.topk(self.config.experts_per_token, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token, slot = torch.where((indices == expert_id) & valid[:, None])
            if token.numel():
                contribution = expert(flat[token]) * weights[token, slot, None].to(flat.dtype)
                result = result.index_add(0, token, contribution)
        for expert in self.shared:
            result = result + expert(flat) * valid[:, None]
        # f_i 为离散分配频率（无梯度），p_i 为平均路由概率（有梯度）。
        # 二者乘积抑制少数专家被挤爆；并不保证每个 batch 完全均衡。
        count = valid.sum().clamp_min(1)
        assignments = F.one_hot(indices, self.config.num_experts).float().sum(1)
        frequency = (assignments * valid[:, None]).sum(0) / (count * self.config.experts_per_token)
        mean_probability = (probabilities * valid[:, None]).sum(0) / count
        balance = self.config.num_experts * (frequency.detach() * mean_probability).sum()
        z_loss = (logits.logsumexp(-1).square() * valid).sum() / count
        auxiliary = self.config.router_aux_coef * balance + self.config.router_z_coef * z_loss
        return result.view(shape), auxiliary
