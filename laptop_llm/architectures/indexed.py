"""MiniMax MSA 风格两分支注意力：可训练 indexer → 每 GQA 组 Top-k 块。

主分支只 gather 入选块。参考实现按 query 循环，索引仍扫描全前缀，
没有 exp-free/KV-outer CUDA 核；不能把稀疏 FLOPs 直接当成速度提升。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from laptop_llm.architectures.cache import IndexedCache


class BlockIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.index_q = nn.Linear(config.dim, config.n_kv_heads * config.index_dim, bias=False)
        self.index_k = nn.Linear(config.dim, config.index_dim, bias=False)
        self.warmup = False  # 调用方显式控制，forward 不修改训练步数。

    def forward(self, x, q, k, v, past, attention_mask, use_cache, past_len):
        batch, heads, queries, dim = q.shape
        groups = self.config.n_kv_heads
        group_heads = heads // groups
        # 辅助 KL 不应把 backbone 训练成“讨好 indexer”。
        index_q = self.index_q(x.detach()).view(batch, queries, groups, -1).transpose(1, 2)
        index_keys = self.index_k(x.detach())
        if past is not None:
            index_keys = torch.cat((past.index_keys, index_keys), 1)
        total = k.size(2)
        valid_keys = torch.ones(batch, total, dtype=torch.bool, device=x.device)
        if attention_mask is not None:
            if attention_mask.shape != (batch, total):
                raise ValueError("indexed attention_mask 必须覆盖完整前缀")
            valid_keys = attention_mask.bool()
        block_size = self.config.index_block_size
        blocks = (total + block_size - 1) // block_size
        outputs, losses = [], []
        for t in range(queries):
            position = past_len + t
            visible = valid_keys & (torch.arange(total, device=x.device)[None] <= position)
            scores = torch.einsum("bgd,bsd->bgs", index_q[:, :, t].float(), index_keys.float())
            scores = scores / math.sqrt(self.config.index_dim)
            # 先遮未来 token，再 max-pool；整个未来块不可进入路由决策。
            masked = scores.masked_fill(~visible[:, None], -torch.inf)
            padded = F.pad(masked, (0, blocks * block_size - total), value=-torch.inf)
            block_scores = padded.view(batch, groups, blocks, block_size).amax(-1)
            local = position // block_size
            selection_scores = block_scores.clone()
            selection_scores[..., local] = torch.inf  # local 块占一个预算槽，不额外扩张预算。
            budget = blocks if self.warmup else min(self.config.index_topk, blocks)
            selected = selection_scores.topk(budget, -1).indices
            indices = selected[..., None] * block_size + torch.arange(block_size, device=x.device)
            indices = indices.flatten(-2)
            in_bounds = indices < total
            safe = indices.clamp_max(total - 1)
            allowed = visible[:, None].expand(-1, groups, -1).gather(-1, safe) & in_bounds
            # selected [B,G,K*block]；主 attention 从未构造 [T,T] score。
            head_indices = safe.repeat_interleave(group_heads, 1)
            head_allowed = allowed.repeat_interleave(group_heads, 1)
            gather_index = head_indices[..., None].expand(-1, -1, -1, dim)
            picked_k = k.gather(2, gather_index)
            picked_v = v.gather(2, gather_index)
            main_scores = (q[:, :, t, None].float() * picked_k.float()).sum(-1) / math.sqrt(dim)
            main_scores = main_scores.masked_fill(~head_allowed, -torch.inf)
            main_scores = main_scores.masked_fill(~head_allowed.any(-1, keepdim=True), 0)
            probabilities = main_scores.softmax(-1).masked_fill(~head_allowed, 0)
            weighted = F.dropout(probabilities.to(v.dtype), self.config.dropout, self.training)
            outputs.append((weighted[..., None] * picked_v).sum(-2))
            # 教师是同组各 Q 头的概率均值，必须 detach；不是平均 logits。
            target = probabilities.view(batch, groups, group_heads, -1).mean(2).detach()
            index_scores = scores.gather(-1, safe).masked_fill(~allowed, -torch.inf)
            index_scores = index_scores.masked_fill(~allowed.any(-1, keepdim=True), 0)
            log_index = index_scores.log_softmax(-1).masked_fill(~allowed, 0)
            kl = (target * (target.clamp_min(1e-12).log() - log_index)).sum(-1)
            active = valid_keys[:, position, None].expand(-1, groups)
            losses.append((kl * active, active))
        numerator = sum(item[0].sum() for item in losses)
        denominator = sum(item[1].sum() for item in losses).clamp_min(1)
        present = (
            IndexedCache(k[:, ::group_heads], v[:, ::group_heads], index_keys)
            if use_cache
            else None
        )
        return (
            torch.stack(outputs, 2),
            present,
            numerator / denominator * self.config.index_loss_coef,
        )
