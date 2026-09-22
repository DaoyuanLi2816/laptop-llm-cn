"""可读版 Multi-head Latent Attention，展示低秩 KV 缓存与权重吸收。

与 DeepSeek 权重布局不同，不支持直接加载其 checkpoint。完整模型还需要 Q 低秩
投影、更细的 RoPE/nope 维度配置和高性能 kernel；这里聚焦可验证的代数等价。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 延迟导入避免 model → mla → model 循环初始化。
        from laptop_llm.model import RMSNorm

        self.heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.rank = config.kv_lora_rank
        self.dropout = config.dropout
        self.kv_down = nn.Linear(config.dim, self.rank, bias=False)
        self.kv_norm = RMSNorm(self.rank, config.norm_eps)
        self.kv_up = nn.Linear(self.rank, 2 * config.dim, bias=False)
        self.q_proj = nn.Linear(config.dim, 2 * config.dim, bias=False)
        self.k_rope = nn.Linear(config.dim, self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x, cos, sin, *, attention_mask=None, past_key_value=None, use_cache=False):
        from laptop_llm.model import apply_rope

        batch, length, _ = x.shape
        latent = self.kv_norm(self.kv_down(x))[:, None]  # [B,1,T,R]
        rope_key = apply_rope(self.k_rope(x)[:, None], cos, sin)  # 共享的位置键
        q = self.q_proj(x).view(batch, length, self.heads, 2 * self.head_dim).transpose(1, 2)
        q_content, q_position = q.chunk(2, dim=-1)
        q_position = apply_rope(q_position, cos, sin)
        past = 0
        if past_key_value is not None:
            past = past_key_value[0].size(2)
            latent = torch.cat((past_key_value[0], latent), dim=2)
            rope_key = torch.cat((past_key_value[1], rope_key), dim=2)
        present = (latent, rope_key) if use_cache else None
        # 原本 K=cW_k^T、V=cW_v^T。吸收后 QK^T=(QW_k)c^T，
        # attention·V=(attention·c)W_v^T；避免把历史 c 展开为每头完整 K/V。
        weights = self.kv_up.weight.view(self.heads, 2 * self.head_dim, self.rank)
        w_key, w_value = weights.split(self.head_dim, dim=1)
        absorbed_q = torch.einsum("bhtd,hdr->bhtr", q_content, w_key)
        scores = torch.einsum(
            "bhtr,bsr->bhts", absorbed_q.float(), latent[:, 0].float()
        ) + torch.einsum("bhtd,bsd->bhts", q_position.float(), rope_key[:, 0].float())
        scores = scores / math.sqrt(2 * self.head_dim)
        positions = past + torch.arange(length, device=x.device)
        keys = torch.arange(latent.size(2), device=x.device)
        mask = (keys[None] <= positions[:, None])[None, None]
        if attention_mask is not None:
            if attention_mask.shape != (batch, latent.size(2)):
                raise ValueError("MLA padding mask 与完整 cache 长度不一致")
            mask = mask & attention_mask[:, None, None].bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = scores.masked_fill(~mask.any(-1, keepdim=True), 0)
        probabilities = scores.softmax(-1).masked_fill(~mask, 0).to(x.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        context = torch.einsum("bhts,bsr->bhtr", probabilities, latent[:, 0])
        output = torch.einsum("bhtr,hdr->bhtd", context, w_value)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1)), present
