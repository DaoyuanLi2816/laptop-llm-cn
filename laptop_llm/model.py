"""现代小型 decoder-only Transformer：RMSNorm + RoPE + GQA + SwiGLU。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from laptop_llm.architectures.mla import LatentAttention
from laptop_llm.architectures.moe import SparseMoE
from laptop_llm.architectures.sparse import sparse_attention
from laptop_llm.config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None
    hidden_states: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    """只按均方根缩放，不减均值；LLaMA 系模型的标准归一化。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化的统计量用 fp32 计算，避免 fp16 下平方/求和损失精度。
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def build_rope_cache(
    head_dim: int, max_seq_len: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算每个绝对位置的旋转角；真正进入 attention 时再转成 q/k 的 dtype。"""

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    return frequencies.cos(), frequencies.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """把最后一维两两视作二维向量，按 token 位置进行旋转。"""

    batch, heads, seq_len, head_dim = x.shape
    paired = x.float().reshape(batch, heads, seq_len, head_dim // 2, 2)
    x_even, x_odd = paired.unbind(dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention。

    例如 8 个 Q 头、2 个 KV 头时，每 4 个 Query 头共享一份 K/V。
    训练计算量变化不大，但 KV Cache 只有普通 MHA 的 1/4。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout
        self.config = config
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, query_len, _ = x.shape
        q = self.q_proj(x).view(batch, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, query_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None

        # PyTorch SDPA 会自动选择 Flash / memory-efficient / math 后端。
        # 为兼容更广的 PyTorch 版本，这里显式复制 KV，而不依赖 enable_gqa 参数。
        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            k_for_attn = k.repeat_interleave(repeat, dim=1)
            v_for_attn = v.repeat_interleave(repeat, dim=1)
        else:
            k_for_attn, v_for_attn = k, v

        if self.config.attention_pattern == "sliding":
            y = sparse_attention(
                q,
                k_for_attn,
                v_for_attn,
                past_len=past_len,
                window=self.config.sliding_window,
                sinks=self.config.attention_sinks,
                padding_mask=attention_mask,
                dropout=self.dropout if self.training else 0.0,
            )
            return self.o_proj(y.transpose(1, 2).contiguous().view(batch, query_len, -1)), present

        key_len = k_for_attn.size(2)
        attn_mask: torch.Tensor | None = None
        is_causal = past_len == 0 and attention_mask is None and query_len > 1

        # 有 padding 或“带 cache 一次追加多个 token”时，构造带位置偏移的显式 mask。
        if attention_mask is not None or (past_len > 0 and query_len > 1):
            query_positions = past_len + torch.arange(query_len, device=x.device)
            key_positions = torch.arange(key_len, device=x.device)
            causal = key_positions[None, :] <= query_positions[:, None]  # [T, S]
            attn_mask = causal[None, None, :, :].expand(batch, 1, query_len, key_len)
            if attention_mask is not None:
                if attention_mask.shape != (batch, key_len):
                    raise ValueError(
                        f"attention_mask 应为 {(batch, key_len)}，实际是 {tuple(attention_mask.shape)}"
                    )
                attn_mask = attn_mask & attention_mask[:, None, None, :].bool()

        y = F.scaled_dot_product_attention(
            q,
            k_for_attn,
            v_for_attn,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.o_proj(y), present


class SwiGLU(nn.Module):
    """门控前馈网络：SiLU(W_gate x) * (W_up x)，再投回残差维度。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.ffn_dim
        self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-Norm 残差块：Norm → Attention → 残差 → Norm → FFN → 残差。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = (
            LatentAttention(config)
            if config.attention_type == "mla"
            else GroupedQueryAttention(config)
        )
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SparseMoE(config, SwiGLU) if config.num_experts else SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None, torch.Tensor]:
        attn_out, present = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        auxiliary = x.new_zeros(())
        if isinstance(self.ffn, SparseMoE):
            valid = None if attention_mask is None else attention_mask[:, -x.size(1) :]
            update, auxiliary = self.ffn(self.ffn_norm(x), valid)
        else:
            update = self.ffn(self.ffn_norm(x))
        x = x + update
        return x, present, auxiliary


class LaptopLLM(nn.Module):
    """完整语言模型。

    训练时输入整段 token，并计算 next-token loss；生成时先 prefill 整个 prompt，
    随后每步只输入一个新 token，历史信息由每层的 KV Cache 保存。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("创建模型前必须把 vocab_size 设置为实际词表大小")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    replace(config, attention_pattern="dense")
                    if config.dense_every and (index + 1) % config.dense_every == 0
                    else config
                )
                for index in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        head_dim = config.dim // config.n_heads
        cos, sin = build_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        # 深层残差分支的输出用较小方差初始化，减少训练初期的数值震荡。
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.layers:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            for module in block.ffn.modules():
                if isinstance(module, SwiGLU):
                    nn.init.normal_(module.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """少存中间激活、反向时重算，以计算时间换显存。"""

        self.gradient_checkpointing = enabled

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [batch, seq_len]")
        batch, seq_len = input_ids.shape
        if seq_len == 0 or batch == 0:
            raise ValueError("batch 和序列长度必须非空")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 的层数与模型不一致")
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"序列总长 {past_len + seq_len} 超过 max_seq_len={self.config.max_seq_len}"
            )
        if labels is not None and use_cache:
            raise ValueError("训练 loss 与 use_cache 不应同时开启")

        x = self.dropout(self.token_embedding(input_ids))
        cos = self.rope_cos[past_len : past_len + seq_len]
        sin = self.rope_sin[past_len : past_len + seq_len]
        presents: list[KVCache] = []
        auxiliary = x.new_zeros(())

        for layer_index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[layer_index]
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("gradient checkpointing 训练时不能同时构建 KV Cache")

                def custom_forward(
                    hidden: torch.Tensor, _layer: TransformerBlock = layer
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    # 用默认参数绑定当前层；否则 backward 重算时闭包会指向最后一层。
                    result = _layer(
                        hidden, cos, sin, attention_mask=attention_mask, use_cache=False
                    )
                    return result[0], result[2]

                x, layer_auxiliary = checkpoint(custom_forward, x, use_reentrant=False)
                present = None
            else:
                x, present, layer_auxiliary = layer(
                    x,
                    cos,
                    sin,
                    attention_mask=attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                )
            auxiliary = auxiliary + layer_auxiliary / len(self.layers)
            if present is not None:
                presents.append(present)

        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.shape != (batch, seq_len):
                raise ValueError("labels 必须与 input_ids 形状相同")
            # 位置 t 的输出预测位置 t+1，因此 logits 去尾、labels 去头。
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return ModelOutput(
            logits=logits,
            loss=loss,
            past_key_values=presents or None,
            hidden_states=hidden_states,
            auxiliary_loss=auxiliary,
        )
