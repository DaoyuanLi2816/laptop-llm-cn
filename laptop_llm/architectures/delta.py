"""Kimi K3/Kimi Linear 的 KDA 递推参考路径（逐 token、无 Triton 依赖）。

保留通道遗忘、delta 写入、因果短卷积、Q/K L2Norm、输出门控。
采用 K3 有下界的 log-decay。生产实现的 chunkwise UT 变换另见课程 11。
"""

import torch
from torch import nn
from torch.nn import functional as F

from laptop_llm.architectures.cache import DeltaCache


def delta_update(state, q, k, v, alpha, beta):
    """先遗忘，再纠错：S'=diag(alpha)S；S=S'+beta*k*(v-k^T S')^T。

    所有头并行，只有时间轴循环。当前 token 先写再读，因果但包含自身。
    """
    decayed = state * alpha.unsqueeze(-1)
    prediction = torch.einsum("bhk,bhkv->bhv", k, decayed)
    correction = (v - prediction) * beta.unsqueeze(-1)
    updated = decayed + k.unsqueeze(-1) * correction.unsqueeze(-2)
    return updated, torch.einsum("bhk,bhkv->bhv", q, updated)


class DeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.kernel = config.delta_conv_kernel
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.conv_weight = nn.Parameter(torch.zeros(3 * config.dim, self.kernel))
        # 初始短卷积为恒等映射，再由训练学习最近几步如何混合。
        with torch.no_grad():
            self.conv_weight[:, -1] = 1
        self.alpha_proj = nn.Linear(config.dim, config.dim)
        self.beta_proj = nn.Linear(config.dim, config.n_heads)
        self.log_scale = nn.Parameter(torch.zeros(config.n_heads))
        self.gate_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.output_norm = nn.Parameter(torch.ones(self.head_dim))
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x, cos, sin, *, attention_mask=None, past_key_value=None, use_cache=False):
        del cos, sin  # KDA 的时序来自递推与因果卷积，不使用 RoPE。
        batch, length, width = x.shape
        past = 0 if past_key_value is None else past_key_value.length
        if past_key_value is None:
            state = x.new_zeros(
                batch, self.heads, self.head_dim, self.head_dim, dtype=torch.float32
            )
            history = x.new_zeros(batch, 3 * width, self.kernel - 1)
        else:
            # 不原地写缓存：同一前缀可被不同 continuation 复用。
            state, history = past_key_value.state, past_key_value.convolution
        valid = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        if attention_mask is not None:
            if attention_mask.shape != (batch, past + length):
                raise ValueError("KDA attention_mask 必须覆盖完整前缀")
            valid = attention_mask[:, -length:].bool()
        projected = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), -1)
        alpha_logits = self.alpha_proj(x).view(batch, length, self.heads, self.head_dim).float()
        alpha = (-5 * (alpha_logits * self.log_scale.exp()[None, None, :, None]).sigmoid()).exp()
        beta = self.beta_proj(x).float().sigmoid()
        outputs = []
        for t in range(length):
            window = torch.cat((history, projected[:, t, :, None]), -1)
            convolved = (window * self.conv_weight[None]).sum(-1)
            q, k, v = F.silu(convolved).chunk(3, -1)
            q = F.normalize(q.view(batch, self.heads, self.head_dim).float(), dim=-1)
            k = F.normalize(k.view(batch, self.heads, self.head_dim).float(), dim=-1)
            v = v.view(batch, self.heads, self.head_dim).float()
            proposed, output = delta_update(state, q, k, v, alpha[:, t], beta[:, t])
            keep = valid[:, t, None, None, None]
            state = torch.where(keep, proposed, state)
            history = torch.where(valid[:, t, None, None], window[:, :, 1:], history)
            outputs.append(output * valid[:, t, None, None])
        y = torch.stack(outputs, 1)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.config.norm_eps)
        y = (y * self.output_norm).reshape(batch, length, width).to(x.dtype)
        y = y * self.gate_proj(x).sigmoid()
        present = DeltaCache(state, history, past + length) if use_cache else None
        return self.o_proj(y), present
