"""真正只收集局部 K/V 的滑窗 attention；不构造 T×T 的注意力分数。

这不是 DeepSeek DSA：这里的边由位置决定，DSA 还会学习 indexer 来选 token。
教学实现仍会物化 [B,H,T,window+sinks,D]，不等于 fused sparse kernel 的性能。
"""

import math

import torch
import torch.nn.functional as F


def sparse_attention(q, k, v, *, past_len, window, sinks, padding_mask=None, dropout=0.0):
    """q=[B,H,T,D]，k/v=[B,H,S,D]；True 表示允许关注。

    sink 是序列开头的固定锚点。局部窗口与 sink 重叠时必须去重，否则同一
    token 会在 softmax 中被重复计票。绝对位置偏移保证 prefill/decode 一致。
    """
    batch, _, query_len, dim = q.shape
    key_len = k.size(2)
    sinks = min(sinks, key_len)
    positions = past_len + torch.arange(query_len, device=q.device)
    local = positions[:, None] - torch.arange(window - 1, -1, -1, device=q.device)
    anchors = torch.arange(sinks, device=q.device).expand(query_len, -1)
    indices = torch.cat((anchors, local), dim=1)
    valid = torch.cat((anchors <= positions[:, None], local >= sinks), dim=1)
    valid = valid & (indices >= 0) & (indices < key_len)
    indices = indices.clamp(0, key_len - 1)
    allowed = valid[None, None].expand(batch, 1, -1, -1)
    if padding_mask is not None:
        if padding_mask.shape != (batch, key_len):
            raise ValueError("padding_mask 与完整 KV 长度不一致")
        allowed = allowed & padding_mask[:, indices][:, None].bool()
    selected_k, selected_v = k[:, :, indices], v[:, :, indices]
    scores = (q.unsqueeze(-2).float() * selected_k.float()).sum(-1) / math.sqrt(dim)
    # 全 padding 的 query 行返回零；不要让 softmax(-inf,...,-inf) 产生 NaN。
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = scores.masked_fill(~allowed.any(-1, keepdim=True), 0)
    weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0).to(v.dtype)
    weights = F.dropout(weights, p=dropout, training=dropout > 0)
    return (weights.unsqueeze(-1) * selected_v).sum(-2)
