"""CSA2/CED 的机制实验：序列压缩、跨层 KV 共享与索引复用。

这是独立代数实验，未接入主模型/serving。均值压缩代替论文的学习式
压缩器，保留“完整块才可见”因果约束；不实现 Bounded Replay。
"""

from dataclasses import dataclass

import torch


@dataclass
class CompressedMemory:
    keys: torch.Tensor  # [B,H,N_blocks,D]
    values: torch.Tensor
    end_positions: torch.Tensor  # 每个完整块最后一项的绝对位置


def compress_complete_blocks(keys, values, ratio):
    if ratio < 1 or keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("压缩要求同形 [B,H,T,D] KV 与正压缩率")
    batch, heads, length, dim = keys.shape
    blocks = length // ratio
    shape = (batch, heads, blocks, ratio, dim)
    # 尾部未完整的块不进入全局压缩存储，应由局部注意力保留。
    k = keys[:, :, : blocks * ratio].reshape(shape).mean(-2)
    v = values[:, :, : blocks * ratio].reshape(shape).mean(-2)
    ends = torch.arange(blocks, device=keys.device) * ratio + ratio - 1
    return CompressedMemory(k, v, ends)


def select_memory(query, memory, query_positions, topk):
    if topk < 1 or query.size(2) != query_positions.numel():
        raise ValueError("索引预算/位置非法")
    scores = torch.einsum("bhtd,bhsd->bhts", query.float(), memory.keys.float())
    visible = memory.end_positions[None, None, None] <= query_positions[None, None, :, None]
    scores = scores.masked_fill(~visible, -torch.inf)
    return scores.topk(min(topk, memory.keys.size(2)), -1).indices


def attend_memory(query, memory, indices, query_positions):
    batch, heads, length, dim = query.shape
    if indices.size(-1) == 0:
        return torch.zeros_like(query)
    expanded_k = memory.keys[:, :, None].expand(-1, -1, length, -1, -1)
    expanded_v = memory.values[:, :, None].expand_as(expanded_k)
    gather = indices[..., None].expand(-1, -1, -1, -1, dim)
    keys, values = expanded_k.gather(3, gather), expanded_v.gather(3, gather)
    visible = memory.end_positions[indices] <= query_positions[None, None, :, None]
    scores = (query[..., None, :].float() * keys.float()).sum(-1) / dim**0.5
    scores = scores.masked_fill(~visible, -torch.inf)
    scores = scores.masked_fill(~visible.any(-1, keepdim=True), 0)
    weights = scores.softmax(-1).masked_fill(~visible, 0)
    return (weights[..., None] * values.float()).sum(-2).to(query.dtype)


def layer_reuse(
    mode,
    query,
    query_positions,
    topk,
    *,
    source_keys=None,
    source_values=None,
    ratio=2,
    previous_memory=None,
    previous_indices=None,
):
    """Full 产生 KV+索引；Reindex 共享 KV 重选；Reuse 共享 KV 与索引。

    CED 中 Full 的 source 应来自最后 encoder hidden 的投影，不能误用
    当前 decoder hidden。这里由调用者显式提供 source，不隐藏数据来源。
    """
    if mode == "full":
        if source_keys is None or source_values is None:
            raise ValueError("Full 必须提供源 KV")
        memory = compress_complete_blocks(source_keys, source_values, ratio)
    elif mode in {"reindex", "reuse"} and previous_memory is not None:
        memory = previous_memory
    else:
        raise ValueError("Full 必须提供源 KV；复用模式必须提供上一层 memory")
    if mode == "reuse":
        if previous_indices is None:
            raise ValueError("Reuse 缺少上一层索引")
        indices = previous_indices
    else:
        indices = select_memory(query, memory, query_positions, topk)
    return attend_memory(query, memory, indices, query_positions), memory, indices
