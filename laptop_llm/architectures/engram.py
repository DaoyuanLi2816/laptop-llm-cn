"""Engram 风格条件记忆：确定性 N-gram 寻址 + 多头哈希 + 上下文门控。

小表驻留本机，展示记忆与计算解耦；不复刻 tokenizer 压缩或 RDMA 预取。
"""

import torch
from torch import nn
from torch.nn import functional as F


class NgramMemory(nn.Module):
    def __init__(self, dim, table_size=257, memory_dim=16, max_order=3):
        super().__init__()
        self.max_order = max_order
        self.tables = nn.ModuleList(
            [nn.Embedding(table_size + 2 * head, memory_dim) for head in range(2 * (max_order - 1))]
        )
        self.key = nn.Linear(memory_dim * len(self.tables), dim, bias=False)
        self.value = nn.Linear(memory_dim * len(self.tables), dim, bias=False)

    def forward(self, ids, context, history=None, valid_mask=None):
        batch, length = ids.shape
        prefix = ids.new_zeros(batch, self.max_order - 1) if history is None else history
        joined = torch.cat((prefix, ids), -1)
        lookups = []
        for table_index, table in enumerate(self.tables):
            order = 2 + table_index // 2
            hashed = torch.zeros_like(ids)
            # 每一步立即取模，避免长 N-gram 的 int64 溢出。
            for offset in range(order):
                start = prefix.size(1) - order + 1 + offset
                hashed = hashed * (31 + 6 * (table_index % 2)) + joined[:, start : start + length]
                hashed = hashed.remainder(table.num_embeddings)
            lookups.append(table(hashed))
        memory = torch.cat(lookups, -1)
        similarity = (
            F.normalize(context.float(), dim=-1) * F.normalize(self.key(memory).float(), dim=-1)
        ).sum(-1)
        update = self.value(memory) * similarity.sigmoid()[..., None].to(context.dtype)
        if valid_mask is not None:
            update = update * valid_mask[:, -length:, None]
        return update, joined[:, -(self.max_order - 1) :].clone()
