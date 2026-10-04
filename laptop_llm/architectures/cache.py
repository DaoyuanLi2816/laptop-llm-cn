"""混合模型的缓存契约：序列长度不再等于某个张量的第三个维度。"""

from dataclasses import dataclass

import torch


@dataclass
class DeltaCache:
    """KDA 固定大小状态。length 是位置计数，不能从 state 的形状猜出来。"""

    state: torch.Tensor  # [B, H, Dk, Dv]，FP32 累积
    convolution: torch.Tensor  # [B, 3*dim, kernel-1]
    length: int


@dataclass
class IndexedCache:
    keys: torch.Tensor
    values: torch.Tensor
    index_keys: torch.Tensor

    @property
    def length(self) -> int:
        return self.keys.size(2)


class CacheBundle(list):
    """保留 list 兼容性，附带 Engram 的短 token 历史；不保存全部输入。"""

    def __init__(self, layers, token_history=None):
        super().__init__(layers)
        self.token_history = token_history


def cache_length(cache) -> int:
    return cache.length if isinstance(cache, (DeltaCache, IndexedCache)) else cache[0].size(2)


def cache_bytes(cache) -> int:
    """实际驻留张量字节数；共享 storage 只计一次。"""
    seen = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        if isinstance(value, (DeltaCache, IndexedCache)):
            return sum(visit(item) for item in vars(value).values())
        return 0

    return visit(cache) + visit(getattr(cache, "token_history", None))
