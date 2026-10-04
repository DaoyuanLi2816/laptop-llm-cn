"""在深度轴上选择信息：AttnRes；在多条残差流上稳定混合：mHC。"""

import torch
from torch import nn


class AttentionResidual(nn.Module):
    """输入 [sources,B,T,D]，每个 token 独立地对深度来源做 softmax。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, sources):
        values = sources.float()
        keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        scores = torch.einsum("sbtd,d->sbt", keys, self.query.float())
        weights = scores.softmax(0)
        return (values * weights[..., None]).sum(0).to(sources.dtype)


def sinkhorn(logits, iterations=20):
    """log 域交替行/列归一化，投到近似双随机矩阵。有限迭代有数值误差。"""
    if iterations < 1 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("Sinkhorn 要求方阵与正迭代次数")
    log_matrix = logits.float()
    for _ in range(iterations):
        log_matrix = log_matrix - log_matrix.logsumexp(-1, keepdim=True)
        log_matrix = log_matrix - log_matrix.logsumexp(-2, keepdim=True)
    return log_matrix.exp()


class ManifoldConnection(nn.Module):
    """mHC 的动态 A/B/C 映射。B 双随机约束不等于整个非线性块非扩张。"""

    def __init__(self, dim, streams=4, iterations=20):
        super().__init__()
        self.streams = streams
        self.iterations = iterations
        self.projection = nn.Linear(dim * streams, 2 * streams + streams**2, bias=False)
        self.dynamic_scale = nn.Parameter(torch.full((3,), 0.01))
        self.pre_bias = nn.Parameter(torch.zeros(streams))
        self.post_bias = nn.Parameter(torch.zeros(streams))
        self.res_bias = nn.Parameter(2 * torch.eye(streams))

    def coefficients(self, x):
        flat = x.flatten(-2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
        raw = self.projection(flat.to(self.projection.weight.dtype)).float()
        pre, post, residual = raw.split([self.streams, self.streams, self.streams**2], -1)
        a = (pre * self.dynamic_scale[0] + self.pre_bias).sigmoid()
        c = 2 * (post * self.dynamic_scale[1] + self.post_bias).sigmoid()
        b = sinkhorn(
            residual.view(*x.shape[:-2], self.streams, self.streams) * self.dynamic_scale[2]
            + self.res_bias,
            self.iterations,
        )
        return a, b, c

    def read(self, x):
        a, b, c = self.coefficients(x)
        return (x.float() * a[..., None]).sum(-2).to(x.dtype), (b, c)

    @staticmethod
    def write(x, update, coefficients):
        b, c = coefficients
        result = torch.einsum("...ij,...jd->...id", b, x.float())
        return (result + c[..., None] * update.float().unsqueeze(-2)).to(x.dtype)
