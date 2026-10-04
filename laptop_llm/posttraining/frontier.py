"""前沿后训练的几个最小数学单元：校正、质量分配、候选集概率。"""

import torch
from torch.nn import functional as F

from laptop_llm.posttraining.objectives import masked_mean


def calibrated_policy_loss(new_logp, behavior_logp, advantages, mask, lower=0.2, upper=0.2):
    """GLM-5 风格双侧 IS 校准：越界 token 完全屏蔽，系数 stop-gradient。

    和 PPO 的 min/clamp surrogate 不同。分母保持原有效 token 数，不能在
    过滤后重新除以幸存数量，否则 batch 的权重会随策略漂移改变。
    """
    if not 0 <= lower < 1 or upper < 0:
        raise ValueError("校准区间非法")
    ratio = (new_logp.detach() - behavior_logp.detach()).clamp(-20, 20).exp()
    keep = (ratio > 1 - lower) & (ratio < 1 + upper) & mask
    coefficient = torch.where(keep, ratio, 0)
    loss = -masked_mean(coefficient * advantages.detach() * new_logp, mask)
    return loss, 1 - keep.sum().float() / mask.sum()


@torch.no_grad()
def redistribute_advantages(rewards, quality, group_size, cap=2.0, hacked=None):
    """MiMo-V2.6 GAR：只重分配成功样本的正优势，再中心化。

    quality 为任务相关、独立验证的 (0,1] 因子，不能是模型自报分数。
    全成功/全失败组仍为零；GAR 无法凭空制造正确性信号。
    """
    if group_size < 2 or rewards.numel() % group_size or quality.shape != rewards.shape:
        raise ValueError("GAR 的组大小/质量形状不匹配")
    if not bool(((rewards == 0) | (rewards == 1)).all()):
        raise ValueError("GAR 输入是有效二值成功奖励")
    if not bool(((quality > 0) & (quality <= 1)).all()) or cap < 1:
        raise ValueError("quality 应在 (0,1]，cap 至少为 1")
    effective = rewards.clone()
    if hacked is not None:
        effective[hacked.bool()] = 0
    grouped = effective.reshape(-1, group_size)
    factors = quality.reshape_as(grouped)
    advantage = grouped - grouped.mean(-1, keepdim=True)
    positive = torch.where(grouped == 1, advantage, 0)
    scale = (
        positive.sum(-1, keepdim=True) / (positive * factors).sum(-1, keepdim=True).clamp_min(1e-8)
    ).clamp_max(cap)
    redistributed = torch.where(grouped == 1, advantage * factors * scale, advantage)
    return (redistributed - redistributed.mean(-1, keepdim=True)).flatten()


def replay_log_probs(logits, token_ids, candidates):
    """在 rollout 真正的候选集合内重新归一化，而不是拿 full-softmax 冒充行为策略。"""
    if candidates.shape != logits.shape or candidates.dtype != torch.bool:
        raise ValueError("candidates 必须是与 logits 同形的 bool bitmap")
    if not bool(candidates.gather(-1, token_ids[..., None]).all()):
        raise ValueError("采样 token 不在记录的候选集合中")
    return (
        F.log_softmax(logits.float().masked_fill(~candidates, -torch.inf), -1)
        .gather(-1, token_ids[..., None])
        .squeeze(-1)
    )


def version_mask(versions, current, max_lag, actions):
    """每个动作携带权重版本；未来版本或过旧动作都不进入训练。"""
    if max_lag < 0 or versions.shape != actions.shape:
        raise ValueError("版本筛选参数非法")
    return actions & (versions <= current) & (current - versions <= max_lag)


def freeze_routers(model):
    """专家可继续训练，路由器冻结；辅助均衡 loss 不会偷偷改变 router。"""
    from laptop_llm.architectures.moe import SparseMoE

    count = 0
    for module in model.modules():
        if isinstance(module, SparseMoE):
            module.router.requires_grad_(False)
            count += 1
    return count
