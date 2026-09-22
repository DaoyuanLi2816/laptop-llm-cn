"""逐行可推导的目标函数。所有 logp 都是 log probability，不是 logits。

shape 约定：B=回答数，T=动作数，V=词表大小。mask=True 的位置才是回答动作，
prompt 和 padding 不参与策略梯度；不能直接对整张 [B,T] 求 mean。
"""

import torch
import torch.nn.functional as F


def masked_mean(values, mask):
    if not bool(mask.any()):
        raise ValueError("没有可训练的回答 token")
    return (values * mask).sum() / mask.sum()


def sequence_mean(values, mask):
    """先每条回答平均，再平均 batch；与全局 token 平均不是同一目标。"""
    if bool((mask.sum(-1) == 0).any()):
        raise ValueError("每条回答至少需要一个有效动作")
    return ((values * mask).sum(-1) / mask.sum(-1)).mean()


def group_advantages(rewards, group_size, eps=1e-6):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("GRPO 要求每题至少两条回答，且 batch 可按 group_size 整分")
    grouped = rewards.reshape(-1, group_size)
    # population std 不使用无偏估计；常数组的 advantage 精确为零。
    centered = grouped - grouped.mean(-1, keepdim=True)
    return (centered / grouped.std(-1, keepdim=True, unbiased=False).clamp_min(eps)).flatten()


def clipped_policy_loss(new_logp, old_logp, advantages, mask, clip=0.2):
    """PPO/GRPO 共用的 clipped surrogate。旧策略和优势必须 stop-gradient。

    A>0 时阻止概率提升超过 1+clip；A<0 时阻止概率下降超过 1-clip。
    不是把 loss 数值 clip，也不是对 token id 做 clip。
    """
    ratio = (new_logp - old_logp.detach()).clamp(-20, 20).exp()
    objective = torch.minimum(
        ratio * advantages.detach(), ratio.clamp(1 - clip, 1 + clip) * advantages.detach()
    )
    return -sequence_mean(objective, mask)


def reference_kl(new_logp, reference_logp, mask):
    """非负 k3 样本估计 exp(log q-log p)-1-(log q-log p)。

    当样本来自 p 时其期望为 KL(p||q)。多轮重用旧轨迹时只是 surrogate，
    不宣称无偏；完整词表 KL 在蒸馏函数中另行精确计算。
    """
    difference = (reference_logp.detach() - new_logp).clamp(-20, 20)
    return sequence_mean(difference.exp() - 1 - difference, mask)


@torch.no_grad()
def generalized_advantage(rewards, values, next_values, mask, terminal, gamma=1.0, lam=0.95):
    """GAE: delta_t=r_t+gamma*V(s_{t+1})-V(s_t)，A_t=delta_t+gamma*lambda*A_{t+1}。

    terminal 仅标真正 EOS；长度截断允许使用 next_values bootstrap。
    padding 不是动作；倒序扫描经过 padding 时重置累积量。
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.size(1))):
        continuation = (~terminal[:, t]).to(values.dtype)
        delta = rewards[:, t] + gamma * next_values[:, t] * continuation - values[:, t]
        carry = (delta + gamma * lam * continuation * carry) * mask[:, t]
        advantages[:, t] = carry
    return advantages, (advantages + values) * mask


def clipped_value_loss(values, old_values, returns, mask, clip=0.2):
    clipped = old_values.detach() + (values - old_values.detach()).clamp(-clip, clip)
    errors = torch.maximum(
        (values - returns.detach()).square(), (clipped - returns.detach()).square()
    )
    return 0.5 * masked_mean(errors, mask)


def distillation_loss(student_logits, teacher_logits, mask, temperature=1.0, reverse=True):
    """同一 tokenizer 下的 full-vocabulary KL；不把 teacher 的 sampled token 当软标签。

    reverse=True: KL(student||teacher)，否则 KL(teacher||student)。T² 缩放保持
    温度改变后梯度量级相近。轨迹本身无梯度，这是 on-policy GKD 风格 surrogate。
    """
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError("温度必须为正，师生 logits 必须同形")
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    per_token = (
        ((student - teacher) * student.exp()).sum(-1)
        if reverse
        else ((teacher - student) * teacher.exp()).sum(-1)
    )
    return masked_mean(per_token, mask) * temperature**2


def preference_reward_loss(chosen, rejected):
    """Bradley–Terry: P(chosen>rejected)=sigmoid(r_chosen-r_rejected)。"""
    return -F.logsigmoid(chosen - rejected).mean()
