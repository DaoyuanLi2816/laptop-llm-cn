"""用手算例子锁定方向、mask、bootstrap，而不是只检查 loss 没有 NaN。"""

import pytest
import torch

from laptop_llm.posttraining.objectives import (
    clipped_policy_loss,
    clipped_value_loss,
    distillation_loss,
    generalized_advantage,
    group_advantages,
    preference_reward_loss,
    reference_kl,
)
from laptop_llm.posttraining.rewards import verify_arithmetic


def test_group_normalization_and_zero_variance():
    actual = group_advantages(torch.tensor([1.0, 3.0, 9.0, 9.0]), 2)
    torch.testing.assert_close(actual, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        group_advantages(torch.ones(3), 2)


def test_ppo_clipping_gradient_and_mask():
    new = torch.tensor([[0.4, -0.4, 0.0, 99.0]], requires_grad=True)
    old = torch.zeros_like(new)
    advantages = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    loss = clipped_policy_loss(new, old, advantages, mask)
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([[0.0, 0.0, -1 / 3, 0.0]]))


def test_gae_terminal_versus_truncation():
    reward = torch.tensor([[0.0, 1.0, 0.0]])
    value = torch.tensor([[0.2, 0.3, 0.0]])
    next_value = torch.tensor([[0.3, 0.5, 0.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    terminal = torch.tensor([[0, 1, 0]], dtype=torch.bool)
    adv, returns = generalized_advantage(reward, value, next_value, mask, terminal, lam=1)
    torch.testing.assert_close(adv, torch.tensor([[0.8, 0.7, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[1.0, 1.0, 0.0]]))
    _, bootstrapped = generalized_advantage(
        reward, value, next_value, mask, terminal * False, lam=1
    )
    torch.testing.assert_close(bootstrapped, torch.tensor([[1.5, 1.5, 0.0]]))


def test_value_clip_and_reward_direction():
    mask = torch.ones(1, 1, dtype=torch.bool)
    torch.testing.assert_close(
        clipped_value_loss(torch.tensor([[1.0]]), torch.zeros(1, 1), torch.ones(1, 1), mask),
        torch.tensor(0.32),
    )
    chosen = torch.zeros(2, requires_grad=True)
    rejected = torch.zeros(2, requires_grad=True)
    preference_reward_loss(chosen, rejected).backward()
    assert (chosen.grad < 0).all() and (rejected.grad > 0).all()


def test_distillation_matches_categorical_kl_and_freezes_teacher():
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    loss = distillation_loss(student, teacher, mask)
    oracle = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=student),
        torch.distributions.Categorical(logits=teacher),
    )
    torch.testing.assert_close(loss, oracle[mask].mean())
    loss.backward()
    assert teacher.grad is None and student.grad[~mask].abs().sum() == 0
    torch.testing.assert_close(
        reference_kl(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)),
        torch.tensor(0.0),
    )


@pytest.mark.parametrize(
    "response,score",
    [
        ("<answer>5</answer>", 1),
        ("<think>2+3=5</think><answer>5</answer>", 1),
        ("5", 0),
        ("<answer>4</answer> 5", 0),
        ("<answer>5</answer><answer>5</answer>", 0),
        ("<answer>05</answer>", 0),
        ("<think><answer>5</answer></think><answer>5</answer>", 0),
        ("<answer>5</answer><|user|>", 0),
        ("<answer>5.0</answer>", 0),
    ],
)
def test_verifier_rejects_reward_hacking(response, score):
    assert verify_arithmetic(response, 5) == score
