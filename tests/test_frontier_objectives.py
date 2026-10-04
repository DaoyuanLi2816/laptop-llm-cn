import copy

import pytest
import torch

from laptop_llm.config import ModelConfig, StageConfig
from laptop_llm.engine import build_optimizer
from laptop_llm.model import LaptopLLM
from laptop_llm.optim import orthogonalize
from laptop_llm.posttraining.agent import ArithmeticTask, parse_action
from laptop_llm.posttraining.frontier import (
    calibrated_policy_loss,
    freeze_routers,
    redistribute_advantages,
    replay_log_probs,
    version_mask,
)


def test_calibration_gradient_is_detached_and_dropped_tokens_have_zero_gradient():
    new = torch.tensor([[0.0, 0.1, 0.8, -0.8]], requires_grad=True)
    old = torch.zeros_like(new, requires_grad=True)
    advantage = torch.ones_like(new, requires_grad=True)
    mask = torch.ones_like(new, dtype=torch.bool)
    loss, dropped = calibrated_policy_loss(new, old, advantage, mask)
    loss.backward()
    torch.testing.assert_close(
        new.grad, torch.tensor([[-0.25, -0.25 * torch.exp(torch.tensor(0.1)), 0, 0]])
    )
    assert old.grad is None and advantage.grad is None and dropped == 0.5


def test_gar_hack_reset_and_group_mean():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    quality = torch.tensor([1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.4, 0.8])
    advantage = redistribute_advantages(rewards, quality, 4)
    assert advantage[0] > advantage[1] > 0
    torch.testing.assert_close(advantage.view(-1, 4).mean(-1), torch.zeros(2))
    torch.testing.assert_close(advantage[2:4], torch.tensor([-0.5, -0.5]))
    assert advantage[4:].count_nonzero() == 0
    hacked = torch.tensor([True, False, False, False, False, False, False, False])
    revised = redistribute_advantages(rewards, quality, 4, hacked=hacked)
    assert revised[0] < 0 and revised[1] > 0


def test_candidate_and_version_replay():
    logits = torch.tensor([[[0.0, 0.0, 100.0]]], requires_grad=True)
    candidates = torch.tensor([[[True, True, False]]])
    logp = replay_log_probs(logits, torch.tensor([[1]]), candidates)
    torch.testing.assert_close(logp, torch.tensor([[-0.6931472]]))
    logp.sum().backward()
    assert logits.grad[0, 0, 2] == 0
    with pytest.raises(ValueError):
        replay_log_probs(logits, torch.tensor([[2]]), candidates)
    assert version_mask(
        torch.tensor([[1, 3, 5]]), 4, 2, torch.ones(1, 3, dtype=torch.bool)
    ).tolist() == [[False, True, False]]


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":true}',
        '{"answer":1,"answer":2}',
        '{"tool":"shell","arguments":{}}',
        '{"tool":"calculator","arguments":{"a":1,"b":2,"op":[]}}',
        '{"answer":7,"reward":1}',
        "7",
        '{"tool":"calculator","arguments":{"a":true,"b":2,"op":"add"}}',
    ],
)
def test_agent_rejects_unsafe_or_ambiguous_actions(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_tool_requirement_is_verified_independently():
    task = ArithmeticTask(3, 4)
    assert task.verify(7, True) == 1 and task.verify(7, False) == 0
    assert task.verify(8, True) == 0


def test_muon_state_roundtrip_and_router_freezing():
    model = LaptopLLM(
        ModelConfig(
            vocab_size=32,
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            num_experts=4,
            max_seq_len=16,
        )
    )
    router = model.layers[0].ffn.router.weight.detach().clone()
    assert freeze_routers(model) == 1
    config = StageConfig(optimizer="muon", learning_rate=0.001)
    optimizer = build_optimizer(model, config)
    ids = torch.randint(0, 32, (2, 8))
    output = model(ids, labels=ids)
    (output.loss + output.auxiliary_loss).backward()
    optimizer.step()
    torch.testing.assert_close(router, model.layers[0].ffn.router.weight)
    clone = copy.deepcopy(model)
    second = build_optimizer(clone, config)
    second.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for current, opt in [(model, optimizer), (clone, second)]:
        opt.zero_grad()
        output = current(ids, labels=ids)
        (output.loss + output.auxiliary_loss).backward()
        opt.step()
    for a, b in zip(model.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert orthogonalize(torch.zeros(3, 5)).count_nonzero() == 0
