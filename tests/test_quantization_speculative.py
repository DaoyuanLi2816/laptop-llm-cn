import copy

import pytest
import torch

from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM
from laptop_llm.quantization import fake_quantize_fp4, pack_fp4, replace_linears, unpack_fp4
from laptop_llm.speculative import correction_distribution, speculative_generate


def test_fp4_actual_packing_and_ste_gradient():
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    packed, scales = pack_fp4(levels, 16)
    assert packed.dtype == torch.uint8 and packed.numel() == 8 and scales.numel() == 1
    torch.testing.assert_close(unpack_fp4(packed, scales, levels.shape, 16), levels)
    weight = torch.randn(13, 17, requires_grad=True)
    fake_quantize_fp4(weight).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))
    packed, scales = pack_fp4(weight)
    assert packed.nbytes + scales.nbytes < weight.nbytes
    with pytest.raises(ValueError):
        pack_fp4(torch.tensor([float("nan")]))


@pytest.mark.parametrize("variant", [{}, {"attention_type": "hybrid", "residual_type": "mhc"}])
def test_packed_model_cache_is_consistent(variant):
    model = LaptopLLM(
        ModelConfig(
            vocab_size=32, dim=32, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=32, **variant
        )
    ).eval()
    original = copy.deepcopy(model)
    names = replace_linears(model)
    assert names and "lm_head" not in names
    assert sum(t.nbytes for t in model.state_dict().values()) < sum(
        t.nbytes for t in original.state_dict().values()
    )
    ids = torch.randint(0, 32, (1, 9))
    full = model(ids).logits
    first = model(ids[:, :4], use_cache=True)
    last = model(ids[:, 4:], past_key_values=first.past_key_values, use_cache=True)
    torch.testing.assert_close(full[:, 4:], last.logits, atol=2e-6, rtol=2e-5)


def test_rejection_mass_equals_target_distribution():
    target = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    draft = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float64)
    accepted_mass = draft * (target / draft).clamp_max(1)
    resulting_mass = accepted_mass + (1 - accepted_mass.sum()) * correction_distribution(
        target, draft
    )
    torch.testing.assert_close(resulting_mass, target)
    with pytest.raises(ValueError):
        correction_distribution(target, target)


def test_speculative_identical_draft_and_greedy_rejection():
    config = ModelConfig(vocab_size=16, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=32)
    model = LaptopLLM(config).eval()
    identical = copy.deepcopy(model)
    prompt = [3, 4, 5]
    generated, stats = speculative_generate(model, identical, prompt, 10, 3, 0)
    expected, prefix = [], list(prompt)
    for _ in range(10):
        token = int(model(torch.tensor([prefix])).logits[0, -1].argmax())
        expected.append(token)
        prefix.append(token)
    assert generated == expected and stats.acceptance_rate == 1 and stats.target_forwards == 3
    for current, selected in [(model, 1), (identical, 2)]:
        current.lm_head = torch.nn.Linear(16, 16)
        with torch.no_grad():
            current.lm_head.weight.zero_()
            current.lm_head.bias.zero_()
            current.lm_head.bias[selected] = 10
    generated, stats = speculative_generate(model, identical, prompt, 7, 3, 0)
    assert generated == [1] * 7 and stats.accepted == 0
    stopped, _ = speculative_generate(model, identical, prompt, 7, 3, 0, {1})
    assert stopped == [1]
