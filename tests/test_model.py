import torch

from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM


def tiny_model() -> LaptopLLM:
    torch.manual_seed(7)
    return LaptopLLM(
        ModelConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            dropout=0.0,
        )
    ).eval()


def test_forward_shape_and_shifted_loss():
    model = tiny_model()
    input_ids = torch.tensor([[1, 4, 9, 2], [1, 8, 3, 2]])
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 4, 64)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_kv_cache_matches_full_recomputation():
    """最关键推理契约：cache 只省计算，不能改变同一位置的 logits。"""

    model = tiny_model()
    input_ids = torch.tensor([[1, 7, 11, 3, 19, 2]])
    with torch.no_grad():
        full = model(input_ids).logits
        cache = None
        pieces = []
        for position in range(input_ids.size(1)):
            output = model(
                input_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            pieces.append(output.logits)
        cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-5)
    assert cache is not None
    # Cache 保持 2 个 KV 头，而不是展开成 4 个 Q 头。
    assert cache[0][0].shape == (1, 2, input_ids.size(1), 8)


def test_padding_mask_does_not_change_real_prefix():
    model = tiny_model()
    short = torch.tensor([[1, 5, 2]])
    padded = torch.tensor([[1, 5, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        short_logits = model(short).logits
        padded_logits = model(padded, attention_mask=mask).logits[:, :3]
    torch.testing.assert_close(short_logits, padded_logits, rtol=1e-4, atol=1e-5)


def test_gradient_checkpointing_recomputes_the_correct_layer():
    model = tiny_model().train()
    model.enable_gradient_checkpointing()
    input_ids = torch.tensor([[1, 7, 11, 3, 2]])
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()
    assert model.layers[0].attn.q_proj.weight.grad is not None
    assert model.layers[1].attn.q_proj.weight.grad is not None
