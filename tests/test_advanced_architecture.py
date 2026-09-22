"""算法契约比“能 forward”更重要：稀疏可见性、缓存等价、路由梯度、合并等价。"""

import copy

import pytest
import torch
import torch.nn.functional as F

from laptop_llm.architectures.lora import inject_lora, merge_lora
from laptop_llm.architectures.sparse import sparse_attention
from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM


def make_model(**kwargs):
    return LaptopLLM(
        ModelConfig(
            vocab_size=48, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32, **kwargs
        )
    )


@pytest.mark.parametrize("past,window,sinks", [(0, 3, 2), (4, 2, 1), (0, 20, 0), (2, 1, 5)])
def test_sparse_matches_dense_mask_oracle(past, window, sinks):
    torch.manual_seed(9)
    q = torch.randn(2, 4, 5, 8, requires_grad=True)
    k = torch.randn(2, 4, 5 + past, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    positions = torch.arange(5) + past
    keys = torch.arange(5 + past)
    mask = (keys[None] <= positions[:, None]) & (
        (keys[None] > positions[:, None] - window) | (keys[None] < sinks)
    )
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    actual = sparse_attention(q, k, v, past_len=past, window=window, sinks=sinks)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert q.grad.isfinite().all() and k.grad.isfinite().all()


@pytest.mark.parametrize("experts", [0, 4])
def test_hybrid_sparse_cache_causality_and_checkpoint(experts):
    torch.manual_seed(2)
    model = make_model(
        attention_pattern="sliding",
        sliding_window=3,
        attention_sinks=1,
        dense_every=2,
        num_experts=experts,
        qk_norm=True,
    ).eval()
    ids = torch.randint(0, 48, (2, 12))
    full = model(ids)
    past, pieces = None, []
    for start, end in [(0, 4), (4, 7), (7, 12)]:
        result = model(ids[:, start:end], past_key_values=past, use_cache=True)
        past = result.past_key_values
        pieces.append(result.logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full.logits, atol=2e-6, rtol=2e-5)
    changed = ids.clone()
    changed[:, 7:] = 0
    torch.testing.assert_close(model(changed).logits[:, :7], full.logits[:, :7])
    baseline = copy.deepcopy(model).train()
    model.train().enable_gradient_checkpointing()
    for candidate in (baseline, model):
        output = candidate(ids, labels=ids)
        (output.loss + output.auxiliary_loss).backward()
    for first, second in zip(baseline.parameters(), model.parameters(), strict=True):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad)
    if experts:
        assert model.layers[0].ffn.router.weight.grad.abs().sum() > 0


def test_moe_padding_excluded_from_router_loss():
    model = make_model(num_experts=4).eval()
    ids = torch.tensor([[1, 2, 3]])
    original = model(ids)
    padded = model(torch.tensor([[1, 2, 3, 0, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0, 0]]))
    torch.testing.assert_close(original.auxiliary_loss, padded.auxiliary_loss)


def test_lora_updates_only_adapters_and_merge_preserves_logits():
    model = make_model().eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids).logits.detach()
    inject_lora(model, rank=2)
    torch.testing.assert_close(model(ids).logits, expected)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(ids, labels=ids).loss.backward()
    assert all(
        p.grad is None for name, p in model.named_parameters() if not name.endswith((".a", ".b"))
    )
    optimizer.step()
    adapted = model(ids).logits.detach()
    assert not torch.equal(adapted, expected)
    merge_lora(model)
    torch.testing.assert_close(model(ids).logits, adapted, atol=1e-6, rtol=1e-5)


def test_mla_latent_cache_and_weight_absorption():
    from laptop_llm.model import apply_rope

    model = make_model(attention_type="mla", kv_lora_rank=4).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    full = model(ids)
    cache, pieces = None, []
    for t in range(5):
        result = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        pieces.append(result.logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full.logits, atol=1e-6, rtol=1e-5)
    assert cache[0][0].shape == (1, 1, 5, 4)
    assert cache[0][1].shape == (1, 1, 5, 8)
    layer = model.layers[0].attn
    x = torch.randn(1, 5, 32)
    cos, sin = model.rope_cos[:5], model.rope_sin[:5]
    latent = layer.kv_norm(layer.kv_down(x))
    key, value = layer.kv_up(latent).view(1, 5, 4, 16).transpose(1, 2).chunk(2, -1)
    q, qr = layer.q_proj(x).view(1, 5, 4, 16).transpose(1, 2).chunk(2, -1)
    qr = apply_rope(qr, cos, sin)
    kr = apply_rope(layer.k_rope(x)[:, None], cos, sin).expand(-1, 4, -1, -1)
    expanded = F.scaled_dot_product_attention(
        torch.cat((q, qr), -1), torch.cat((key, kr), -1), value, is_causal=True
    )
    expected = layer.o_proj(expanded.transpose(1, 2).reshape(1, 5, 32))
    actual, _ = layer(x, cos, sin)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    actual.sum().backward()
    assert layer.kv_up.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["sliding", "mla"])
def test_all_padding_backward_is_finite(kind):
    options = {"attention_pattern": "sliding"} if kind == "sliding" else {"attention_type": "mla"}
    model = make_model(**options)
    ids = torch.ones(1, 4, dtype=torch.long)
    output = model(ids, attention_mask=torch.zeros_like(ids))
    output.logits.sum().backward()
    assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="本机没有 CUDA；CPU CI 不冒充 GPU 验证")
@pytest.mark.parametrize("kind", ["dense", "sliding", "mla"])
def test_cuda_mixed_precision_backward(kind):
    options = (
        {"attention_type": "mla"}
        if kind == "mla"
        else {"attention_pattern": kind, "num_experts": 4}
    )
    model = make_model(**options).cuda().train()
    model.enable_gradient_checkpointing()
    ids = torch.randint(0, 48, (2, 8), device="cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        output = model(ids, labels=ids)
        loss = output.loss + output.auxiliary_loss
    loss.backward()
    assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
