import copy

import pytest
import torch

from laptop_llm.architectures.cache import DeltaCache, cache_bytes
from laptop_llm.architectures.delta import delta_update
from laptop_llm.architectures.residual import AttentionResidual, sinkhorn
from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM

VARIANTS = [
    {"attention_type": "kda"},
    {"attention_type": "hybrid"},
    {"attention_pattern": "indexed", "index_block_size": 3, "index_topk": 2},
    {"residual_type": "attnres"},
    {"residual_type": "attnres", "attnres_block_size": 3},
    {"residual_type": "mhc"},
    {"engram_table_size": 31, "engram_max_order": 4},
    {
        "attention_type": "hybrid",
        "residual_type": "attnres",
        "attnres_block_size": 3,
        "engram_table_size": 31,
        "num_experts": 4,
        "mtp_depth": 2,
    },
]


def small_model(**kwargs):
    return LaptopLLM(
        ModelConfig(
            vocab_size=64, dim=32, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=40, **kwargs
        )
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_frontier_causality_cache_and_backward(variant):
    torch.set_num_threads(2)
    torch.manual_seed(5)
    model = small_model(**variant).eval()
    ids = torch.randint(0, 64, (2, 14))
    padding = torch.ones_like(ids, dtype=torch.bool)
    padding[1, :2] = False
    output = model(ids, labels=ids, attention_mask=padding)
    (output.loss + output.auxiliary_loss).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    changed = ids.clone()
    changed[:, 8:] = torch.randint(0, 64, changed[:, 8:].shape)
    torch.testing.assert_close(
        output.logits[:, :8], model(changed, attention_mask=padding).logits[:, :8]
    )
    cache, pieces = None, []
    for start, end in [(0, 4), (4, 7), (7, 8), (8, 14)]:
        result = model(
            ids[:, start:end],
            attention_mask=padding[:, :end],
            past_key_values=cache,
            use_cache=True,
        )
        cache = result.past_key_values
        pieces.append(result.logits)
    torch.testing.assert_close(output.logits, torch.cat(pieces, 1), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_recomputation_has_same_gradients(variant):
    torch.manual_seed(11)
    model = small_model(**variant).train()
    recomputed = copy.deepcopy(model)
    recomputed.enable_gradient_checkpointing()
    ids = torch.randint(0, 64, (2, 10))
    first, second = model(ids, labels=ids), recomputed(ids, labels=ids)
    torch.testing.assert_close(
        first.loss + first.auxiliary_loss, second.loss + second.auxiliary_loss
    )
    (first.loss + first.auxiliary_loss).backward()
    (second.loss + second.auxiliary_loss).backward()
    for p, q in zip(model.parameters(), recomputed.parameters(), strict=True):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)


def test_delta_matrix_oracle_and_fixed_cache_storage():
    torch.manual_seed(2)
    state = torch.randn(2, 3, 4, 4)
    q, k, v, alpha = [torch.randn(2, 3, 4) for _ in range(4)]
    alpha = alpha.sigmoid()
    beta = torch.rand(2, 3)
    result, output = delta_update(state, q, k, v, alpha, beta)
    explicit = torch.eye(4) - beta[..., None, None] * k[..., :, None] * k[..., None, :]
    explicit = (
        explicit @ torch.diag_embed(alpha) @ state
        + beta[..., None, None] * k[..., :, None] * v[..., None, :]
    )
    torch.testing.assert_close(result, explicit)
    torch.testing.assert_close(output, (explicit.transpose(-1, -2) @ q[..., None]).squeeze(-1))
    model = small_model(attention_type="kda").eval()
    cache1 = model(torch.ones(1, 4, dtype=torch.long), use_cache=True).past_key_values
    cache2 = model(torch.ones(1, 24, dtype=torch.long), use_cache=True).past_key_values
    assert isinstance(cache1[0], DeltaCache)
    assert cache_bytes(cache1) == cache_bytes(cache2)
    frozen_state = cache1[0].state.clone()
    model(torch.ones(1, 2, dtype=torch.long), past_key_values=cache1, use_cache=True)
    torch.testing.assert_close(cache1[0].state, frozen_state)


def test_indexer_dense_limit_and_gradient_isolation():
    sparse = small_model(attention_pattern="indexed", index_topk=100).eval()
    dense = small_model().eval()
    dense.load_state_dict(sparse.state_dict(), strict=False)
    ids = torch.randint(0, 64, (2, 9))
    output = sparse(ids)
    torch.testing.assert_close(output.logits, dense(ids).logits, atol=2e-6, rtol=2e-5)
    embedding_gradient = torch.autograd.grad(
        output.auxiliary_loss, sparse.token_embedding.weight, allow_unused=True, retain_graph=True
    )[0]
    assert embedding_gradient is None
    output.auxiliary_loss.backward()
    assert sparse.layers[0].attn.indexer.index_q.weight.grad.abs().sum() > 0
    masked = sparse(ids, attention_mask=torch.zeros_like(ids, dtype=torch.bool))
    (masked.logits.sum() + masked.auxiliary_loss).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in sparse.parameters())


def test_indexer_ties_do_not_depend_on_prefill_chunk_shape():
    """同分的历史块必须一致路由；不能靠放宽 logits 容差隐藏硬路由差异。"""
    torch.manual_seed(5)
    model = small_model(attention_pattern="indexed", index_block_size=3, index_topk=3).eval()
    projection_shapes = []
    # 零 Q 投影制造精确同分，重复 token 则覆盖不同长度 GEMM 的舍入陷阱。
    for layer in model.layers:
        layer.attn.indexer.index_q.weight.data.zero_()
        for projection in [layer.attn.indexer.index_q, layer.attn.indexer.index_k]:
            projection.register_forward_pre_hook(
                lambda module, args: projection_shapes.append(tuple(args[0].shape))
            )
    ids = torch.tensor([[7, 9, 7, 9, 7, 9, 7, 9, 7, 9, 7, 9, 7, 9]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, :2] = False
    full = model(ids, attention_mask=mask).logits
    for chunks in [[1] * 14, [4, 3, 1, 6], [8, 6]]:
        cache, pieces, start = None, [], 0
        for width in chunks:
            end = start + width
            result = model(
                ids[:, start:end],
                attention_mask=mask[:, :end],
                past_key_values=cache,
                use_cache=True,
            )
            cache = result.past_key_values
            pieces.append(result.logits)
            start = end
        torch.testing.assert_close(full, torch.cat(pieces, 1), atol=2e-6, rtol=2e-5)
    # 同输入的 canonical reference 投影不能随 chunk 长度切换 GEMM 形状。
    assert set(projection_shapes) == {(1, model.config.dim)}


def test_depth_mixer_and_sinkhorn_constraints():
    sources = torch.randn(5, 2, 3, 8, requires_grad=True)
    mixer = AttentionResidual(8)
    torch.testing.assert_close(mixer(sources), sources.mean(0))
    mixer(sources).square().sum().backward()
    assert mixer.query.grad.abs().sum() > 0
    matrix = sinkhorn(torch.randn(2, 3, 4, 4), 50)
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 3, 4))
    torch.testing.assert_close(matrix.sum(-2), torch.ones(2, 3, 4))
    assert (matrix >= 0).all()
    assert torch.linalg.matrix_norm(matrix, ord=2).max() <= 1.00001


def test_mtp_label_alignment():
    model = small_model(mtp_depth=2).eval()
    ids = torch.randint(0, 64, (1, 8))
    labels = ids.clone()
    labels[:, :4] = -100
    output = model(ids, labels=labels)
    hidden = output.hidden_states
    losses = []
    for depth, head in enumerate(model.mtp, 1):
        hidden = head(
            hidden[:, :-1],
            model.token_embedding(ids[:, depth:]),
            model.rope_cos[depth:8],
            model.rope_sin[depth:8],
            None,
        )
        logits = model.lm_head(model.norm(hidden))[:, :-1]
        losses.append(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, 64), labels[:, depth + 1 :].reshape(-1)
            )
        )
    torch.testing.assert_close(output.mtp_loss, torch.stack(losses).mean())
    torch.testing.assert_close(output.auxiliary_loss, output.mtp_loss * model.config.mtp_loss_coef)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("variant", [VARIANTS[2], VARIANTS[5], VARIANTS[7]])
def test_frontier_cuda_autocast(variant):
    model = small_model(**variant).cuda().train()
    ids = torch.randint(0, 64, (2, 12), device="cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        output = model(ids, labels=ids)
        loss = output.loss + output.auxiliary_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
