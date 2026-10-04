import pytest
import torch

from laptop_llm.architectures.compression import layer_reuse
from laptop_llm.config import ModelConfig
from laptop_llm.model import LaptopLLM
from laptop_llm.multimodal import VisionPrefixModel


def test_csa2_modes_share_storage_and_do_not_read_future():
    keys, values = torch.randn(1, 2, 12, 8), torch.randn(1, 2, 12, 8)
    query = torch.randn_like(keys)
    positions = torch.arange(12)
    full, memory, indices = layer_reuse(
        "full", query, positions, 2, source_keys=keys, source_values=values
    )
    reindexed, shared, fresh = layer_reuse("reindex", query, positions, 2, previous_memory=memory)
    reused, shared_again, reused_indices = layer_reuse(
        "reuse", query, positions, 2, previous_memory=shared, previous_indices=fresh
    )
    assert shared is memory and shared_again is memory and reused_indices is fresh
    torch.testing.assert_close(full, reindexed)
    torch.testing.assert_close(reindexed, reused)
    changed, changed_keys = values.clone(), keys.clone()
    changed[:, :, 8:] += 1000
    changed_keys[:, :, 8:] -= 1000
    altered, _, _ = layer_reuse(
        "full", query, positions, 2, source_keys=changed_keys, source_values=changed
    )
    torch.testing.assert_close(full[:, :, :8], altered[:, :, :8])
    assert full[:, :, 0].count_nonzero() == 0  # 第一个压缩块尚未完成，需要 local 分支。
    with pytest.raises(ValueError):
        layer_reuse("reuse", query, positions, 2, previous_memory=memory)
    with pytest.raises(ValueError, match="源 KV"):
        layer_reuse("full", query, positions, 2)


def test_pretrain_engine_actually_trains_mtp():
    from laptop_llm.engine import language_model_batch_loss

    model = LaptopLLM(
        ModelConfig(
            vocab_size=32, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=32, mtp_depth=2
        )
    )
    tokens = torch.randint(0, 32, (1, 9))
    batch = {"input_ids": tokens[:, :-1], "next_token_labels": tokens[:, 1:]}
    loss = language_model_batch_loss(model, batch)
    loss.backward()
    assert model.mtp[0].fuse.weight.grad.abs().sum() > 0


def test_visual_prefix_backward_and_masked_image_labels():
    lm = LaptopLLM(
        ModelConfig(vocab_size=32, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=32)
    )
    model = VisionPrefixModel(lm)
    images = torch.rand(2, 3, 8, 8)
    ids = torch.randint(0, 32, (2, 8))
    output = model(images, ids, labels=ids)
    assert output.logits.shape == (2, 12, 32)
    output.loss.backward()
    assert model.projector.projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model(torch.rand(2, 3, 9, 9), ids)
