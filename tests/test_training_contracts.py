import copy
from functools import partial

import torch
from torch.utils.data import DataLoader

from laptop_llm.config import ExperimentConfig, ModelConfig, StageConfig
from laptop_llm.data import pad_lm_batch
from laptop_llm.engine import evaluate, read_checkpoint, save_checkpoint
from laptop_llm.model import LaptopLLM


def test_eval_token_weighting_is_independent_of_batch_size():
    model = LaptopLLM(
        ModelConfig(vocab_size=16, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16)
    ).eval()
    examples = [{"input_ids": ids, "labels": ids} for ids in ([1, 2, 3], [1, 4, 5, 6, 7], [1, 8])]
    config = StageConfig(eval_batches=10)
    outputs = []
    for batch in (1, 2, 3):
        loader = DataLoader(examples, batch_size=batch, collate_fn=partial(pad_lm_batch, pad_id=0))
        outputs.append(
            evaluate(model, None, loader, "sft", config, torch.device("cpu"), torch.float32)[
                "eval_loss"
            ]
        )
    assert max(outputs) - min(outputs) < 1e-6


def test_dpo_checkpoint_keeps_original_reference(tmp_path):
    class TokenizerStub:
        def to_str(self):
            return "test-only"

    model = LaptopLLM(
        ModelConfig(vocab_size=16, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16)
    )
    reference = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        model.token_embedding.weight.add_(0.1)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        TokenizerStub(),
        torch.optim.AdamW(model.parameters()),
        ExperimentConfig(),
        "dpo",
        1,
        reference_model=reference,
    )
    payload = read_checkpoint(path)
    assert torch.equal(
        payload["reference_model"]["token_embedding.weight"], reference.token_embedding.weight
    )
    assert not torch.equal(
        payload["reference_model"]["token_embedding.weight"],
        payload["model"]["token_embedding.weight"],
    )
