from pathlib import Path

import pytest

from laptop_llm.config import ExperimentConfig, ModelConfig


def test_smoke_config_resolves_paths_from_repo():
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_yaml(root / "configs" / "smoke.yaml")
    assert Path(config.data["sft_train"]).is_file()
    assert Path(config.tokenizer_corpus[0]).is_file()
    assert config.model.dim == 64


def test_model_config_rejects_invalid_gqa():
    with pytest.raises(ValueError, match="n_kv_heads"):
        ModelConfig(vocab_size=100, dim=48, n_heads=6, n_kv_heads=4)
