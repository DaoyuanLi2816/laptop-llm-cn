"""配置系统：把模型结构、数据路径和每个训练阶段的旋钮集中管理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """decoder-only Transformer 的结构配置。

    ``n_heads`` 是 Query 头数，``n_kv_heads`` 是 Key/Value 头数。
    后者更少就是 GQA：多个 Query 头共享 K/V，可显著缩小生成时的 KV Cache。
    """

    vocab_size: int = 0
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int | None = None
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 0:
            raise ValueError("vocab_size 不能是负数")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim 必须能被 n_heads 整除")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads 必须能被 n_kv_heads 整除")
        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("每个注意力头的维度必须是偶数，RoPE 才能两两旋转")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len 太小，至少应为 8")

    @property
    def ffn_dim(self) -> int:
        if self.hidden_dim is not None:
            return self.hidden_dim
        # LLaMA 的 SwiGLU 常用约 8/3 倍扩张，再向上对齐到 64，利于 GPU 矩阵核。
        raw = int(8 * self.dim / 3)
        return 64 * ((raw + 63) // 64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageConfig:
    """单个训练阶段的配置；预训练、SFT、DPO 共用同一套训练引擎。"""

    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 1_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 500
    dtype: str = "auto"
    num_workers: int = 0
    compile: bool = False
    gradient_checkpointing: bool = False
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "save_interval": self.save_interval,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype 必须是 auto/float32/float16/bfloat16 之一")


@dataclass
class ExperimentConfig:
    """一个完整实验：分词器 + 模型 + 三阶段数据与超参数。"""

    name: str = "laptop-llm"
    seed: int = 42
    output_dir: str = "artifacts/default"
    tokenizer_path: str = "artifacts/default/tokenizer.json"
    tokenizer_vocab_size: int = 8_000
    tokenizer_min_frequency: int = 2
    tokenizer_corpus: list[str] = field(default_factory=list)
    data: dict[str, str] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    stages: dict[str, StageConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("配置文件顶层必须是 YAML mapping")

        model = ModelConfig(**raw.pop("model", {}))
        stage_raw = raw.pop("stages", {})
        stages = {name: StageConfig(**values) for name, values in stage_raw.items()}
        cfg = cls(model=model, stages=stages, **raw)

        # 所有相对路径都相对仓库根目录（配置文件上一级），而不是当前 shell。
        base = config_path.parent.parent
        cfg.output_dir = _resolve_path(base, cfg.output_dir)
        cfg.tokenizer_path = _resolve_path(base, cfg.tokenizer_path)
        cfg.tokenizer_corpus = [_resolve_path(base, p) for p in cfg.tokenizer_corpus]
        cfg.data = {key: _resolve_path(base, value) for key, value in cfg.data.items()}
        return cfg

    def stage(self, name: str) -> StageConfig:
        try:
            return self.stages[name]
        except KeyError as exc:
            raise KeyError(f"配置里没有 stages.{name}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_path(base: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())
