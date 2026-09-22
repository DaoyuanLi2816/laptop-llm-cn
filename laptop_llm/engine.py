"""统一训练引擎：预训练、SFT、DPO 共用设备、AMP、调度、评测和 checkpoint。"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from laptop_llm.config import ExperimentConfig, ModelConfig, StageConfig
from laptop_llm.data import (
    PackedTokenDataset,
    PreferenceDataset,
    SFTDataset,
    pad_lm_batch,
    pad_preference_batch,
    prepare_pretrain_cache,
)
from laptop_llm.model import LaptopLLM
from laptop_llm.tokenizer import LLMTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但 torch.cuda.is_available() 为 False")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA GPU 不支持 bfloat16，请改用 float16 或 auto")
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # CPU 上默认 fp32 最稳；用户可显式选择 bf16。
    return torch.float32


def load_experiment_tokenizer(config: ExperimentConfig) -> LLMTokenizer:
    path = Path(config.tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，请先运行 train-tokenizer 或 pipeline")
    return LLMTokenizer.from_file(path)


def run_stage(
    stage: str,
    config: ExperimentConfig,
    *,
    init_from: str | Path | None = None,
    resume: str | Path | None = None,
    device_name: str = "auto",
) -> Path:
    """运行一个阶段并返回 final.pt 路径。"""

    if stage not in {"pretrain", "sft", "dpo"}:
        raise ValueError("stage 必须是 pretrain/sft/dpo")
    if init_from is not None and resume is not None:
        raise ValueError("init_from 与 resume 不能同时使用")

    stage_config = config.stage(stage)
    tokenizer = load_experiment_tokenizer(config)
    device = resolve_device(device_name)
    dtype = resolve_dtype(stage_config.dtype, device)
    set_seed(config.seed)

    model_config = copy.deepcopy(config.model)
    model_config.vocab_size = tokenizer.vocab_size
    model = LaptopLLM(model_config)
    optimizer_state = None
    start_step = 0
    checkpoint_source = resume or init_from
    if checkpoint_source is not None:
        checkpoint_data = read_checkpoint(checkpoint_source, map_location="cpu")
        saved_config = ModelConfig(**checkpoint_data["model_config"])
        if saved_config.vocab_size != tokenizer.vocab_size:
            raise ValueError("checkpoint 的词表大小与当前 tokenizer 不一致")
        if checkpoint_data.get("tokenizer_json") != tokenizer.to_str():
            raise ValueError("checkpoint 内嵌 tokenizer 与当前 tokenizer 内容不一致")
        model = LaptopLLM(saved_config)
        model.load_state_dict(checkpoint_data["model"])
        model_config = saved_config
        if resume is not None:
            if checkpoint_data.get("stage") != stage:
                raise ValueError("resume checkpoint 的训练阶段与当前 stage 不同")
            optimizer_state = checkpoint_data.get("optimizer")
            start_step = int(checkpoint_data.get("step", 0))

    model.enable_gradient_checkpointing(stage_config.gradient_checkpointing)
    model.to(device)
    reference_model: LaptopLLM | None = None
    if stage == "dpo":
        if checkpoint_source is None:
            raise ValueError("DPO 必须通过 --init-from 从 SFT checkpoint 开始")
        reference_model = copy.deepcopy(model).eval()
        if resume is not None:
            if "reference_model" not in checkpoint_data:
                raise ValueError("旧 DPO checkpoint 未保存固定 reference；不能伪装成等价续训")
            reference_model.load_state_dict(checkpoint_data["reference_model"])
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    # 在创建 optimizer 后再 compile；checkpoint 始终保存未包装的原模型参数名。
    optimizer = build_optimizer(model, stage_config, device)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    train_model: torch.nn.Module = model
    if stage_config.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("当前 PyTorch 不支持 torch.compile")
        train_model = torch.compile(model)

    train_dataset, eval_dataset, collate_fn = build_datasets(
        stage, config, tokenizer, model_config.max_seq_len
    )
    train_loader = make_loader(train_dataset, stage_config, collate_fn, shuffle=True)
    eval_loader = make_loader(eval_dataset, stage_config, collate_fn, shuffle=False)
    train_iterator = infinite_batches(train_loader)

    amp_enabled = dtype != torch.float32
    scaler = make_grad_scaler(device, enabled=amp_enabled and dtype == torch.float16)
    stage_dir = Path(config.output_dir) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = stage_dir / "metrics.jsonl"

    print("=" * 80)
    print(f"阶段: {stage} | 设备: {device} | 计算 dtype: {str(dtype).split('.')[-1]}")
    print(f"参数量: {model.num_parameters():,} | 上下文: {model_config.max_seq_len}")
    print(
        f"有效 batch: {stage_config.batch_size} × "
        f"{stage_config.gradient_accumulation_steps} = "
        f"{stage_config.batch_size * stage_config.gradient_accumulation_steps}"
    )
    print("=" * 80)

    last_time = time.perf_counter()
    for step in range(start_step + 1, stage_config.max_steps + 1):
        learning_rate = cosine_learning_rate(step, stage_config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_aux = 0.0

        for _ in range(stage_config.gradient_accumulation_steps):
            batch = move_to_device(next(train_iterator), device)
            with autocast_context(device, dtype):
                if stage == "dpo":
                    assert reference_model is not None
                    loss, aux = dpo_batch_loss(
                        train_model,
                        reference_model,
                        batch,
                        beta=stage_config.dpo_beta,
                        label_smoothing=stage_config.dpo_label_smoothing,
                    )
                else:
                    loss = language_model_batch_loss(train_model, batch)
                    aux = 0.0
                scaled_loss = loss / stage_config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach()) / stage_config.gradient_accumulation_steps
            accumulated_aux += float(aux) / stage_config.gradient_accumulation_steps

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), stage_config.max_grad_norm, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % stage_config.log_interval == 0:
            now = time.perf_counter()
            seconds = now - last_time
            last_time = now
            record = {
                "stage": stage,
                "step": step,
                "train_loss": accumulated_loss,
                "lr": learning_rate,
                "grad_norm": float(grad_norm),
                "seconds_since_log": seconds,
            }
            if stage == "dpo":
                record["preference_accuracy"] = accumulated_aux
            append_metric(metrics_path, record)
            extra = f" | 偏好正确率 {accumulated_aux:.1%}" if stage == "dpo" else ""
            print(
                f"step {step:>6}/{stage_config.max_steps} | loss {accumulated_loss:.4f} "
                f"| lr {learning_rate:.2e} | grad {float(grad_norm):.3f}{extra}"
            )

        if step % stage_config.eval_interval == 0 or step == stage_config.max_steps:
            evaluation = evaluate(
                train_model,
                reference_model,
                eval_loader,
                stage,
                stage_config,
                device,
                dtype,
            )
            evaluation.update({"stage": stage, "step": step})
            append_metric(metrics_path, evaluation)
            print(
                f"  eval | loss {evaluation['eval_loss']:.4f}"
                + (
                    f" | 偏好正确率 {evaluation['eval_preference_accuracy']:.1%}"
                    if "eval_preference_accuracy" in evaluation
                    else f" | ppl {evaluation['perplexity']:.2f}"
                )
            )
            train_model.train()

        if step % stage_config.save_interval == 0 and step < stage_config.max_steps:
            save_checkpoint(
                stage_dir / f"step_{step:07d}.pt",
                model,
                tokenizer,
                optimizer,
                config,
                stage,
                step,
                reference_model=reference_model,
            )

    final_path = stage_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        tokenizer,
        optimizer,
        config,
        stage,
        stage_config.max_steps,
        reference_model=reference_model,
    )
    print(f"完成：{final_path}")
    return final_path


def build_datasets(
    stage: str,
    config: ExperimentConfig,
    tokenizer: LLMTokenizer,
    max_length: int,
) -> tuple[Dataset[Any], Dataset[Any], Any]:
    if stage == "pretrain":
        cache_dir = Path(config.output_dir) / "token_cache"
        train_cache = cache_dir / "train.bin"
        eval_cache = cache_dir / "eval.bin"
        prepare_pretrain_cache(required_data(config, "pretrain_train"), train_cache, tokenizer)
        prepare_pretrain_cache(required_data(config, "pretrain_eval"), eval_cache, tokenizer)
        return (
            PackedTokenDataset(train_cache, max_length),
            PackedTokenDataset(eval_cache, max_length),
            None,
        )
    if stage == "sft":
        train = SFTDataset(required_data(config, "sft_train"), tokenizer, max_length)
        evaluation = SFTDataset(required_data(config, "sft_eval"), tokenizer, max_length)
        return train, evaluation, partial(pad_lm_batch, pad_id=tokenizer.pad_id)
    train = PreferenceDataset(required_data(config, "dpo_train"), tokenizer, max_length)
    evaluation = PreferenceDataset(required_data(config, "dpo_eval"), tokenizer, max_length)
    return train, evaluation, partial(pad_preference_batch, pad_id=tokenizer.pad_id)


def required_data(config: ExperimentConfig, key: str) -> str:
    path = config.data.get(key)
    if path is None:
        raise KeyError(f"配置缺少 data.{key}")
    return path


def make_loader(
    dataset: Dataset[Any], stage_config: StageConfig, collate_fn: Any, *, shuffle: bool
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=stage_config.batch_size,
        shuffle=shuffle,
        num_workers=stage_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= stage_config.batch_size,
        collate_fn=collate_fn,
    )


def infinite_batches(loader: DataLoader[Any]) -> Iterator[Any]:
    """反复遍历 DataLoader，但不使用 itertools.cycle 缓存所有 batch。"""

    while True:
        yield from loader


def language_model_batch_loss(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], *, include_auxiliary: bool = True
) -> torch.Tensor:
    if "next_token_labels" in batch:
        output = model(batch["input_ids"])
        return F.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)),
            batch["next_token_labels"].reshape(-1),
        ) + (output.auxiliary_loss if include_auxiliary else 0)
    output = model(
        batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch.get("attention_mask"),
    )
    if output.loss is None:
        raise RuntimeError("模型没有返回 loss")
    return output.loss + (output.auxiliary_loss if include_auxiliary else 0)


def sequence_log_probabilities(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], *, return_auxiliary: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    output = model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
    logits = output.logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    result = (token_logps * mask).sum(dim=-1)
    return (result, output.auxiliary_loss) if return_auxiliary else result


def dpo_batch_loss(
    policy: torch.nn.Module,
    reference: LaptopLLM,
    batch: dict[str, dict[str, torch.Tensor]],
    *,
    beta: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_chosen, chosen_aux = sequence_log_probabilities(
        policy, batch["chosen"], return_auxiliary=True
    )
    policy_rejected, rejected_aux = sequence_log_probabilities(
        policy, batch["rejected"], return_auxiliary=True
    )
    with torch.no_grad():
        reference_chosen = sequence_log_probabilities(reference, batch["chosen"])
        reference_rejected = sequence_log_probabilities(reference, batch["rejected"])

    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    losses = -(
        (1 - label_smoothing) * F.logsigmoid(logits) + label_smoothing * F.logsigmoid(-logits)
    )
    accuracy = (logits > 0).float().mean()
    return losses.mean() + (chosen_aux + rejected_aux) / 2, accuracy


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    reference: LaptopLLM | None,
    loader: DataLoader[Any],
    stage: str,
    stage_config: StageConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    weights: list[int] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= stage_config.eval_batches:
            break
        batch = move_to_device(batch, device)
        with autocast_context(device, dtype):
            if stage == "dpo":
                assert reference is not None
                loss, accuracy = dpo_batch_loss(
                    model,
                    reference,
                    batch,
                    beta=stage_config.dpo_beta,
                    label_smoothing=stage_config.dpo_label_smoothing,
                )
                accuracies.append(float(accuracy))
                weight = batch["chosen"]["input_ids"].size(0)
            else:
                loss = language_model_batch_loss(model, batch, include_auxiliary=False)
                weight = (
                    batch["next_token_labels"].numel()
                    if "next_token_labels" in batch
                    else int((batch["labels"][:, 1:] != -100).sum())
                )
        losses.append(float(loss))
        weights.append(weight)
    if not weights or sum(weights) == 0:
        raise ValueError("评测集没有有效样本/token")
    mean_loss = sum(loss * weight for loss, weight in zip(losses, weights, strict=True)) / sum(
        weights
    )
    result = {"eval_loss": mean_loss}
    if accuracies:
        result["eval_preference_accuracy"] = sum(
            acc * weight for acc, weight in zip(accuracies, weights, strict=True)
        ) / sum(weights)
    else:
        result["perplexity"] = math.exp(min(mean_loss, 20.0))
    return result


def build_optimizer(
    model: LaptopLLM, config: StageConfig, device: torch.device | None = None
) -> AdamW:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=device is not None and device.type == "cuda",
    )


def cosine_learning_rate(step: int, config: StageConfig) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device, *, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:  # PyTorch 2.3 的兼容分支
        return torch.amp.GradScaler(enabled=enabled)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def save_checkpoint(
    path: str | Path,
    model: LaptopLLM,
    tokenizer: LLMTokenizer,
    optimizer: AdamW,
    experiment: ExperimentConfig,
    stage: str,
    step: int,
    *,
    reference_model: LaptopLLM | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "stage": stage,
            "step": step,
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer_json": tokenizer.to_str(),
            "experiment": experiment.to_dict(),
            **(
                {"reference_model": reference_model.state_dict()}
                if reference_model is not None
                else {}
            ),
        },
        temporary,
    )
    temporary.replace(target)


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {source}")
    # torch.save 是 pickle 格式：这里只应加载自己训练或明确可信来源的 checkpoint。
    return torch.load(source, map_location=map_location, weights_only=False)


def load_inference_bundle(
    checkpoint_path: str | Path, *, device_name: str = "auto", dtype_name: str = "auto"
) -> tuple[LaptopLLM, LLMTokenizer, torch.device]:
    device = resolve_device(device_name)
    data = read_checkpoint(checkpoint_path, map_location="cpu")
    tokenizer = LLMTokenizer.from_str(data["tokenizer_json"])
    model = LaptopLLM(ModelConfig(**data["model_config"]))
    model.load_state_dict(data["model"])
    dtype = resolve_dtype(dtype_name, device)
    # CPU fp16 算子支持不完整；CUDA 推理则直接把权重转成目标 dtype，减少显存。
    if device.type == "cuda" and dtype != torch.float32:
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)
    model.eval()
    return model, tokenizer, device


def append_metric(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
