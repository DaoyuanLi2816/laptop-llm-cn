"""三类数据管线：连续文本预训练、assistant-only SFT、偏好对 DPO。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from laptop_llm.tokenizer import LLMTokenizer


def prepare_pretrain_cache(
    source_path: str | Path,
    output_path: str | Path,
    tokenizer: LLMTokenizer,
) -> dict[str, Any]:
    """把文本变成连续 uint32 token 文件，训练时通过 memmap 零拷贝按块读取。

    每个非空文本段被编码为 ``<bos> 正文 <eos>``。与保存 Python list 相比，
    二进制缓存没有 pickle 风险，数亿 token 时也不必一次装进内存。
    """

    source = Path(source_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到预训练文本: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    digest.update(tokenizer.to_str().encode("utf-8"))
    digest.update(str(source.resolve()).encode("utf-8"))
    digest.update(str(source.stat().st_mtime_ns).encode("ascii"))

    token_count = 0
    document_count = 0
    with source.open("r", encoding="utf-8") as reader, target.open("wb") as writer:
        for line in reader:
            text = line.strip()
            if not text:
                continue
            token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            np.asarray(token_ids, dtype="<u4").tofile(writer)
            token_count += len(token_ids)
            document_count += 1

    if token_count < 2:
        raise ValueError(f"{source} 中没有足够的非空文本")
    metadata = {
        "format": "uint32-little-endian",
        "tokens": token_count,
        "documents": document_count,
        "source": str(source.resolve()),
        "fingerprint": digest.hexdigest(),
    }
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """把连续 token 流切成固定长度窗口；边界处不浪费 padding。"""

    def __init__(self, path: str | Path, block_size: int):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 token 缓存: {self.path}")
        self.tokens = np.memmap(self.path, dtype="<u4", mode="r")
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError(
                f"token 缓存只有 {len(self.tokens)} 个 token，不足一个 {block_size + 1} 窗口"
            )

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        window = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        # memmap 切片通常不可写；复制后交给 PyTorch，避免 undefined behavior 警告。
        input_ids = torch.from_numpy(window[:-1].copy())
        labels = torch.from_numpy(window[1:].copy())
        # labels 已经右移一位，因此引擎用专门的 causal_lm_loss，不再让模型二次 shift。
        return {"input_ids": input_ids, "next_token_labels": labels}


class SFTDataset(Dataset[dict[str, list[int]]]):
    """读取 messages JSONL，并只监督 assistant 的内容。"""

    def __init__(self, path: str | Path, tokenizer: LLMTokenizer, max_length: int):
        rows = read_jsonl(path)
        self.examples: list[dict[str, list[int]]] = []
        for line_number, row in enumerate(rows, start=1):
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number} 缺少 messages 数组")
            input_ids, labels = tokenizer.build_sft_example(messages, max_length)
            if not any(label != -100 for label in labels[1:]):
                raise ValueError(f"{path}:{line_number} 没有可监督的 assistant 回复")
            self.examples.append({"input_ids": input_ids, "labels": labels})
        if not self.examples:
            raise ValueError(f"{path} 没有有效 SFT 样本")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class PreferenceDataset(Dataset[dict[str, dict[str, list[int]]]]):
    """读取 DPO 偏好对：相同 prompt 下 chosen 应比 rejected 更可能。"""

    def __init__(self, path: str | Path, tokenizer: LLMTokenizer, max_length: int):
        rows = read_jsonl(path)
        self.examples: list[dict[str, dict[str, list[int]]]] = []
        for line_number, row in enumerate(rows, start=1):
            prompt = row.get("prompt")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            if (
                not isinstance(prompt, list)
                or not isinstance(chosen, str)
                or not isinstance(rejected, str)
            ):
                raise ValueError(
                    f"{path}:{line_number} 必须包含 prompt(list)、chosen(str)、rejected(str)"
                )
            if chosen.strip() == rejected.strip():
                raise ValueError(f"{path}:{line_number} chosen 与 rejected 不能相同")
            chosen_ids, chosen_labels = tokenizer.build_preference_example(
                prompt, chosen, max_length, prompt_max_length=max_length - 3
            )
            rejected_ids, rejected_labels = tokenizer.build_preference_example(
                prompt, rejected, max_length, prompt_max_length=max_length - 3
            )
            self.examples.append(
                {
                    "chosen": {"input_ids": chosen_ids, "labels": chosen_labels},
                    "rejected": {"input_ids": rejected_ids, "labels": rejected_labels},
                }
            )
        if not self.examples:
            raise ValueError(f"{path} 没有有效偏好样本")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, dict[str, list[int]]]:
        return self.examples[index]


def pad_lm_batch(examples: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(example["input_ids"]) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[row, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(example["labels"], dtype=torch.long)
        attention_mask[row, :length] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def pad_preference_batch(
    examples: Sequence[dict[str, dict[str, list[int]]]], pad_id: int
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "chosen": pad_lm_batch([example["chosen"] for example in examples], pad_id),
        "rejected": pad_lm_batch([example["rejected"] for example in examples], pad_id),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 JSONL 数据: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} 不是合法 JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} 顶层必须是对象")
            rows.append(value)
    return rows
