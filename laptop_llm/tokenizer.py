"""Byte-level BPE 分词器与 ChatML 风格对话模板。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

# 这些 token 是模型生命周期里的“控制字符”，不能被 BPE 拆开。
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


class LLMTokenizer:
    """给底层 Rust Tokenizer 加上一层适合因果语言模型的中文友好接口。"""

    def __init__(self, tokenizer: Tokenizer):
        self.backend = tokenizer
        missing = [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
        if missing:
            raise ValueError(f"分词器缺少特殊 token: {missing}")

    @classmethod
    def from_file(cls, path: str | Path) -> LLMTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_str(cls, serialized: str) -> LLMTokenizer:
        return cls(Tokenizer.from_str(serialized))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(target))

    def to_str(self) -> str:
        return self.backend.to_str()

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def token_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise KeyError(f"词表里没有 {token!r}")
        return token_id

    @property
    def pad_id(self) -> int:
        return self.token_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self.token_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.token_id("<eos>")

    @property
    def end_id(self) -> int:
        return self.token_id("<|end|>")

    @property
    def assistant_id(self) -> int:
        return self.token_id("<|assistant|>")

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def build_chat_prompt(
        self,
        messages: Sequence[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        """把多轮消息变成模型看到的 token 序列。

        格式是 ``<bos><|system|>...<|end|><|user|>...``。如果太长，
        优先保留 system 与最近几轮，而不是粗暴截掉最新问题。
        """

        normalized = _validate_messages(messages)
        segments: list[list[int]] = []
        for message in normalized:
            role_id = self.token_id(ROLE_TOKENS[message["role"]])
            segment = [role_id, *self.encode(message["content"]), self.end_id]
            segments.append(segment)

        suffix = [self.assistant_id] if add_generation_prompt else []
        ids = [self.bos_id, *(token for segment in segments for token in segment), *suffix]
        if max_length is None or len(ids) <= max_length:
            return ids

        # system 与最新消息不可静默丢弃，超长时要求调用方明确处理。
        system = segments[:1] if normalized and normalized[0]["role"] == "system" else []
        recent = segments[1:] if system else segments[:]
        kept: list[list[int]] = recent[-1:] if recent else []
        budget = max_length - 1 - len(suffix) - sum(map(len, system))
        if sum(map(len, kept)) > budget:
            raise ValueError("system 与最新消息超过上下文预算，请缩短消息或换用更长上下文模型")
        for segment in reversed(recent[:-1]):
            if sum(map(len, kept)) + len(segment) > budget:
                break
            kept.insert(0, segment)
        while len(kept) > 1 and kept[0][0] == self.assistant_id:
            kept.pop(0)
        ids = [self.bos_id, *(t for seg in [*system, *kept] for t in seg), *suffix]
        return ids

    def build_sft_example(
        self, messages: Sequence[dict[str, str]], max_length: int
    ) -> tuple[list[int], list[int]]:
        """构造 SFT 的 input/label；只让 assistant 的正文与结束符产生 loss。"""

        normalized = _validate_messages(messages)
        input_ids = [self.bos_id]
        labels = [-100]
        for message in normalized:
            role_id = self.token_id(ROLE_TOKENS[message["role"]])
            content_ids = self.encode(message["content"])
            segment = [role_id, *content_ids, self.end_id]
            input_ids.extend(segment)
            if message["role"] == "assistant":
                labels.extend([-100, *content_ids, self.end_id])
            else:
                labels.extend([-100] * len(segment))
        input_ids.append(self.eos_id)
        labels.append(self.eos_id if normalized[-1]["role"] == "assistant" else -100)
        return input_ids[:max_length], labels[:max_length]

    def build_preference_example(
        self,
        prompt_messages: Sequence[dict[str, str]],
        response: str,
        max_length: int,
        prompt_max_length: int | None = None,
    ) -> tuple[list[int], list[int]]:
        prompt_ids = self.build_chat_prompt(
            prompt_messages,
            add_generation_prompt=True,
            max_length=prompt_max_length or max_length - 3,
        )
        response_ids = [*self.encode(response), self.end_id, self.eos_id]
        # chosen/rejected 必须共享完全一致的 prompt，不能随回复长度改变。
        if len(prompt_ids) >= max_length:
            raise ValueError("偏好样本没有回答空间")
        input_ids = [*prompt_ids, *response_ids][:max_length]
        labels = [-100] * len(prompt_ids) + response_ids
        return input_ids, labels[: len(input_ids)]


def train_tokenizer(
    files: Iterable[str | Path],
    output_path: str | Path,
    *,
    vocab_size: int = 8_000,
    min_frequency: int = 2,
) -> LLMTokenizer:
    """从 UTF-8 文本训练 Byte-level BPE；任意中文、英文或符号都不会变成 OOV。"""

    paths = [str(Path(path)) for path in files]
    if not paths:
        raise ValueError("至少需要一个 tokenizer 训练语料文件")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"找不到 tokenizer 语料: {missing}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    added = [AddedToken(token, special=True, normalized=False) for token in SPECIAL_TOKENS]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=added,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(paths, trainer)
    wrapped = LLMTokenizer(tokenizer)
    wrapped.save(output_path)
    return wrapped


def _validate_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages 不能为空")
    result: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] 必须是对象")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_TOKENS:
            raise ValueError(f"messages[{index}].role 不支持: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content 必须是非空字符串")
        if any(token in content for token in SPECIAL_TOKENS):
            raise ValueError("消息正文不能嵌入保留的角色控制 token")
        result.append({"role": role, "content": content.strip()})
    return result
