"""KV Cache 增量解码、采样策略和多轮聊天状态。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from laptop_llm.model import LaptopLLM
from laptop_llm.tokenizer import LLMTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    min_new_tokens: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.temperature < 0:
            raise ValueError("temperature 不能为负")
        if self.top_k < 0:
            raise ValueError("top_k 不能为负")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 内")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty 应不小于 1")


class TokenGenerator:
    """单请求生成器：prefill 一次，之后每步只计算一个 token。"""

    def __init__(self, model: LaptopLLM, tokenizer: LLMTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate_tokens(
        self,
        prompt_ids: Sequence[int],
        config: GenerationConfig,
        *,
        stop_ids: set[int] | None = None,
    ) -> Iterator[int]:
        if not prompt_ids:
            raise ValueError("prompt_ids 不能为空")
        if len(prompt_ids) >= self.model.config.max_seq_len:
            raise ValueError("prompt 已占满上下文，模型没有空间生成")
        available = self.model.config.max_seq_len - len(prompt_ids)
        max_new_tokens = min(config.max_new_tokens, available)
        stops = stop_ids or {self.tokenizer.end_id, self.tokenizer.eos_id}
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        output = self.model(input_ids, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        history = list(prompt_ids)

        for generated_count in range(max_new_tokens):
            next_id = sample_next_token(logits, history, config)
            if next_id in stops and generated_count >= config.min_new_tokens:
                break
            history.append(next_id)
            yield next_id
            if generated_count + 1 >= max_new_tokens:
                break
            token = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            output = self.model(token, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[:, -1, :]

    def generate_text(
        self, prompt_ids: Sequence[int], config: GenerationConfig
    ) -> tuple[str, list[int]]:
        ids = list(self.generate_tokens(prompt_ids, config))
        return self.tokenizer.decode(ids), ids

    def stream_text(
        self, prompt_ids: Sequence[int], config: GenerationConfig
    ) -> Iterator[str]:
        """逐 token 解码为增量字符串；累计解码可正确处理 byte-level 中文 token。"""

        generated: list[int] = []
        emitted = ""
        for token_id in self.generate_tokens(prompt_ids, config):
            generated.append(token_id)
            current = self.tokenizer.decode(generated)
            # 一个中文字符可能横跨多个 byte token。末尾出现替换字符时先不发送，
            # 等后续 byte 补齐再发送，避免客户端看到“�”或重复整段文本。
            stable = current.rstrip("�")
            delta = stable[len(emitted) :] if stable.startswith(emitted) else ""
            if delta:
                emitted = stable
                yield delta
        final = self.tokenizer.decode(generated)
        if final.startswith(emitted) and final != emitted:
            yield final[len(emitted) :]


def sample_next_token(
    logits: torch.Tensor, history: Sequence[int], config: GenerationConfig
) -> int:
    scores = logits[0].float().clone()
    if config.repetition_penalty != 1.0:
        used = torch.tensor(sorted(set(history)), device=scores.device, dtype=torch.long)
        selected = scores[used]
        scores[used] = torch.where(
            selected < 0,
            selected * config.repetition_penalty,
            selected / config.repetition_penalty,
        )

    if config.temperature == 0:
        return int(scores.argmax())
    scores /= max(config.temperature, 1e-5)

    if config.top_k > 0 and config.top_k < scores.numel():
        threshold = torch.topk(scores, config.top_k).values[-1]
        scores[scores < threshold] = -torch.inf

    if config.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf

    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1))


class ChatEngine:
    """管理多轮 history；模型只负责 token，角色模板由 tokenizer 统一处理。"""

    def __init__(
        self,
        generator: TokenGenerator,
        *,
        system_prompt: str = "你是一个诚实、友好、简洁的中文助手。",
    ):
        self.generator = generator
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def prompt_ids(
        self, user_message: str, generation_config: GenerationConfig
    ) -> list[int]:
        candidate = [*self.messages, {"role": "user", "content": user_message}]
        budget = self.generator.model.config.max_seq_len - generation_config.max_new_tokens
        return self.generator.tokenizer.build_chat_prompt(
            candidate, add_generation_prompt=True, max_length=max(8, budget)
        )

    def reply(self, user_message: str, config: GenerationConfig) -> str:
        prompt = self.prompt_ids(user_message, config)
        text, _ = self.generator.generate_text(prompt, config)
        self.messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": text},
            ]
        )
        return text
