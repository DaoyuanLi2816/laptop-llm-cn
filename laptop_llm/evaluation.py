"""可复现的生成评测：固定提示、贪心解码、保留全部失败样例与延迟。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from laptop_llm.data import read_jsonl
from laptop_llm.engine import load_inference_bundle
from laptop_llm.generation import GenerationConfig, TokenGenerator


def run_generation_evaluation(
    checkpoint: str | Path,
    suite_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    model, tokenizer, resolved_device = load_inference_bundle(
        checkpoint, device_name=device, dtype_name=dtype
    )
    generator = TokenGenerator(model, tokenizer, resolved_device)
    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0)
    rows = read_jsonl(suite_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    scored = 0
    passed = 0
    total_tokens = 0
    total_seconds = 0.0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            messages = row.get("messages")
            expected = row.get("expected_contains")
            if not isinstance(messages, list):
                raise ValueError(f"评测第 {index + 1} 行缺少 messages 数组")
            if expected is not None and not (
                isinstance(expected, list) and all(isinstance(item, str) for item in expected)
            ):
                raise ValueError("expected_contains 必须是字符串数组")

            prompt_ids = tokenizer.build_chat_prompt(
                messages,
                add_generation_prompt=True,
                max_length=model.config.max_seq_len - 1,
            )
            if resolved_device.type == "cuda":
                import torch

                torch.cuda.synchronize()
            started = time.perf_counter()
            text, completion_ids = generator.generate_text(prompt_ids, config)
            if resolved_device.type == "cuda":
                import torch

                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            is_pass = None if expected is None else all(fragment in text for fragment in expected)
            if is_pass is not None:
                scored += 1
                passed += int(is_pass)
            total_tokens += len(completion_ids)
            total_seconds += elapsed
            result = {
                "index": index,
                "messages": messages,
                "response": text,
                "expected_contains": expected,
                "passed": is_pass,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "latency_seconds": elapsed,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "examples": len(rows),
        "scored": scored,
        "passed": passed,
        "accuracy": passed / scored if scored else None,
        "completion_tokens": total_tokens,
        "seconds": total_seconds,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else None,
        "output": str(output.resolve()),
    }
    return summary
