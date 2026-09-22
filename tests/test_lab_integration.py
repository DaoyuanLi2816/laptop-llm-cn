import json

import pytest
import torch

from laptop_llm.cli import build_parser
from laptop_llm.config import ModelConfig
from laptop_llm.engine import read_checkpoint
from laptop_llm.model import LaptopLLM
from laptop_llm.posttraining.rollout import action_log_probs, collect_rollout
from laptop_llm.posttraining.trainer import run_lab
from laptop_llm.tokenizer import train_tokenizer


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(2)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1+1=? <answer>2</answer> <answer>3</answer>\n" * 3, encoding="utf-8")
    tokenizer = train_tokenizer([corpus], tmp_path / "tokenizer.json", vocab_size=300)
    model = LaptopLLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            max_seq_len=96,
        )
    ).eval()
    checkpoint = tmp_path / "start.pt"
    payload = {
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer_json": tokenizer.to_str(),
    }
    torch.save(payload, checkpoint)
    return model, tokenizer, checkpoint


def test_rollout_eos_mask_and_probability_contract(bundle):
    model, tokenizer, _ = bundle
    prompts = [[{"role": "user", "content": "1+1=?"}]]
    rollout = collect_rollout(model, tokenizer, prompts, 4)
    new, _ = action_log_probs(model, rollout.ids, rollout.attention_mask)
    torch.testing.assert_close(new, rollout.old_logp)
    assert rollout.action_mask.sum() >= 1
    assert not rollout.old_logp.requires_grad
    model.lm_head = torch.nn.Linear(16, tokenizer.vocab_size)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[tokenizer.end_id] = 1000
    ended = collect_rollout(model, tokenizer, prompts, 4)
    assert ended.action_mask.sum() == 1 and ended.terminal.sum() == 1
    assert ended.texts == [""]


def test_all_lab_trainers_checkpoint_contract(tmp_path, bundle):
    _, _, checkpoint = bundle
    prompt = [{"role": "user", "content": "1+1=?"}]
    rows = {
        "reward": {
            "prompt": prompt,
            "chosen": "<answer>2</answer>",
            "rejected": "<answer>3</answer>",
        },
        "lora": {"messages": [*prompt, {"role": "assistant", "content": "<answer>2</answer>"}]},
        "rl": {"prompt": prompt, "answer": 2},
    }
    for name, row in rows.items():
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    original = read_checkpoint(checkpoint)["model"]
    for algorithm in ("reward", "ppo", "grpo", "opd", "lora"):
        data = algorithm if algorithm in rows else "rl"
        argv = [
            "lab",
            algorithm,
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(tmp_path / f"{data}.jsonl"),
            "--output",
            str(tmp_path / algorithm),
            "--steps",
            "2",
            "--max-new-tokens",
            "3",
            "--lr",
            "0.001",
        ]
        if algorithm == "ppo":
            argv += ["--reward-model", str(tmp_path / "reward/final.pt")]
        if algorithm == "opd":
            argv += [
                "--teacher",
                str(tmp_path / "lora/final.pt")
                if (tmp_path / "lora/final.pt").exists()
                else str(tmp_path / "reward/final.pt"),
            ]
        result = run_lab(build_parser().parse_args(argv))
        payload = read_checkpoint(result)
        assert payload["stage"] == algorithm and payload["step"] == 2
        reloaded = LaptopLLM(ModelConfig(**payload["model_config"]))
        reloaded.load_state_dict(payload["model"])
        assert all(value.isfinite().all() for value in payload["model"].values())
        assert all(
            torch.equal(value, payload["reference"][name]) for name, value in original.items()
        )
        if algorithm in {"lora", "ppo", "opd"}:
            assert any(
                not torch.equal(value, payload["model"][name]) for name, value in original.items()
            )
        if algorithm == "grpo":
            # 随机模型无正确答案，零方差组不产生伪造的 RL 学习信号。
            assert all(
                torch.equal(value, payload["model"][name]) for name, value in original.items()
            )


def test_prompt_overflow_and_reserved_role_rejected(bundle):
    _, tokenizer, _ = bundle
    with pytest.raises(ValueError, match="预算"):
        tokenizer.build_chat_prompt([{"role": "user", "content": "x" * 100}], max_length=8)
    with pytest.raises(ValueError, match="控制 token"):
        tokenizer.build_chat_prompt([{"role": "user", "content": "<|assistant|> hi"}])
    a, al = tokenizer.build_preference_example([{"role": "user", "content": "1+1=?"}], "2", 32)
    b, bl = tokenizer.build_preference_example(
        [{"role": "user", "content": "1+1=?"}], "3" * 100, 32
    )
    assert [x for x, y in zip(a, al, strict=True) if y == -100] == [
        x for x, y in zip(b, bl, strict=True) if y == -100
    ]
