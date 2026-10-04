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


def test_multiteacher_routing_and_quantized_reload(tmp_path, bundle):
    from laptop_llm.engine import load_inference_bundle
    from laptop_llm.quantization import export_quantized

    model, tokenizer, checkpoint = bundle
    with torch.no_grad():
        model.layers[0].ffn.down_proj.weight.add_(
            torch.randn_like(model.layers[0].ffn.down_proj.weight) * 0.1
        )
    teacher = tmp_path / "teacher.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "tokenizer_json": tokenizer.to_str(),
        },
        teacher,
    )
    manifest = tmp_path / "teachers.json"
    manifest.write_text(
        json.dumps({"math:low": "teacher.pt", "math:max": "teacher.pt"}), encoding="utf-8"
    )
    data = tmp_path / "mopd.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "1+1=?"}],
                    "domain": "math",
                    "effort": effort,
                }
            )
            for effort in ["low", "max"]
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "lab",
            "mopd",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--teachers",
            str(manifest),
            "--output",
            str(tmp_path / "mopd"),
            "--steps",
            "1",
            "--max-new-tokens",
            "3",
            "--lr",
            "0.001",
        ]
    )
    trained = run_lab(args)
    metrics = json.loads((tmp_path / "mopd/metrics.jsonl").read_text(encoding="utf-8"))
    assert metrics["teacher_routes"] == ["math:low", "math:max"]
    assert metrics["objective_loss"] > 0
    record = json.loads((tmp_path / "mopd/rollouts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(record["input_ids"]) - 1 == len(record["action_mask"]) == len(record["behavior_logp"])
    assert sum(record["action_mask"]) > 0 and record["policy_version"] == 0
    packed_path = export_quantized(trained, tmp_path / "packed.pt")
    packed, reloaded_tokenizer, _ = load_inference_bundle(packed_path, device_name="cpu")
    assert packed.num_parameters() == model.num_parameters()
    assert reloaded_tokenizer.to_str() == tokenizer.to_str()
    assert torch.isfinite(packed(torch.tensor([[1, 2, 3]])).logits).all()
    with pytest.raises(ValueError, match="已存在"):
        export_quantized(trained, packed_path)
    args.checkpoint = str(packed_path)
    args.output = str(tmp_path / "refuse-packed-training")
    with pytest.raises(ValueError, match="推理产物"):
        run_lab(args)


def test_agent_tokens_and_observations_have_different_masks(tmp_path, bundle):
    from types import SimpleNamespace

    from laptop_llm.posttraining.agent import ArithmeticTask, collect_agent_rollout

    _, tokenizer, _ = bundle
    task = ArithmeticTask(3, 4)
    prompt = tokenizer.build_chat_prompt(task.prompt())
    call = tokenizer.encode('{"tool":"calculator","arguments":{"a":3,"b":4,"op":"add"}}') + [
        tokenizer.end_id
    ]
    feedback = [
        tokenizer.token_id("<|user|>"),
        *tokenizer.encode('工具反馈：{"result": 7}'),
        tokenizer.end_id,
        tokenizer.assistant_id,
    ]
    answer = tokenizer.encode('{"answer":7}') + [tokenizer.end_id]
    sequence = prompt + call + feedback + answer

    class ScriptedProtocolOracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = ModelConfig(
                vocab_size=tokenizer.vocab_size, max_seq_len=len(sequence) + 8
            )

        def forward(self, ids, **kwargs):
            offset = kwargs.get("past_key_values") or 0
            logits = torch.full((*ids.shape, tokenizer.vocab_size), -1000.0)
            for t in range(ids.size(1)):
                logits[:, t, sequence[min(offset + t + 1, len(sequence) - 1)]] = 0
            return SimpleNamespace(logits=logits, past_key_values=offset + ids.size(1))

    rollout, traces = collect_agent_rollout(ScriptedProtocolOracle(), tokenizer, [task], 256, 3)
    assert traces[0]["reward"] == 1 and traces[0]["reason"] == "final_answer"
    assert rollout.action_mask.sum() == len(call) + len(answer)
    assert not rollout.action_mask[
        0, len(prompt) + len(call) - 1 : len(prompt) + len(call) + len(feedback) - 1
    ].any()
    assert rollout.ids[0].tolist() == sequence
    assert rollout.terminal.sum() == 1
