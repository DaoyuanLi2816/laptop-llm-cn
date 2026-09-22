"""无需联网/云算力，生成原创算术数据并运行全部课程阶段。

生成文件是实验产物，默认放入 gitignore 的 artifacts。输出非空则拒绝覆盖。
--sft-steps 加大只是算术实验；不能据此推断通用聊天或推理能力。
"""

import argparse
import json
from pathlib import Path

import torch

from laptop_llm.cli import build_parser
from laptop_llm.config import ExperimentConfig, ModelConfig, StageConfig
from laptop_llm.engine import run_stage
from laptop_llm.posttraining.trainer import run_lab
from laptop_llm.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/lab-smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sft-steps", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("请使用全新的 output 目录")
    root.mkdir(parents=True, exist_ok=True)
    training, evaluation, preferences, prompts, corpus = [], [], [], [], []
    for a in range(8):
        for b in range(8):
            prompt = [{"role": "user", "content": f"{a}+{b}=?"}]
            correct = f"<think>{a}+{b}={a + b}.</think><answer>{a + b}</answer>"
            wrong = f"<think>{a}+{b}={a + b + 1}.</think><answer>{a + b + 1}</answer>"
            example = {"messages": [*prompt, {"role": "assistant", "content": correct}]}
            # 以问题切分，绝不把同一个问题的两个回答分到 train/eval 两侧。
            if (a * 8 + b) % 5 == 0:
                evaluation.append(example)
            else:
                training.append(example)
                preferences.append({"prompt": prompt, "chosen": correct, "rejected": wrong})
                prompts.append({"prompt": prompt, "answer": a + b})
                corpus.append(prompt[0]["content"] + correct)
    files = {
        "sft_train": training,
        "sft_eval": evaluation,
        "dpo_train": preferences[:-8],
        "dpo_eval": preferences[-8:],
        "rl_train": prompts[:-8],
    }
    # 这是同分布合成诊断集；后训练之间复用训练题，不是独立通用能力榜单。
    for name, rows in files.items():
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    (root / "pretrain_train.txt").write_text("\n".join(corpus * 3), encoding="utf-8")
    (root / "pretrain_eval.txt").write_text(
        "\n".join(
            row["messages"][0]["content"] + row["messages"][1]["content"] for row in evaluation
        ),
        encoding="utf-8",
    )
    tokenizer_path = root / "tokenizer.json"
    train_tokenizer([root / "pretrain_train.txt"], tokenizer_path, vocab_size=384, min_frequency=1)
    stages = {
        name: StageConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=args.sft_steps if name == "sft" else 2,
            learning_rate=1e-3,
            warmup_steps=1,
            log_interval=1,
            eval_interval=100,
            eval_batches=2,
            save_interval=100,
            dtype="float32",
        )
        for name in ("pretrain", "sft", "dpo")
    }
    config = ExperimentConfig(
        name="advanced-smoke",
        output_dir=str(root),
        tokenizer_path=str(tokenizer_path),
        model=ModelConfig(
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            attention_pattern="sliding",
            sliding_window=16,
            attention_sinks=2,
            dense_every=2,
            qk_norm=True,
            num_experts=4,
            experts_per_token=2,
            shared_experts=1,
        ),
        stages=stages,
        data={
            **{key: str(root / f"{key}.jsonl") for key in files},
            "pretrain_train": str(root / "pretrain_train.txt"),
            "pretrain_eval": str(root / "pretrain_eval.txt"),
        },
    )
    pretrain = run_stage("pretrain", config, device_name=args.device)
    sft = run_stage("sft", config, init_from=pretrain, device_name=args.device)
    run_stage("dpo", config, init_from=sft, device_name=args.device)
    for algorithm in ("reward", "ppo", "grpo", "opd", "lora"):
        source = (
            "dpo_train"
            if algorithm == "reward"
            else "sft_train"
            if algorithm == "lora"
            else "rl_train"
        )
        argv = [
            "lab",
            algorithm,
            "--checkpoint",
            str(sft),
            "--data",
            str(root / f"{source}.jsonl"),
            "--output",
            str(root / algorithm),
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--max-new-tokens",
            "8",
            "--device",
            args.device,
        ]
        if algorithm == "ppo":
            argv += ["--reward-model", str(root / "reward/final.pt")]
        if algorithm == "opd":
            # student=pretrain，teacher=SFT；不宣称几步 SFT 后教师已具备有用能力。
            argv[argv.index("--checkpoint") + 1] = str(pretrain)
            argv += ["--teacher", str(sft)]
        run_lab(build_parser().parse_args(argv))
    print(f"All 8 stages completed: {root}")


if __name__ == "__main__":
    main()
