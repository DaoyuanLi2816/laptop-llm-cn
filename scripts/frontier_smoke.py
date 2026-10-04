"""v0.3 前沿实验室：九个训练阶段 + 两种后训练 + FP4 + 精确投机解码。

只使用本地原创数据，不请求模型 API，不下载权重。所有数值是链路诊断，
不是能力榜单；零奖励也原样保留。拒绝覆盖非空目录，便于比较不同实验。
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from laptop_llm.architectures.cache import cache_bytes
from laptop_llm.cli import build_parser
from laptop_llm.config import ExperimentConfig
from laptop_llm.console import configure_console
from laptop_llm.engine import load_inference_bundle, run_stage, set_seed
from laptop_llm.posttraining.agent import ArithmeticTask
from laptop_llm.posttraining.trainer import run_lab
from laptop_llm.quantization import export_quantized, fake_quantize_fp4
from laptop_llm.speculative import speculative_generate
from laptop_llm.tokenizer import train_tokenizer


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/frontier-smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps 必须为正")
    torch.set_num_threads(2)
    set_seed(42)
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("output 非空；请使用新的实验目录")
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parent.parent
    started = time.perf_counter()

    # 所有模型共享逐字节相同的 tokenizer。协议进入训练语料，不把 eval 答案
    # 加进词表训练。词表学会字符串不等于模型学会工具调用。
    tasks = [ArithmeticTask(3, 4), ArithmeticTask(5, 2, "sub")]
    corpus = (repository / "data/demo/pretrain_train.txt").read_text(encoding="utf-8")
    corpus += "\n" + "\n".join(
        task.prompt()[0]["content"]
        + '\n{"tool":"calculator","arguments":'
        + json.dumps({"a": task.a, "b": task.b, "op": task.op})
        + '}\n{"answer":7}'
        for task in tasks
    )
    corpus_path = root / "tokenizer-corpus.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    tokenizer_path = root / "tokenizer.json"
    train_tokenizer([corpus_path], tokenizer_path, vocab_size=768, min_frequency=1)
    checkpoints, configurations = {}, {}
    for variant in ["hybrid", "indexed", "mhc"]:
        config = ExperimentConfig.from_yaml(repository / f"configs/frontier_{variant}.yaml")
        config.output_dir = str(root / variant)
        config.tokenizer_path = str(tokenizer_path)
        # 训练窗口缩到 192；同样的递推/索引算法，不把 CI 当性能压力测试。
        config.model.max_seq_len = 192
        for stage in config.stages.values():
            stage.max_steps = args.steps
        configurations[variant] = config.to_dict()
        checkpoint = run_stage("pretrain", config, device_name=args.device)
        checkpoint = run_stage("sft", config, init_from=checkpoint, device_name=args.device)
        checkpoints[variant] = checkpoint
        run_stage("dpo", config, init_from=checkpoint, device_name=args.device)
    write_json(root / "resolved-configs.json", configurations)

    # 域/推理预算是明确数据字段，不靠把两个教师 logits 平均来“融合能力”。
    teachers_path = root / "teachers.json"
    write_json(
        teachers_path,
        {"math:low": str(checkpoints["indexed"]), "math:max": str(checkpoints["hybrid"])},
    )
    rows = [
        {"domain": "math", "effort": effort, "prompt": [{"role": "user", "content": "3+4=?"}]}
        for effort in ["low", "max"]
    ]
    mopd_data = root / "mopd.jsonl"
    mopd_data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    agent_data = root / "agent.jsonl"
    agent_data.write_text(
        "".join(json.dumps({"task": vars(task)}) + "\n" for task in tasks), encoding="utf-8"
    )
    for algorithm, data in [("mopd", mopd_data), ("agent-grpo", agent_data)]:
        argv = [
            "lab",
            algorithm,
            "--checkpoint",
            str(checkpoints["hybrid"]),
            "--data",
            str(data),
            "--output",
            str(root / algorithm),
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--batch-size",
            "2" if algorithm == "mopd" else "1",
            "--max-new-tokens",
            "8",
            "--group-size",
            "2",
            "--epochs",
            "2",
        ]
        argv += (
            ["--teachers", str(teachers_path)]
            if algorithm == "mopd"
            else ["--policy-objective", "calibrated", "--freeze-router", "--gar"]
        )
        run_lab(build_parser().parse_args(argv))

    source = checkpoints["hybrid"]
    packed_path = export_quantized(source, root / "hybrid-fp4.pt")
    target, tokenizer, device = load_inference_bundle(
        source, device_name=args.device, dtype_name="float32"
    )
    packed, _, _ = load_inference_bundle(packed_path, device_name=args.device, dtype_name="float32")
    draft, _, _ = load_inference_bundle(
        checkpoints["indexed"], device_name=args.device, dtype_name="float32"
    )
    prompt = tokenizer.build_chat_prompt([{"role": "user", "content": "3+4=?"}])
    ids = torch.tensor([prompt], device=device)
    with torch.inference_mode():
        full = target(ids, use_cache=True)
        quantized = packed(ids, use_cache=True)
        rmse = (full.logits - quantized.logits).float().square().mean().sqrt().item()
        cache_storage = cache_bytes(full.past_key_values)
        ordinary, prefix = [], list(prompt)
        for _ in range(8):
            token = int(target(torch.tensor([prefix], device=device)).logits[0, -1].argmax())
            prefix.append(token)
            ordinary.append(token)
        proposed, stats = speculative_generate(target, draft, prompt, 8, 3, temperature=0)
        if proposed != ordinary:
            raise AssertionError("精确投机解码与目标模型贪心结果不一致")
    # 真正跑一次 STE backward；这是量化算子的梯度实验，不是完整模型 QAT。
    weight = torch.randn(8, 8, requires_grad=True, device=device)
    fake_quantize_fp4(weight).square().mean().backward()
    if not torch.isfinite(weight.grad).all():
        raise AssertionError("QAT 直通梯度非有限")
    agent_metrics = [
        json.loads(line)
        for line in (root / "agent-grpo/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = {
        "scope": "pipeline_and_numerical_checks_only_not_model_capability",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_stages": 11,
        "steps_per_stage": args.steps,
        "agent_reward_means": [row["reward_mean"] for row in agent_metrics],
        "agent_zero_variance_groups": [row["zero_variance_groups"] for row in agent_metrics],
        "fp4_logit_rmse": rmse,
        "cache_bytes_at_prompt": cache_storage,
        # 权重张量字节数，不拿带 optimizer 的训练文件和推理文件比较压缩倍数。
        "float_model_storage_bytes": cache_bytes(list(target.state_dict().values())),
        "packed_model_storage_bytes": cache_bytes(list(packed.state_dict().values())),
        "speculative_greedy_equal": True,
        "speculation": vars(stats),
        "seconds": time.perf_counter() - started,
        "tokenizer_sha256": sha256(tokenizer_path),
        "checkpoints_sha256": {key: sha256(value) for key, value in checkpoints.items()},
        "data_sha256": {
            name: sha256(path)
            for name, path in {
                "corpus": corpus_path,
                "mopd": mopd_data,
                "agent": agent_data,
            }.items()
        },
    }
    write_json(root / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
