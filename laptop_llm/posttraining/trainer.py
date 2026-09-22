"""笔记本后训练实验室：小而完整的同步采样—优化闭环。

这里没有 Trainer 魔法。顺序是 load → freeze reference → rollout → reward →
advantage → 多轮更新 → 记录 → 保存。先在 CPU 跑 2 步，再逐项查数值。
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from laptop_llm.architectures.lora import inject_lora, merge_lora
from laptop_llm.data import pad_lm_batch
from laptop_llm.engine import append_metric, load_inference_bundle, read_checkpoint, set_seed
from laptop_llm.posttraining.objectives import (
    clipped_policy_loss,
    clipped_value_loss,
    distillation_loss,
    generalized_advantage,
    group_advantages,
    masked_mean,
    preference_reward_loss,
    reference_kl,
)
from laptop_llm.posttraining.rewards import ScalarHead, terminal_scores, verify_arithmetic
from laptop_llm.posttraining.rollout import action_log_probs, collect_rollout


def add_lab_parser(subparsers):
    parser = subparsers.add_parser(
        "lab", help="reward / PPO / GRPO-RLVR / on-policy distillation / LoRA"
    )
    parser.add_argument("algorithm", choices=["reward", "ppo", "grpo", "opd", "lora"])
    parser.add_argument("--checkpoint", required=True, help="可信的 SFT/预训练 checkpoint")
    parser.add_argument(
        "--data", required=True, help="JSONL：偏好对、SFT messages 或 prompt+answer"
    )
    parser.add_argument("--output", required=True, help="新的实验目录，不覆盖已有实验")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2, help="同一批 rollout 的更新轮数")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher", help="OPD 必填：同 tokenizer 的教师 checkpoint")
    parser.add_argument("--reward-model", help="PPO 必填：lab reward 保存的 checkpoint")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    return parser


def padded_examples(rows, tokenizer, device, max_length, response_key=None):
    examples = []
    for row in rows:
        messages = (
            row["messages"]
            if response_key is None
            else [*row["prompt"], {"role": "assistant", "content": row[response_key]}]
        )
        ids, labels = tokenizer.build_sft_example(messages, max_length + 1)
        if len(ids) > max_length:
            raise ValueError("后训练样本超长；明确清洗或缩短数据，不隐式截掉答案")
        if not any(label != -100 for label in labels[1:]):
            raise ValueError("样本没有 assistant 监督 token")
        examples.append({"input_ids": ids, "labels": labels})
    return {
        key: value.to(device) for key, value in pad_lm_batch(examples, tokenizer.pad_id).items()
    }


def frozen_bundle(path, tokenizer, device):
    model, other_tokenizer, _ = load_inference_bundle(
        path, device_name=str(device), dtype_name="float32"
    )
    if other_tokenizer.to_str() != tokenizer.to_str():
        raise ValueError("师生/奖励模型 tokenizer 必须逐字节一致；词表大小相同并不足够")
    return model.eval().requires_grad_(False)


def run_lab(args: argparse.Namespace):
    if min(args.steps, args.batch_size, args.epochs, args.max_new_tokens) < 1 or args.lr <= 0:
        raise ValueError("steps/batch/epochs/生成长度/学习率必须为正")
    if args.kl_coef < 0:
        raise ValueError("KL 系数不能为负")
    if args.algorithm == "grpo" and args.group_size < 2:
        raise ValueError("GRPO group_size 至少为 2")
    if args.algorithm == "opd" and not args.teacher:
        raise ValueError("OPD 需要 --teacher；教师更强不是算法自动保证的")
    if args.algorithm == "ppo" and not args.reward_model:
        raise ValueError("PPO-RLHF 需要先训练 --reward-model；规则奖励请运行 grpo")
    target = Path(args.output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("output 非空；使用新的实验目录以保存旧结果")
    data_path = Path(args.data)
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("数据为空")
    set_seed(args.seed)
    model, tokenizer, device = load_inference_bundle(
        args.checkpoint, device_name=args.device, dtype_name="float32"
    )
    model.eval()  # 策略概率需与 rollout 一致；eval 不会阻止 autograd。
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    teacher = frozen_bundle(args.teacher, tokenizer, device) if args.teacher else None
    reward_model, reward_head, critic, value_head = None, None, None, None
    parameters = list(model.parameters())
    if args.algorithm == "reward":
        reward_head = ScalarHead(model.config.dim).to(device)
        parameters += list(reward_head.parameters())
    if args.algorithm == "ppo":
        reward_model = frozen_bundle(args.reward_model, tokenizer, device)
        reward_head = ScalarHead(reward_model.config.dim).to(device)
        reward_head.load_state_dict(read_checkpoint(args.reward_model)["reward_head"])
        reward_head.requires_grad_(False).eval()
        critic = copy.deepcopy(model)
        value_head = ScalarHead(model.config.dim).to(device)
        parameters += [*critic.parameters(), *value_head.parameters()]
    if args.algorithm == "lora":
        inject_lora(model, rank=args.rank, alpha=2 * args.rank)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    target.mkdir(parents=True, exist_ok=True)
    manifest = vars(args) | {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "teacher_sha256": file_sha256(args.teacher) if args.teacher else None,
        "reward_model_sha256": file_sha256(args.reward_model) if args.reward_model else None,
        "torch_version": torch.__version__,
        "device_resolved": str(device),
        "scope": "single-device educational experiment; not a capability benchmark",
    }
    (target / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for step in range(args.steps):
        batch_rows = [
            rows[(step * args.batch_size + index) % len(rows)] for index in range(args.batch_size)
        ]
        metrics = {"step": step + 1, "algorithm": args.algorithm}
        if args.algorithm in {"reward", "lora"}:
            optimizer.zero_grad(set_to_none=True)
            if args.algorithm == "lora":
                batch = padded_examples(batch_rows, tokenizer, device, model.config.max_seq_len)
                output = model(**batch)
                loss = output.loss + output.auxiliary_loss
            else:
                scores = []
                for key in ("chosen", "rejected"):
                    batch = padded_examples(
                        batch_rows, tokenizer, device, model.config.max_seq_len, key
                    )
                    scores.append(
                        terminal_scores(
                            model, reward_head, batch["input_ids"], batch["attention_mask"]
                        )
                    )
                loss = preference_reward_loss(*scores)
                metrics["pair_accuracy"] = float((scores[0] > scores[1]).float().mean())
            optimize(loss, parameters, optimizer)
        else:
            groups = args.group_size if args.algorithm == "grpo" else 1
            prompts = [row["prompt"] for row in batch_rows for _ in range(groups)]
            rollout = collect_rollout(model, tokenizer, prompts, args.max_new_tokens)
            mask = rollout.action_mask
            with torch.no_grad():
                ref_logp, _ = action_log_probs(reference, rollout.ids, rollout.attention_mask)
                if args.algorithm == "grpo":
                    expected = [row["answer"] for row in batch_rows for _ in range(groups)]
                    if any(type(answer) is not int for answer in expected):
                        raise ValueError("RLVR answer 必须是整数，不接受字符串或布尔值")
                    rewards = torch.tensor(
                        [
                            verify_arithmetic(text, answer)
                            for text, answer in zip(rollout.texts, expected, strict=True)
                        ],
                        device=device,
                    )
                    advantages = group_advantages(rewards, groups)[:, None].expand_as(mask)
                    metrics["zero_variance_groups"] = float(
                        (rewards.view(-1, groups).std(-1, unbiased=False) == 0).float().mean()
                    )
                elif args.algorithm == "ppo":
                    rewards = terminal_scores(
                        reward_model, reward_head, rollout.ids, rollout.attention_mask
                    )
                    full_values = value_head(
                        critic(rollout.ids, attention_mask=rollout.attention_mask).hidden_states
                    )
                    old_values = full_values[:, :-1]
                    shaped = -args.kl_coef * (rollout.old_logp - ref_logp) * mask
                    last = rollout.attention_mask.sum(-1).long() - 2
                    shaped[torch.arange(len(prompts), device=device), last] += rewards
                    advantages, returns = generalized_advantage(
                        shaped, old_values, full_values[:, 1:], mask, rollout.terminal
                    )
                if args.algorithm != "opd":
                    metrics["reward_mean"] = float(rewards.mean())
                else:
                    teacher_logits = (
                        teacher(rollout.ids, attention_mask=rollout.attention_mask)
                        .logits[:, :-1]
                        .detach()
                    )
            metrics["response_tokens"] = int(mask.sum())
            metrics["eos_fraction"] = float(rollout.terminal.any(-1).float().mean())
            for _ in range(args.epochs):
                optimizer.zero_grad(set_to_none=True)
                logp, output = action_log_probs(model, rollout.ids, rollout.attention_mask)
                if args.algorithm == "opd":
                    loss = distillation_loss(output.logits[:, :-1], teacher_logits, mask)
                else:
                    loss = clipped_policy_loss(logp, rollout.old_logp, advantages, mask)
                    kl = reference_kl(logp, ref_logp, mask)
                    metrics["reference_kl_estimate"] = float(kl.detach())
                    metrics["old_policy_log_ratio_abs"] = float(
                        masked_mean((logp.detach() - rollout.old_logp).abs(), mask)
                    )
                    if args.algorithm == "grpo":
                        loss = loss + args.kl_coef * kl
                    else:
                        values = value_head(
                            critic(rollout.ids, attention_mask=rollout.attention_mask).hidden_states
                        )[:, :-1]
                        value_loss = clipped_value_loss(values, old_values, returns, mask)
                        loss = loss + 0.5 * value_loss
                        metrics["value_loss"] = float(value_loss.detach())
                metrics["objective_loss"] = float(loss.detach())
                metrics["router_auxiliary_loss"] = float(output.auxiliary_loss.detach())
                loss = loss + output.auxiliary_loss
                optimize(loss, parameters, optimizer)
            # 保存原始输出和奖励，零奖励也必须留下，不能只展示成功样例。
            with (target / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                for index, text in enumerate(rollout.texts):
                    handle.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "prompt": prompts[index],
                                "text": text,
                                "reward": None
                                if args.algorithm == "opd"
                                else float(rewards[index]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        metrics["loss"] = float(loss.detach())
        append_metric(target / "metrics.jsonl", metrics)
        print(json.dumps(metrics, ensure_ascii=True))
    if args.algorithm == "lora":
        merge_lora(model)
    payload = {
        "format_version": 2,
        "stage": args.algorithm,
        "step": args.steps,
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "tokenizer_json": tokenizer.to_str(),
        "optimizer": optimizer.state_dict(),
        "reference": reference.state_dict(),
        "experiment": manifest,
        "torch_rng": torch.get_rng_state(),
    }
    if reward_head is not None and args.algorithm == "reward":
        payload["reward_head"] = reward_head.state_dict()
    if critic is not None:
        payload["critic"] = critic.state_dict()
        payload["value_head"] = value_head.state_dict()
    temporary = target / "final.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(target / "final.pt")
    return target / "final.pt"


def optimize(loss, parameters, optimizer):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("loss 非有限，拒绝写入损坏权重")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()


def file_sha256(path):
    """分块校验权重来源，不把整个大 checkpoint 额外读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
