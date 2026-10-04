"""18 个与章节一一对应的 CPU 小实验。

用法：python scripts/lesson_examples.py 06 或 python scripts/lesson_examples.py all。
这里只验证机制，不下载权重、不请求 API、不把随机小模型包装成聊天能力。
每个函数独立重置 seed；assert 是本章最重要的“预期输出”。
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from laptop_llm.architectures.cache import cache_bytes  # noqa: E402
from laptop_llm.architectures.delta import delta_update  # noqa: E402
from laptop_llm.architectures.lora import LoRALinear  # noqa: E402
from laptop_llm.architectures.residual import AttentionResidual, sinkhorn  # noqa: E402
from laptop_llm.config import ModelConfig  # noqa: E402
from laptop_llm.console import configure_console  # noqa: E402
from laptop_llm.model import LaptopLLM  # noqa: E402
from laptop_llm.multimodal import VisionPrefixModel  # noqa: E402
from laptop_llm.optim import HybridMuon  # noqa: E402
from laptop_llm.posttraining.agent import ArithmeticTask, calculate, parse_action  # noqa: E402
from laptop_llm.posttraining.frontier import (  # noqa: E402
    calibrated_policy_loss,
    redistribute_advantages,
)
from laptop_llm.posttraining.objectives import (  # noqa: E402
    clipped_policy_loss,
    distillation_loss,
    generalized_advantage,
    group_advantages,
    preference_reward_loss,
)
from laptop_llm.posttraining.rewards import verify_arithmetic  # noqa: E402
from laptop_llm.quantization import pack_fp4, unpack_fp4  # noqa: E402
from laptop_llm.speculative import correction_distribution  # noqa: E402


def small_model(**options):
    return LaptopLLM(
        ModelConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            **options,
        )
    ).eval()


def lesson_01():
    """输入和目标错开一格；-100 是 loss 忽略符，不是词表中的 token。"""
    ids = torch.tensor([[1, 11, 12, 2]])
    inputs, targets = ids[:, :-1], ids[:, 1:]
    assert inputs.tolist() == [[1, 11, 12]]
    assert targets.tolist() == [[11, 12, 2]]
    labels = torch.tensor([[-100, -100, 12, 2]])
    mask = labels[:, 1:] != -100
    assert mask.tolist() == [[False, True, True]]
    return {"input": inputs.tolist(), "target": targets.tolist(), "loss_mask": mask.tolist()}


def lesson_02():
    """保留每个位置的词表轴，再由 CE 把有监督的位置聚合成标量。"""
    model = small_model()
    ids = torch.randint(0, 64, (2, 7))
    output = model(ids, labels=ids)
    assert output.logits.shape == (2, 7, 64)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert model.layers[0].attn.q_proj.weight.grad is not None
    return {"logits_shape": list(output.logits.shape), "loss_is_finite": True}


def lesson_03():
    """cache 复用不能改变同一前缀对应的输出。"""
    model = small_model()
    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        full = model(ids).logits
        prefix = model(ids[:, :4], use_cache=True)
        suffix = model(ids[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)
    joined = torch.cat((prefix.logits, suffix.logits), 1)
    torch.testing.assert_close(full, joined, atol=2e-6, rtol=2e-5)
    return {"cache_equal": True, "cache_bytes": cache_bytes(suffix.past_key_values)}


def lesson_04():
    """低秩增量合并后不再需要 adapter，但函数值应保持相同。"""
    layer = LoRALinear(nn.Linear(8, 6), rank=2, alpha=4)
    x = torch.randn(3, 8)
    torch.testing.assert_close(layer(x), layer.base(x))  # B 初始为 0。
    with torch.no_grad():
        layer.b.normal_(std=0.1)
    torch.testing.assert_close(layer(x), layer.merged()(x))
    assert not layer.base.weight.requires_grad
    moe = small_model(num_experts=4)
    ids = torch.randint(0, 64, (1, 6))
    output = moe(ids, labels=ids)
    (output.loss + output.auxiliary_loss).backward()
    assert moe.layers[0].ffn.router.weight.grad is not None
    return {"lora_merge_equal": True, "router_has_gradient": True}


def lesson_05():
    """成对偏好只给相对次序，不直接提供下一 token 的正确标签。"""
    chosen = torch.tensor([1.0, 2.0], requires_grad=True)
    rejected = torch.tensor([0.0, -1.0], requires_grad=True)
    loss = preference_reward_loss(chosen, rejected)
    loss.backward()
    assert bool((chosen.grad < 0).all())
    assert bool((rejected.grad > 0).all())
    # DPO 的 preference logit 还要扣掉 reference 的同一对 log-prob 差。
    policy_gap, reference_gap, beta = torch.tensor(2.0), torch.tensor(2.0), 0.1
    dpo = -F.logsigmoid(beta * (policy_gap - reference_gap))
    torch.testing.assert_close(dpo, torch.tensor(2.0).log())
    return {"chosen_should_increase": True, "initial_dpo_loss": round(dpo.item(), 6)}


def lesson_06():
    """两动作 GAE 手算：EOS 与长度截断的 bootstrap 不相同。"""
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.3]])
    next_values = torch.tensor([[0.3, 0.5]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    eos = torch.tensor([[False, True]])
    advantage, returns = generalized_advantage(rewards, values, next_values, mask, eos, lam=1)
    torch.testing.assert_close(advantage, torch.tensor([[0.8, 0.7]]))
    torch.testing.assert_close(returns, torch.ones(1, 2))
    _, truncated = generalized_advantage(rewards, values, next_values, mask, ~mask, lam=1)
    torch.testing.assert_close(truncated, torch.full((1, 2), 1.5))
    old = torch.zeros(1, 2)
    new = torch.zeros(1, 2, requires_grad=True)
    clipped_policy_loss(new, old, advantage, mask).backward()
    assert bool((new.grad < 0).all())
    zero = group_advantages(torch.zeros(4), 4)
    assert torch.equal(zero, torch.zeros(4))
    return {
        "eos_returns": returns.tolist(),
        "truncated_returns": truncated.tolist(),
        "zero_group": zero.tolist(),
    }


def lesson_07():
    """同一批 token 上做 reverse KL：student 有梯度，teacher 必须冻结。"""
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 5, requires_grad=True)
    mask = torch.tensor([[False, True, True], [False, True, False]])
    loss = distillation_loss(student, teacher, mask)
    loss.backward()
    assert teacher.grad is None
    assert torch.equal(student.grad[~mask], torch.zeros_like(student.grad[~mask]))
    assert student.grad[mask].abs().sum() > 0
    return {"teacher_gradient": None, "prompt_gradient_zero": True, "student_gradient": True}


def lesson_08():
    """activation checkpoint 少存激活，不应改变本次梯度。"""
    original = small_model().train()
    recomputed = copy.deepcopy(original)
    recomputed.enable_gradient_checkpointing()
    ids = torch.randint(0, 64, (1, 8))
    original(ids, labels=ids).loss.backward()
    recomputed(ids, labels=ids).loss.backward()
    for first, second in zip(original.parameters(), recomputed.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5)
    return {"recomputed_gradient_equal": True, "distributed_training": False}


def lesson_09():
    """缓存必须允许分叉；一个分支追加 token 不能覆盖另一个分支。"""
    model = small_model()
    with torch.no_grad():
        prefix = model(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
        saved = prefix[0][0].clone()
        first = model(torch.tensor([[4]]), past_key_values=prefix, use_cache=True)
        second = model(torch.tensor([[5]]), past_key_values=prefix, use_cache=True)
    torch.testing.assert_close(prefix[0][0], saved)
    assert first.past_key_values[0][0].size(2) == second.past_key_values[0][0].size(2) == 4
    return {"prefix_unchanged": True, "branch_cache_bytes": cache_bytes(first.past_key_values)}


def lesson_10():
    """先固定样例再比较：平均分相同也可能有不同的失败集合。"""
    baseline = torch.tensor([1.0, 0.0, 1.0, 0.0])
    candidate = torch.tensor([1.0, 1.0, 0.0, 0.0])
    paired = candidate - baseline
    assert paired.mean() == 0 and paired.abs().sum() == 2
    data = json.dumps(["case-a", "case-b", "case-c", "case-d"], ensure_ascii=False)
    fingerprint = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return {"paired_delta": paired.tolist(), "mean_improvement": 0, "data_sha256": fingerprint}


def lesson_11():
    """把逐项递推对照显式矩阵 oracle：必须先遗忘，再纠错。"""
    state = torch.randn(1, 2, 4, 4)
    q, k, v = [torch.randn(1, 2, 4) for _ in range(3)]
    alpha, beta = torch.rand(1, 2, 4), torch.rand(1, 2)
    updated, output = delta_update(state, q, k, v, alpha, beta)
    projection = torch.eye(4) - beta[..., None, None] * k[..., :, None] * k[..., None, :]
    oracle = (
        projection @ torch.diag_embed(alpha) @ state
        + beta[..., None, None] * k[..., :, None] * v[..., None, :]
    )
    torch.testing.assert_close(updated, oracle)
    torch.testing.assert_close(output, (oracle.transpose(-1, -2) @ q[..., None]).squeeze(-1))
    model = small_model(attention_type="kda")
    with torch.no_grad():
        short = model(torch.ones(1, 4, dtype=torch.long), use_cache=True).past_key_values
        long = model(torch.ones(1, 16, dtype=torch.long), use_cache=True).past_key_values
    assert cache_bytes(short) == cache_bytes(long)
    return {"matrix_oracle_equal": True, "pure_kda_fixed_cache_bytes": cache_bytes(long)}


def lesson_12():
    """AttnRes 在来源轴上混合；Sinkhorn 在残差流之间施加约束。"""
    sources = torch.randn(3, 1, 4, 8)
    mixed = AttentionResidual(8)(sources)
    torch.testing.assert_close(mixed, sources.mean(0))  # 零 query → 均匀来源权重。
    matrix = sinkhorn(torch.randn(2, 4, 4), iterations=50)
    torch.testing.assert_close(matrix.sum(-1), torch.ones(2, 4))
    torch.testing.assert_close(matrix.sum(-2), torch.ones(2, 4))
    assert (matrix >= 0).all()
    return {
        "zero_query_is_mean": True,
        "row_column_sums": 1,
        "whole_network_nonexpansive": "not_proved",
    }


def lesson_13():
    """索引预算覆盖全部块时，主注意力必须退化为 dense oracle。"""
    indexed = small_model(attention_pattern="indexed", index_topk=100)
    dense = small_model()
    dense.load_state_dict(indexed.state_dict(), strict=False)
    ids = torch.randint(0, 64, (1, 9))
    output = indexed(ids)
    torch.testing.assert_close(output.logits, dense(ids).logits, atol=2e-6, rtol=2e-5)
    embedding_grad = torch.autograd.grad(
        output.auxiliary_loss, indexed.token_embedding.weight, allow_unused=True, retain_graph=True
    )[0]
    assert embedding_grad is None
    output.auxiliary_loss.backward()
    assert indexed.layers[0].attn.indexer.index_q.weight.grad.abs().sum() > 0
    return {"dense_limit_equal": True, "index_kl_backbone_gradient": None}


def lesson_14():
    """Engram/MTP 增加额外信号；Muon 的状态也要进入恢复契约。"""
    model = small_model(engram_table_size=31, mtp_depth=2)
    optimizer = HybridMuon(model)
    ids = torch.randint(0, 64, (1, 8))
    output = model(ids, labels=ids)
    assert output.mtp_loss is not None
    (output.loss + output.auxiliary_loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mtp.parameters())
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    restored = HybridMuon(model)
    restored.load_state_dict(saved)
    assert len(restored.state) == len(optimizer.state) > 0
    return {
        "mtp_has_gradient": True,
        "optimizer_states": len(restored.state),
        "muon_and_adamw": [g["kind"] for g in restored.param_groups],
    }


def lesson_15():
    """按域/预算选教师，不在 rollout 后根据“谁给的分高”挑教师。"""
    teachers = {"math:low": torch.randn(1, 2, 5), "math:max": torch.randn(1, 2, 5)}
    row = {"domain": "math", "effort": "max"}
    key = f"{row['domain']}:{row['effort']}"
    student = torch.randn(1, 2, 5, requires_grad=True)
    teacher = teachers[key].clone().requires_grad_(True)
    loss = distillation_loss(student, teacher, torch.ones(1, 2, dtype=torch.bool))
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0
    return {"route": key, "objective": "full_vocabulary_reverse_KL", "upstream_sampled_PG": False}


def lesson_16():
    """工具只接受类型化参数；校正系数冻结，过滤后不改原分母。"""
    action = parse_action('{"tool":"calculator","arguments":{"a":3,"b":4,"op":"add"}}')
    assert calculate(action["arguments"]) == 7
    current = torch.tensor([[0.0, 1.0]], requires_grad=True)
    behavior = torch.zeros(1, 2)
    loss, dropped = calibrated_policy_loss(
        current, behavior, torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool)
    )
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-0.5, 0.0]]))
    advantage = redistribute_advantages(
        torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0, 0.5, 1.0]), 3
    )
    assert advantage.sum().abs() < 1e-6
    return {
        "tool_result": 7,
        "dropped_fraction": dropped.item(),
        "calibrated_gradient": current.grad.tolist(),
        "gar_group_mean": round(advantage.mean().item(), 6),
    }


def lesson_17():
    """打包、拒绝修正、视觉 label 是三个独立契约，不能混成能力提升。"""
    weight = torch.randn(16, 16)
    codes, scales = pack_fp4(weight)
    recovered = unpack_fp4(codes, scales, weight.shape)
    packed_bytes = codes.numel() * codes.element_size() + scales.numel() * scales.element_size()
    assert codes.dtype == torch.uint8 and packed_bytes < weight.numel() * weight.element_size()
    corrected = correction_distribution(torch.tensor([0.7, 0.3]), torch.tensor([0.4, 0.6]))
    torch.testing.assert_close(corrected, torch.tensor([1.0, 0.0]))
    vision = VisionPrefixModel(small_model())
    ids = torch.randint(0, 64, (1, 5))
    result = vision(torch.randn(1, 3, 8, 8), ids, labels=ids)
    assert result.logits.shape == (1, 9, 64)
    result.loss.backward()
    assert vision.projector.projection.weight.grad.abs().sum() > 0
    return {
        "packed_bytes": packed_bytes,
        "weight_rmse": round((weight - recovered).square().mean().sqrt().item(), 6),
        "correction": corrected.tolist(),
        "visual_prefix_tokens": 4,
    }


def lesson_18():
    """正例控制不等于模型能力：正确数字、合法格式、实际工具事件分别验证。"""
    assert verify_arithmetic("<answer>7</answer>", 7) == 1
    assert verify_arithmetic("7 is somewhere in this sentence", 7) == 0
    task = ArithmeticTask(3, 4, require_tool=True)
    assert task.verify(7, used_matching_tool=False) == 0
    assert task.verify(7, used_matching_tool=True) == 1
    return {
        "format_attack_reward": 0,
        "missing_tool_reward": 0,
        "protocol_positive_control": 1,
        "learned_agent_capability": "not_measured",
    }


LESSONS = {f"{number:02}": globals()[f"lesson_{number:02}"] for number in range(1, 19)}


def run_lesson(chapter):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return LESSONS[chapter]()


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="逐章 CPU 原理实验，不是能力评测")
    parser.add_argument("chapter", choices=[*LESSONS, "all"])
    args = parser.parse_args()
    for chapter in LESSONS if args.chapter == "all" else [args.chapter]:
        print(f"第 {chapter} 章：" + json.dumps(run_lesson(chapter), ensure_ascii=False))


if __name__ == "__main__":
    main()
