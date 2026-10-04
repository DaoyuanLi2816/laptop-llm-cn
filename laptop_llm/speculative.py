"""精确投机采样参考实现：小草稿模型提议，大模型一次验证多个位置。

这里刻意全量重算前缀，便于验证分布；不是高速 serving 后端。
MTP/DFlash/DSpark 决定“怎样提议”，拒绝采样决定“怎样保持目标分布”。
"""

from dataclasses import dataclass

import torch


def correction_distribution(target, draft):
    residual = (target.double() - draft.double()).clamp_min(0)
    total = residual.sum()
    if not bool(total > 0):
        raise ValueError("相同分布不会发生拒绝；不存在可归一化的修正分布")
    return (residual / total).to(target.dtype)


@dataclass
class SpeculationStats:
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0

    @property
    def acceptance_rate(self):
        return self.accepted / max(1, self.proposed)


@torch.inference_mode()
def speculative_generate(
    target, draft, prompt, max_new_tokens=32, draft_tokens=3, temperature=1.0, stop_ids=None
):
    if not prompt or min(max_new_tokens, draft_tokens) < 1 or temperature < 0:
        raise ValueError("投机生成的 prompt/预算/温度非法")
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("师生词表不匹配；调用者还须验证 tokenizer 完全一致")
    target.eval()
    draft.eval()
    device = next(target.parameters()).device
    if next(draft.parameters()).device != device:
        raise ValueError("参考解码器要求师生在同一设备")
    limit = min(target.config.max_seq_len, draft.config.max_seq_len) - len(prompt)
    if limit < 1:
        raise ValueError("prompt 没有留下生成空间")
    desired = min(max_new_tokens, limit)
    ids, generated, stats = list(prompt), [], SpeculationStats()
    stops = set() if stop_ids is None else set(stop_ids)

    def distribution(logits):
        if temperature == 0:
            return torch.nn.functional.one_hot(logits.argmax(-1), logits.size(-1)).float()
        return (logits.float() / temperature).softmax(-1)

    while len(generated) < desired:
        proposals, draft_probabilities = [], []
        budget = min(draft_tokens, desired - len(generated))
        for _ in range(budget):
            input_ids = torch.tensor([ids + proposals], device=device)
            q = distribution(draft(input_ids).logits[0, -1])
            candidate = int(torch.multinomial(q, 1))
            proposals.append(candidate)
            draft_probabilities.append(q)
        # position len(ids)-1 预测第一个 proposal，最后位置给出全部接受后的 bonus。
        logits = target(torch.tensor([ids + proposals], device=device)).logits[0]
        stats.target_forwards += 1
        stats.proposed += budget
        all_accepted = True
        for offset, candidate in enumerate(proposals):
            p = distribution(logits[len(ids) - 1 + offset])
            q = draft_probabilities[offset]
            acceptance = (p[candidate] / q[candidate].clamp_min(1e-30)).clamp_max(1)
            if bool(torch.rand((), device=device) < acceptance):
                token = candidate
                stats.accepted += 1
            else:
                token = int(torch.multinomial(correction_distribution(p, q), 1))
                all_accepted = False
            generated.append(token)
            if token in stops or len(generated) == desired:
                return generated, stats
            if not all_accepted:
                break  # 拒绝后面的 proposal 都基于错误前缀，不能继续验收。
        accepted_this_round = offset + 1
        if all_accepted and len(generated) < desired:
            bonus = int(torch.multinomial(distribution(logits[-1]), 1))
            generated.append(bonus)
            if bonus in stops or len(generated) == desired:
                return generated, stats
            accepted_this_round += 1
        ids.extend(generated[-accepted_this_round:])
    return generated, stats
