"""同步 on-policy rollout，刻意不用高吞吐 serving 引擎隐藏概率对齐过程。"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class Rollout:
    ids: torch.Tensor  # [B,L]，右侧 padding
    attention_mask: torch.Tensor  # [B,L]，包括 prompt
    action_mask: torch.Tensor  # [B,L-1]，只包括回答，EOS 也算动作
    terminal: torch.Tensor  # [B,L-1]，真正结束而不是长度截断
    old_logp: torch.Tensor  # [B,L-1]，采样策略概率，更新过程中固定
    texts: list[str]


def action_log_probs(model, ids, attention_mask):
    output = model(ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1].float()
    logp = F.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return logp, output


@torch.no_grad()
def collect_rollout(model, tokenizer, prompts, max_new_tokens=16):
    """直接从 softmax(logits) 采样：temperature=1，不做 top-k/top-p。

    否则采样分布与训练 logp 不是同一个策略，importance ratio 从第零步就错。
    model.eval() 只关闭 dropout，不等于禁用梯度；后续更新同样保持 eval 模式。
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须为正")
    model.eval()
    device = next(model.parameters()).device
    sequences, starts, ended, texts = [], [], [], []
    for prompt in prompts:
        ids = tokenizer.build_chat_prompt(prompt)
        if len(ids) + max_new_tokens > model.config.max_seq_len:
            raise ValueError("RL prompt+回答预算超过上下文；请缩短 prompt 或提高 max_seq_len")
        starts.append(len(ids) - 1)
        current = torch.tensor([ids], device=device)
        past, response, done = None, [], False
        for _ in range(max_new_tokens):
            output = model(current, past_key_values=past, use_cache=True)
            past = output.past_key_values
            token = torch.multinomial(output.logits[:, -1].float().softmax(-1), 1)
            response.append(int(token.item()))
            if response[-1] in {tokenizer.eos_id, tokenizer.end_id}:
                done = True
                break
            current = token
        sequences.append(ids + response)
        ended.append(done)
        # 只剥离实际结尾 EOS；其他控制符保留给 verifier 拒绝，不能悄悄抹掉。
        texts.append(
            tokenizer.decode(response[:-1] if done else response, skip_special_tokens=False)
        )
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), tokenizer.pad_id, device=device, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    actions = torch.zeros_like(ids[:, :-1], dtype=torch.bool)
    terminal = torch.zeros_like(actions)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, : len(sequence)] = True
        actions[row, starts[row] : len(sequence) - 1] = True
        terminal[row, len(sequence) - 2] = ended[row]
    logp, _ = action_log_probs(model, ids, attention)
    return Rollout(ids, attention, actions, terminal, logp.detach(), texts)
