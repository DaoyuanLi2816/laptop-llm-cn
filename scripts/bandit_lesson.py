"""可控正例：两动作 bandit 上 GRPO 是否真的提高高奖励动作概率。

这是目标函数的学习动力学诊断，不是语言模型/数学推理能力证据。
用它区分“实现无效”与“LLM rollout 全零奖励导致没有信号”。
"""

import torch

from laptop_llm.posttraining.objectives import clipped_policy_loss, group_advantages


def learn(seed=7, steps=20):
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([logits], lr=0.3)
    before = float(logits.softmax(-1)[1].detach())
    for _ in range(steps):
        with torch.no_grad():
            actions = torch.multinomial(logits.softmax(-1), 128, replacement=True)
            old = logits.log_softmax(-1)[actions, None]
            advantages = group_advantages((actions == 1).float(), 8)[:, None]
        for _ in range(2):
            new = logits.log_softmax(-1)[actions, None]
            loss = clipped_policy_loss(new, old, advantages, torch.ones_like(new, dtype=torch.bool))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return before, float(logits.softmax(-1)[1].detach())


if __name__ == "__main__":
    before, after = learn()
    print(f"Controlled bandit P(rewarded action): {before:.4f} -> {after:.4f}")
    assert after > 0.85
