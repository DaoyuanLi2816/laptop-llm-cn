"""奖励并不等于真理：规则限定任务，学习型奖励依赖标注者与覆盖面。"""

import re

import torch
from torch import nn


def verify_arithmetic(response: str, expected: int) -> float:
    """只接受完整、唯一的 <answer>整数</answer>；允许前面有一段可见 CoT。

    不用 eval 执行模型输出，不搜索任意位置的正确数字，不奖励重复 answer。
    CoT 不核验步骤正确性：最后答对不代表推理过程真实或可靠。
    """
    match = re.fullmatch(
        r"\s*(?:<think>(?:(?!</?think>|</?answer>)[\s\S])*</think>\s*)?"
        r"<answer>(-?(?:0|[1-9][0-9]{0,8}))</answer>\s*",
        response,
    )
    return float(match is not None and int(match.group(1)) == expected)


class ScalarHead(nn.Module):
    """同样一个线性头：接终点 hidden 是 reward，接每个状态 hidden 是 value。

    实际 PPO 使用独立 critic backbone，防止 value 更新偷偷改变 frozen reference。
    """

    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, hidden):
        return self.projection(hidden).squeeze(-1).float()


def terminal_scores(backbone, head, ids, attention_mask):
    output = backbone(ids, attention_mask=attention_mask)
    lengths = attention_mask.sum(-1).long()
    return head(output.hidden_states)[torch.arange(ids.size(0), device=ids.device), lengths - 1]
