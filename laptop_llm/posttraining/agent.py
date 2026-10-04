"""无网络、无 shell 的 agent 环境：模型动作 → 类型检查 → 工具 → 反馈 → 再行动。

TITO（token-in/token-out）：原始动作 ID 与行为 logp 原样进入训练；
工具反馈不是模型动作，绝不能给它策略梯度。为了沿用旧词表，工具反馈使用
显式 user 消息承载；工业 chat template 中应有独立 tool 角色。
"""

import json
from dataclasses import dataclass

import torch

from laptop_llm.posttraining.rollout import Rollout


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 不能含重复键")
        result[key] = value
    return result


def parse_action(text):
    if len(text) > 4096:
        raise ValueError("动作过长")
    try:
        action = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("动作必须是一个 JSON 对象") from exc
    if not isinstance(action, dict):
        raise ValueError("动作必须是 JSON 对象")
    if set(action) == {"answer"}:
        if type(action["answer"]) is not int or abs(action["answer"]) > 10**12:
            raise ValueError("answer 必须是有界整数")
        return action
    if set(action) != {"tool", "arguments"} or action["tool"] != "calculator":
        raise ValueError("只允许 calculator 工具，或最终 answer")
    args = action["arguments"]
    if not isinstance(args, dict) or set(args) != {"a", "b", "op"}:
        raise ValueError("calculator 参数必须是 a/b/op")
    if type(args["op"]) is not str or args["op"] not in {"add", "sub", "mul"}:
        raise ValueError("op 只允许 add/sub/mul")
    if any(type(args[key]) is not int or abs(args[key]) > 10**6 for key in ("a", "b")):
        raise ValueError("工具只接受有界整数，不执行代码")
    return action


def calculate(arguments):
    a, b, op = arguments["a"], arguments["b"], arguments["op"]
    return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[op]()


@dataclass
class ArithmeticTask:
    a: int
    b: int
    op: str = "add"
    require_tool: bool = True

    def __post_init__(self):
        parse_action(
            json.dumps(
                {"tool": "calculator", "arguments": {"a": self.a, "b": self.b, "op": self.op}}
            )
        )
        if type(self.require_tool) is not bool:
            raise ValueError("require_tool 必须是 bool")

    def prompt(self):
        rule = '每轮只输出 JSON。调用格式：{"tool":"calculator","arguments":{"a":整数,"b":整数,"op":"add/sub/mul"}}。最终格式：{"answer":整数}。'
        request = f"计算 {self.a} {self.op} {self.b}。" + (
            "必须先调用工具。" if self.require_tool else ""
        )
        return [{"role": "system", "content": rule}, {"role": "user", "content": request}]

    def verify(self, final, used_matching_tool):
        # 标准答案只存在可信 verifier 中，不暴露给模型或奖励模型。
        expected = calculate({"a": self.a, "b": self.b, "op": self.op})
        return float(final == expected and (used_matching_tool or not self.require_tool))


@torch.no_grad()
def collect_agent_rollout(
    model, tokenizer, tasks, max_new_tokens=48, max_turns=3, policy_version=0
):
    if min(max_new_tokens, max_turns) < 1:
        raise ValueError("agent 预算必须为正")
    model.eval()
    device = next(model.parameters()).device
    traces, sequences, flags, probabilities, terminals = [], [], [], [], []
    for task in tasks:
        ids = tokenizer.build_chat_prompt(task.prompt())
        if len(ids) >= model.config.max_seq_len:
            raise ValueError("agent prompt 超过上下文")
        actions = [False] * len(ids)
        logps = [0.0] * len(ids)
        events, used_tool, reward, finished = [], False, 0.0, False
        reason = "turn_limit"
        for turn in range(max_turns):
            remaining = model.config.max_seq_len - len(ids)
            if remaining < 1:
                reason = "context_limit"
                break
            output = model(torch.tensor([ids], device=device), use_cache=True)
            cache, generated, stopped = output.past_key_values, [], False
            for _ in range(min(max_new_tokens, remaining)):
                log_distribution = output.logits[:, -1].float().log_softmax(-1)
                token = torch.multinomial(log_distribution.exp(), 1)
                token_id = int(token.item())
                ids.append(token_id)
                actions.append(True)
                logps.append(float(log_distribution[0, token_id]))
                generated.append(token_id)
                if token_id in {tokenizer.end_id, tokenizer.eos_id}:
                    stopped = True
                    break
                if len(ids) < model.config.max_seq_len:
                    output = model(token, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
            text = tokenizer.decode(
                generated[:-1] if stopped else generated, skip_special_tokens=False
            )
            events.append({"turn": turn, "actor": "model", "text": text, "token_ids": generated})
            if not stopped:
                reason = "token_limit"
                break
            try:
                action = parse_action(text)
                if "answer" in action:
                    reward = task.verify(action["answer"], used_tool)
                    finished, reason = True, "final_answer"
                    break
                arguments = action["arguments"]
                observed = {"result": calculate(arguments)}
                used_tool |= arguments == {"a": task.a, "b": task.b, "op": task.op}
            except ValueError as exc:
                observed = {"error": str(exc)}
            events.append({"turn": turn, "actor": "environment", "observation": observed})
            observation = [
                tokenizer.token_id("<|user|>"),
                *tokenizer.encode("工具反馈：" + json.dumps(observed, ensure_ascii=False)),
                tokenizer.end_id,
                tokenizer.assistant_id,
            ]
            if len(ids) + len(observation) >= model.config.max_seq_len:
                reason = "context_limit"
                break
            ids.extend(observation)
            actions.extend([False] * len(observation))
            logps.extend([0.0] * len(observation))
        if not any(actions):
            raise ValueError("agent 轨迹没有生成动作")
        traces.append(
            {
                "events": events,
                "reward": reward,
                "reason": reason,
                "policy_version": policy_version,
                "quality": 1 / max(1, len(events)),
            }
        )
        sequences.append(ids)
        flags.append(actions)
        probabilities.append(logps)
        terminals.append(finished)
    length = max(map(len, sequences))
    ids = torch.full((len(tasks), length), tokenizer.pad_id, device=device, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    action_mask = torch.zeros_like(ids[:, :-1], dtype=torch.bool)
    old_logp = torch.zeros_like(action_mask, dtype=torch.float32)
    terminal = torch.zeros_like(action_mask)
    for row, sequence in enumerate(sequences):
        n = len(sequence)
        ids[row, :n] = torch.tensor(sequence, device=device)
        attention[row, :n] = True
        # flags[i] 标记 token_i；策略 logits_(i-1) 才预测这个动作。
        action_mask[row, : n - 1] = torch.tensor(flags[row][1:], device=device)
        old_logp[row, : n - 1] = torch.tensor(probabilities[row][1:], device=device)
        last_action = action_mask[row].nonzero()[-1, 0]
        terminal[row, last_action] = terminals[row]
    rollout = Rollout(
        ids,
        attention,
        action_mask,
        terminal,
        old_logp,
        [json.dumps(trace, ensure_ascii=False) for trace in traces],
    )
    return rollout, traces
