"""命令行入口：一条 pipeline 跑全流程，也允许逐阶段观察。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from laptop_llm.config import ExperimentConfig
from laptop_llm.engine import load_inference_bundle, read_checkpoint, run_stage
from laptop_llm.evaluation import run_generation_evaluation
from laptop_llm.generation import GenerationConfig, TokenGenerator
from laptop_llm.server import create_app
from laptop_llm.tokenizer import train_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laptop-llm",
        description="从零训练、对齐并部署一台笔记本跑得动的现代小型 LLM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenizer_parser = subparsers.add_parser("train-tokenizer", help="训练 Byte-level BPE")
    tokenizer_parser.add_argument("--config", required=True)
    tokenizer_parser.add_argument("--force", action="store_true", help="覆盖已有 tokenizer")

    train_parser = subparsers.add_parser("train", help="运行 pretrain / sft / dpo 阶段")
    train_parser.add_argument("stage", choices=["pretrain", "sft", "dpo"])
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--init-from", help="从上一阶段 checkpoint 初始化权重")
    train_parser.add_argument("--resume", help="恢复本阶段 optimizer 与 step")
    train_parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0/mps")

    pipeline_parser = subparsers.add_parser("pipeline", help="依次运行 tokenizer→预训练→SFT→DPO")
    pipeline_parser.add_argument("--config", required=True)
    pipeline_parser.add_argument("--device", default="auto")
    pipeline_parser.add_argument("--force-tokenizer", action="store_true")
    pipeline_parser.add_argument("--skip-dpo", action="store_true")

    chat_parser = subparsers.add_parser("chat", help="加载 checkpoint 进入流式终端聊天")
    chat_parser.add_argument("--checkpoint", required=True)
    chat_parser.add_argument("--device", default="auto")
    chat_parser.add_argument("--dtype", default="auto")
    chat_parser.add_argument("--system", default="你是一个诚实、友好、简洁的中文助手。")
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("--max-new-tokens", type=int, default=160)

    serve_parser = subparsers.add_parser("serve", help="启动 OpenAI 兼容 API 与网页聊天")
    serve_parser.add_argument("--checkpoint", required=True)
    serve_parser.add_argument("--device", default="auto")
    serve_parser.add_argument("--dtype", default="auto")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--api-key", default=None, help="可选；也可通过 LAPTOP_LLM_API_KEY 环境变量设置"
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="运行固定生成评测并保留逐条结果")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--suite", required=True)
    evaluate_parser.add_argument("--output", default="artifacts/evaluation.jsonl")
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--dtype", default="auto")
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=96)

    inspect_parser = subparsers.add_parser("inspect", help="查看 checkpoint 结构与训练来源")
    inspect_parser.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train-tokenizer":
        config = ExperimentConfig.from_yaml(args.config)
        train_tokenizer_from_config(config, force=args.force)
        return
    if args.command == "train":
        config = ExperimentConfig.from_yaml(args.config)
        run_stage(
            args.stage,
            config,
            init_from=args.init_from,
            resume=args.resume,
            device_name=args.device,
        )
        return
    if args.command == "pipeline":
        run_pipeline(args)
        return
    if args.command == "chat":
        run_chat(args)
        return
    if args.command == "serve":
        run_server(args)
        return
    if args.command == "evaluate":
        summary = run_generation_evaluation(
            args.checkpoint,
            args.suite,
            args.output,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "inspect":
        inspect_checkpoint(args.checkpoint)


def train_tokenizer_from_config(config: ExperimentConfig, *, force: bool = False) -> Path:
    target = Path(config.tokenizer_path)
    if target.exists() and not force:
        print(f"复用已有 tokenizer: {target}（如需重训请加 --force）")
        return target
    tokenizer = train_tokenizer(
        config.tokenizer_corpus,
        target,
        vocab_size=config.tokenizer_vocab_size,
        min_frequency=config.tokenizer_min_frequency,
    )
    print(f"tokenizer 已保存：{target}（实际词表 {tokenizer.vocab_size}）")
    return target


def run_pipeline(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    train_tokenizer_from_config(config, force=args.force_tokenizer)
    pretrain = run_stage("pretrain", config, device_name=args.device)
    sft = run_stage("sft", config, init_from=pretrain, device_name=args.device)
    final = sft
    if not args.skip_dpo and "dpo" in config.stages:
        final = run_stage("dpo", config, init_from=sft, device_name=args.device)
    print("\n" + "=" * 80)
    print(f"完整流水线结束。最终模型：{final}")
    print(f"聊天：laptop-llm chat --checkpoint \"{final}\"")
    print(f"服务：laptop-llm serve --checkpoint \"{final}\"")


def run_chat(args: argparse.Namespace) -> None:
    model, tokenizer, device = load_inference_bundle(
        args.checkpoint, device_name=args.device, dtype_name=args.dtype
    )
    generator = TokenGenerator(model, tokenizer, device)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": args.system}]
    print(f"模型已加载到 {device}。输入 /reset 清空历史，/exit 退出。")
    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_text:
            continue
        if user_text == "/exit":
            print("再见！")
            break
        if user_text == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("（历史已清空）")
            continue

        candidate = [*messages, {"role": "user", "content": user_text}]
        budget = model.config.max_seq_len - generation_config.max_new_tokens
        prompt_ids = tokenizer.build_chat_prompt(
            candidate, add_generation_prompt=True, max_length=max(8, budget)
        )
        print("助手> ", end="", flush=True)
        pieces: list[str] = []
        for piece in generator.stream_text(prompt_ids, generation_config):
            print(piece, end="", flush=True)
            pieces.append(piece)
        print()
        answer = "".join(pieces)
        messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]
        )


def run_server(args: argparse.Namespace) -> None:
    import uvicorn

    api_key = args.api_key or os.getenv("LAPTOP_LLM_API_KEY")
    app = create_app(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        api_key=api_key,
    )
    print(f"网页聊天：http://{args.host}:{args.port}")
    print(f"API 文档：http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


def inspect_checkpoint(path: str | Path) -> None:
    data = read_checkpoint(path, map_location="cpu")
    model_config = data["model_config"]
    parameter_count = sum(tensor.numel() for tensor in data["model"].values())
    summary = {
        "path": str(Path(path).resolve()),
        "format_version": data.get("format_version"),
        "stage": data.get("stage"),
        "step": data.get("step"),
        "parameters_in_state_dict": parameter_count,
        "model_config": model_config,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
