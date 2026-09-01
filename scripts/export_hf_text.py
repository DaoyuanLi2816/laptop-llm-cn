"""把 Hugging Face Dataset 的一个文本字段流式导出为预训练纯文本。

示例：
  pip install -e ".[data]"
  python scripts/export_hf_text.py \
    --dataset DATASET_ID --split train --field text --output data/train.txt \
    --max-documents 100000

数据集 ID、revision、许可证与是否需要 trust_remote_code 必须由使用者审查；
本脚本不会替你选择或静默信任远程代码。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", default=None, help="可选的 dataset config/subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="text")
    parser.add_argument("--revision", default=None, help="建议固定 commit SHA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('请先运行 pip install -e ".[data]"') from exc

    dataset = load_dataset(
        args.dataset,
        args.name,
        split=args.split,
        streaming=True,
        revision=args.revision,
        trust_remote_code=False,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get(args.field)
            if not isinstance(text, str) or not text.strip():
                continue
            # 一行一个文档；把内部换行归一为空格，保持缓存阶段的文档边界契约。
            handle.write(" ".join(text.split()) + "\n")
            written += 1
            if args.max_documents is not None and written >= args.max_documents:
                break
    print(f"已导出 {written} 个文档到 {output.resolve()}")


if __name__ == "__main__":
    main()
