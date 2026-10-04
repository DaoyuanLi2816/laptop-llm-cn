"""教材的源码桥：构建时按 AST 取真实函数，不把手抄代码当实现。

Markdown 内写 <!-- source: laptop_llm/model.py::RMSNorm.forward -->。
GitHub 原文件仍能阅读文字；文档站插入带原始行号、固定提交链接的代码。
缺文件/缺符号直接让构建失败；纯 AST 读取不会 import 或执行训练模块。
"""

import ast
import os
import posixpath
import re
import subprocess
import textwrap
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/DaoyuanLi2816/laptop-llm-cn"
SOURCE = re.compile(r"<!-- source: ([\w./-]+\.py)::([\w.]+) -->")
LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\s]+)\)")


def symbol_excerpt(relative, symbol, root=ROOT):
    """返回 (去公共缩进的原文, 起始行, 结束行)，拒绝越出仓库的路径。"""
    file = (root / relative).resolve()
    if not file.is_relative_to(root.resolve()) or file.suffix != ".py":
        raise ValueError(f"非法源码路径：{relative}")
    source = file.read_text(encoding="utf-8")
    nodes = ast.parse(source).body
    found = None
    for name in symbol.split("."):
        found = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ),
            None,
        )
        if found is None:
            raise ValueError(f"找不到源码符号：{relative}::{symbol}")
        nodes = found.body
    start = min([found.lineno, *[item.lineno for item in found.decorator_list]])
    end = found.end_lineno
    return textwrap.dedent("\n".join(source.splitlines()[start - 1 : end])), start, end


def source_ref():
    reference = os.environ.get("DOCS_SOURCE_REF") or os.environ.get("GITHUB_SHA")
    if reference:
        if not re.fullmatch(r"[0-9a-f]{40}", reference):
            raise ValueError("源码引用必须是完整 commit SHA")
        return reference
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def embed_sources(markdown, page_path, reference):
    def replace(match):
        relative, symbol = match.groups()
        code, start, end = symbol_excerpt(relative, symbol)
        source_page = "source/" + relative.removesuffix(".py") + ".md"
        local = posixpath.relpath(source_page, posixpath.dirname(page_path) or ".")
        remote = f"{REPO}/blob/{reference}/{relative}#L{start}-L{end}"
        return (
            f"[查看 `{symbol}` 的原文件]({remote}) · [完整源码页]({local})\n\n"
            f'```python title="{relative} · {symbol}" linenums="{start}"\n{code}\n```'
        )

    # 维护文档会展示标记的写法。围栏里的示例必须保持原样，不能嵌入第二层代码块。
    result, fence = [], None
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            delimiter, suffix = marker.groups()
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence) and not suffix.strip():
                fence = None
            result.append(line)
        else:
            result.append(line if fence else SOURCE.sub(replace, line))
    return "".join(result)


def on_files(files, config):
    from mkdocs.structure.files import File

    # Logo 只保留一份真实资产；虚拟文件只进入 site/，不写回 docs/。
    files.append(
        File.generated(
            config, "assets/logo.png", content=(ROOT / "laptop_llm/assets/logo-v3.png").read_bytes()
        )
    )
    reference = source_ref()
    modules = [*sorted((ROOT / "laptop_llm").rglob("*.py")), ROOT / "scripts/lesson_examples.py"]
    for file in modules:
        relative = file.relative_to(ROOT).as_posix()
        code = file.read_text(encoding="utf-8")
        content = (
            f"# {relative}\n\n"
            "这是本次构建对应的完整源码，不是自动生成的伪代码。\n\n"
            f"[固定版本原文件]({REPO}/blob/{reference}/{relative}) · "
            "[回到源码地图]("
            + posixpath.relpath(
                "source-map.md", "source/" + file.relative_to(ROOT).parent.as_posix()
            )
            + ")\n\n"
            f'```python title="{relative}" linenums="1"\n{code}\n```\n'
        )
        files.append(
            File.generated(
                config, "source/" + relative.removesuffix(".py") + ".md", content=content
            )
        )
    return files


def on_page_markdown(markdown, page, config, **kwargs):
    reference = source_ref()
    markdown = embed_sources(markdown, page.file.src_uri, reference)
    if "<!-- module-index -->" in markdown:
        modules = [
            *sorted((ROOT / "laptop_llm").rglob("*.py")),
            ROOT / "scripts/lesson_examples.py",
        ]
        index = "\n".join(
            f"- [{file.relative_to(ROOT).as_posix()}](source/{file.relative_to(ROOT).as_posix().removesuffix('.py')}.md)"
            for file in modules
        )
        markdown = markdown.replace("<!-- module-index -->", index)
    docs_root = Path(config.docs_dir).resolve()

    def repository_link(match):
        label, target = match.groups()
        if ":" in target or target.startswith("#"):
            return match.group(0)
        path, separator, anchor = target.partition("#")
        resolved = (docs_root / page.file.src_uri).parent.joinpath(path).resolve()
        if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(docs_root):
            suffix = "#" + anchor if separator else ""
            return f"[{label}]({REPO}/blob/{reference}/{quote(resolved.relative_to(ROOT).as_posix())}{suffix})"
        return match.group(0)

    return LINK.sub(repository_link, markdown)
