"""教材入口也属于可测试接口：课程缺页或本地链接失效要让 CI 失败。"""

import re
from pathlib import Path


def test_curriculum_and_local_markdown_links_exist():
    root = Path(__file__).resolve().parent.parent
    chapters = list((root / "docs/course").glob("[0-9][0-9]-*.md"))
    assert len(chapters) == 18
    for file in [root / "README.md", *root.joinpath("docs").rglob("*.md")]:
        text = file.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#")[0]
            assert (file.parent / path).exists(), f"{file.name}: {target}"
