"""教材入口也属于可测试接口：课程缺页或本地链接失效要让 CI 失败。"""

import re
from pathlib import Path

import pytest

from scripts.check_docs import RenderedPage, check_chapters
from scripts.docs_support import embed_sources, symbol_excerpt
from scripts.lesson_examples import LESSONS, run_lesson


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


def test_chapters_have_real_source_and_executable_experiments():
    chapters, excerpts = check_chapters()
    assert chapters == 18 and excerpts >= 45


@pytest.mark.parametrize("chapter", LESSONS)
def test_chapter_cpu_experiment(chapter):
    assert isinstance(run_lesson(chapter), dict)


def test_source_extraction_rejects_missing_symbols_and_path_escape():
    with pytest.raises(ValueError, match="找不到源码符号"):
        symbol_excerpt("laptop_llm/model.py", "NotAFunction")
    with pytest.raises(ValueError, match="非法源码路径"):
        symbol_excerpt("../outside.py", "f")
    code, start, end = symbol_excerpt("laptop_llm/model.py", "RMSNorm.forward")
    assert code.startswith("def forward") and end >= start


def test_render_audit_ignores_python_stars_but_checks_prose():
    page = RenderedPage()
    page.feed("<p>bad **format**</p><pre><code>value ** 2</code></pre>")
    assert "".join(page.text) == "bad **format**"


def test_source_markers_in_fenced_examples_are_not_expanded():
    marker = "<!-- source: laptop_llm/model.py::RMSNorm.forward -->"
    for fence in ("```", "~~~~"):
        example = f"{fence}text\n{marker}\n{fence}\n"
        rendered = embed_sources(example + marker, "documentation.md", "a" * 40)
        assert rendered.startswith(example)
        assert rendered.count(marker) == 1
        assert "def forward" in rendered and 'linenums="' in rendered
