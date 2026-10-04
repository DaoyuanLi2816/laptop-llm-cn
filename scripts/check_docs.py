"""不依赖训练库的文档审计：AST 标记、渲染文本、内部链接与片段锚点。"""

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docs_support import ROOT, SOURCE, symbol_excerpt  # noqa: E402


class RenderedPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.text = set(), [], []
        self.hidden = 0

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("name"):
            self.ids.add(attributes["name"])
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(attributes[key])
        if tag in {"pre", "code", "script", "style", "head"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"pre", "code", "script", "style", "head"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def check_chapters():
    chapters = sorted((ROOT / "docs/course").glob("[0-9][0-9]-*.md"))
    if len(chapters) != 18:
        raise ValueError("中文课程必须有 18 章")
    symbols = 0
    for chapter in chapters:
        text = chapter.read_text(encoding="utf-8")
        for heading in ["本章问题", "源码精读", "可运行小实验", "小结", "答案提示"]:
            if heading not in text:
                raise ValueError(f"{chapter.name} 缺少 {heading}")
        number = chapter.name[:2]
        if f"scripts/lesson_examples.py::lesson_{number}" not in text:
            raise ValueError(f"{chapter.name} 没有对应的可运行实验")
        for relative, symbol in SOURCE.findall(text):
            symbol_excerpt(relative, symbol)
            symbols += 1
    return len(chapters), symbols


def check_site(site):
    site = site.resolve()
    pages = {}
    for file in site.rglob("*.html"):
        parser = RenderedPage()
        parser.feed(file.read_text(encoding="utf-8"))
        pages[file.resolve()] = parser
    errors = []
    for file, parser in pages.items():
        visible = "".join(parser.text)
        if "**" in visible:
            position = visible.index("**")
            errors.append(
                f"{file.relative_to(site)} 有未渲染星号：{visible[max(0, position - 35) : position + 55]!r}"
            )
        for link in parser.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            path = unquote(parts.path)
            if path.startswith("/laptop-llm-cn/"):
                target = site / path.removeprefix("/laptop-llm-cn/")
            elif path.startswith("/"):
                target = site / path.lstrip("/")
            else:
                target = file.parent / path if path else file
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(site) or not target.is_file():
                errors.append(f"{file.relative_to(site)} 的链接失效：{link}")
            elif (
                parts.fragment
                and target in pages
                and unquote(parts.fragment) not in pages[target].ids
            ):
                errors.append(f"{file.relative_to(site)} 的锚点失效：{link}")
    if errors:
        raise ValueError("\n".join(errors))
    if not pages:
        raise ValueError("未找到构建后的 HTML")
    return len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    chapters, symbols = check_chapters()
    pages = check_site(args.site)
    # README 不在 docs/ 中，但它是读者的第一个入口，也必须审计实际渲染文本。
    import markdown

    readme = RenderedPage()
    readme.feed(
        markdown.markdown(
            (ROOT / "README.md").read_text(encoding="utf-8"),
            extensions=["tables", "fenced_code"],
        )
    )
    if "**" in "".join(readme.text):
        raise ValueError("README 有未渲染的双星号；请检查 HTML 混排和强调边界")
    print(
        f"Documentation verified: {chapters} chapters, {symbols} source excerpts, {pages} HTML pages"
    )


if __name__ == "__main__":
    main()
