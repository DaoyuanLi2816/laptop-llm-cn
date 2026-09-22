import re
from pathlib import Path

from laptop_llm.config import ExperimentConfig


def test_documentation_relative_links_exist():
    root = Path(__file__).resolve().parents[1]
    for document in [root / "README.md", *(root / "docs").rglob("*.md")]:
        for target in re.findall(r"\]\(([^)\s]+)\)", document.read_text(encoding="utf-8")):
            target = target.split("#")[0]
            if target and "://" not in target and not target.startswith("mailto:"):
                assert (document.parent / target).exists(), f"{document}: broken link {target}"


def test_all_shipped_configs_parse():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.yaml"):
        config = ExperimentConfig.from_yaml(path)
        assert config.model.dim > 0
        assert all(Path(value).exists() for value in config.data.values())
