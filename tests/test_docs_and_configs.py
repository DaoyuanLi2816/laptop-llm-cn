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


def test_cli_chinese_output_survives_western_windows_encoding():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "laptop_llm", "--help"],
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        check=True,
    )
    assert "中文" in result.stdout.decode("utf-8") or "笔记本" in result.stdout.decode("utf-8")
