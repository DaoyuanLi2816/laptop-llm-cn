"""命令入口统一 UTF-8：英文 Windows 的管道默认 cp1252 无法输出中文。

只在 CLI/脚本入口调用，不在 import 包时修改宿主程序的全局 I/O。
pytest/StringIO 等替代流没有 reconfigure，保留原状。
"""

import sys


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
