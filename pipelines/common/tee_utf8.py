"""
把 stdout 同时打到控制台和 UTF-8 文件（带 BOM，方便 Windows/Cursor 打开不乱码）。

不要用 PowerShell Tee-Object 重定向中文——容易按系统代码页写坏文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO


class TeeUtf8:
    """sys.stdout 替换件：控制台 + utf-8-sig 文件双写。"""

    def __init__(self, console: TextIO, path: Path):
        self._console = console
        path.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig：带 BOM，记事本 / 部分编辑器不会误判成 GBK
        self._file = open(path, "w", encoding="utf-8-sig", newline="\n")
        self.path = path

    def write(self, s: str) -> int:
        self._console.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def reconfigure(self, *args, **kwargs) -> None:
        # Python 3.7+ TextIOWrapper.reconfigure；转发给控制台即可
        if hasattr(self._console, "reconfigure"):
            self._console.reconfigure(*args, **kwargs)

    @property
    def encoding(self) -> str:
        return "utf-8"

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())

    def close(self) -> None:
        self._file.close()
