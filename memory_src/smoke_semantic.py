"""
SemanticMemory 冒烟（向量 + 内存图）

    conda run -n aitest01_py310 python memory_src/smoke_semantic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_THIS))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from base import MemoryConfig, MemoryItem
from semantic import SemanticMemory


def main() -> None:
    config = MemoryConfig(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )
    sm = SemanticMemory(config)
    samples = [
        MemoryItem(
            id="20000000-0000-4000-8000-000000000001",
            content="Python 是一种广泛使用的编程语言，常用于机器学习",
            memory_type="semantic",
            importance=0.8,
            user_id="smoke_user",
        ),
        MemoryItem(
            id="20000000-0000-4000-8000-000000000002",
            content="React 是前端框架，适合构建用户界面",
            memory_type="semantic",
            importance=0.7,
            user_id="smoke_user",
        ),
    ]
    for s in samples:
        mid = sm.add(s)
        print("[add]", mid[:8], "entities=", list(sm.entities.keys())[:5])

    hits = sm.retrieve("机器学习用什么语言", limit=2, user_id="smoke_user")
    print("[retrieve]", [h.content[:30] for h in hits])
    assert hits and "Python" in hits[0].content
    print("[ok] semantic smoke passed")


if __name__ == "__main__":
    main()
