"""
MemoryManager 冒烟（working 巩固到 episodic；不强制开 perceptual）

    conda run -n aitest01_py310 python memory_src/smoke_manager.py
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
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from base import MemoryConfig
from manager import MemoryManager
from memory_tool import MemoryTool


def main() -> None:
    config = MemoryConfig(
        database_path=str(_ROOT / "memory_data" / "manager_smoke.db"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )
    mgr = MemoryManager(
        config=config,
        user_id="smoke_user",
        enable_working=True,
        enable_episodic=True,
        enable_semantic=False,
        enable_perceptual=False,
    )
    mgr.add_memory("今天学习了工作记忆", memory_type="working", importance=0.9)
    mgr.add_memory("临时备注可以忘掉", memory_type="working", importance=0.05)
    print("[stats]", mgr.stats())

    n = mgr.consolidate_memories(from_type="working", to_type="episodic", importance_threshold=0.7)
    print("[consolidate]", n)
    forgot = mgr.forget_memories(strategy="importance_based", threshold=0.1)
    print("[forget]", forgot)

    hits = mgr.retrieve_memories("工作记忆", limit=3, memory_types=["working", "episodic"])
    print("[retrieve]", [h.memory_type + ":" + h.content[:20] for h in hits])

    tool = MemoryTool(
        user_id="smoke_user",
        memory_config=config,
        memory_types=["working"],
    )
    print(tool.execute("add", content="工具写入一条", memory_type="working", importance=0.6))
    print(tool.execute("search", query="工具", limit=2))
    print(tool.execute("stats"))
    print("[ok] manager smoke passed")


if __name__ == "__main__":
    main()
