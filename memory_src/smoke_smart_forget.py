"""
智能遗忘（smart）最小冒烟 —— 只开 working，不依赖 Qdrant。

    D:\\Anaconda_envs\\envs\\aitest01_py310\\python.exe memory_src/smoke_smart_forget.py
"""

from __future__ import annotations

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

from base import MemoryConfig
from manager import MemoryManager


def main() -> None:
    mgr = MemoryManager(
        config=MemoryConfig(),
        user_id="smart_smoke",
        enable_working=True,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=False,
    )

    # A：重要但从未被检索 → 频率低，smart 下可能仍因重要性留下
    mgr.add_memory("重要事实：项目截止日期是周五", memory_type="working", importance=0.9)
    # B：不重要，但会被多次 retrieve → 频率抬高 keep_score
    mgr.add_memory("临时：会议室在三楼", memory_type="working", importance=0.2)
    # C：不重要且从不检索 → keep_score 应最低，优先被 smart 忘掉
    mgr.add_memory("噪音：随便写的草稿", memory_type="working", importance=0.05)

    wm = mgr.memory_types["working"]
    # 多次检索「会议室」，抬高 B 的 access_count
    for _ in range(5):
        mgr.retrieve_memories("会议室", limit=2, memory_types=["working"])

    print("--- 遗忘前 ---")
    for m in wm.memories:
        score = mgr._keep_score(m)
        ac = (m.metadata or {}).get("access_count", 0)
        print(f"  imp={m.importance:.2f} access={ac} keep={score:.3f} | {m.content[:24]}")

    # threshold=0.35：预期 C 被删；B 因访问次数可能留下；A 因重要性留下
    n = mgr.forget_memories(strategy="smart", threshold=0.35)
    print(f"[smart forget] dropped={n}")

    left = [m.content for m in wm.memories]
    print("--- 遗忘后 ---", left)

    assert any("截止日期" in c for c in left), "重要记忆不应被误删"
    assert not any("草稿" in c for c in left), "低重要+零访问应被忘掉"
    print("[ok] smart forget smoke passed")


if __name__ == "__main__":
    main()
