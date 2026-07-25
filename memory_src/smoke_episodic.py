"""
EpisodicMemory 本地验证脚本（学习计划第二步：双写）

怎么跑（用 aitest01_py310，在项目根目录）：
    conda run -n aitest01_py310 python memory_src/smoke_episodic.py

验收：
1. add 之后 SQLite 能查到行、Qdrant 能搜到点
2. 查「前端」时李四相关应靠前
3. 能说清：先写 SQLite 再写 Qdrant；向量失败会回滚 SQLite
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from base import MemoryConfig, MemoryItem
from episodic import EpisodicMemory


def main() -> None:
    config = MemoryConfig(
        database_path=str(_ROOT / "memory_data" / "episodic_smoke.db"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )
    print(f"[config] sqlite = {config.database_path}")
    print(f"[config] qdrant = {config.qdrant_url}")

    em = EpisodicMemory(config)
    print(f"[ok] embedder dim = {getattr(em.embedder, 'dimension', '?')}")

    # 固定 id，重复跑 smoke 会 upsert 覆盖，避免 Qdrant 里堆重复点
    samples = [
        MemoryItem(
            id="00000000-0000-4000-8000-000000000001",
            content="2024年3月15日，用户张三完成了第一个Python机器学习项目",
            importance=0.8,
            memory_type="episodic",
            user_id="smoke_user",
            metadata={"session_id": "session_smoke", "event_type": "milestone"},
        ),
        MemoryItem(
            id="00000000-0000-4000-8000-000000000002",
            content="李四作为前端工程师，今天用React完成了页面重构",
            importance=0.7,
            memory_type="episodic",
            user_id="smoke_user",
            metadata={"session_id": "session_smoke", "event_type": "work"},
        ),
        MemoryItem(
            id="00000000-0000-4000-8000-000000000003",
            content="王五产品经理组织了用户体验评审会，讨论需求分析",
            importance=0.6,
            memory_type="episodic",
            user_id="smoke_user",
            metadata={"session_id": "session_smoke", "event_type": "meeting"},
        ),
    ]

    ids = []
    for item in samples:
        mid = em.add(item)
        ids.append(mid)
        row = em.doc_store.get_episode(mid)
        print(f"[add] {mid[:8]}... sqlite={'OK' if row else 'MISS'} | {item.content[:28]}...")

    # 向量检索
    query = "前端工程师 React"
    results = em.retrieve(query, limit=3, user_id="smoke_user")
    print("\n" + "=" * 60)
    print(f"查询: 「{query}」")
    print("=" * 60)
    for i, m in enumerate(results, 1):
        preview = m.content if len(m.content) <= 48 else m.content[:48] + "..."
        print(f"#{i} importance={m.importance:.2f}  {preview}")

    if results and "李四" in results[0].content:
        print("\n[ok] 排序符合预期：李四相关在前（双写 + 向量检索通路正常）")
    elif results:
        print("\n[!] 第一条不是李四，请对照内容看语义是否仍合理")
    else:
        print("\n[!] 无检索结果：检查 Qdrant / Embedding / .env")

    # 结构化过滤：提高 min_importance，应更容易只留下张三那条高分
    filtered = em.retrieve(
        "项目",
        limit=5,
        user_id="smoke_user",
        min_importance=0.75,
    )
    print("\n" + "=" * 60)
    print("查询: 「项目」 + min_importance=0.75")
    print("=" * 60)
    for i, m in enumerate(filtered, 1):
        print(f"#{i} importance={m.importance:.2f}  {m.content[:40]}...")

    print("\n核对口令：SQLite 有行 + Qdrant 能搜到 = 双写成功。")


if __name__ == "__main__":
    main()
