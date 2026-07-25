"""
WorkingMemory 本地验证脚本（学习计划第一步的验收点）

怎么跑（在项目根目录 d:\\myagent）：
    python memory_src/smoke_working.py

验收：看排序是否符合语义预期；vector_score 不应全是 0。
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from base import MemoryConfig, MemoryItem
from working import WorkingMemory


def explain(wm: WorkingMemory, query: str, limit: int = 5):
    """在脚本里拆分项分数，不改 WorkingMemory 本身。"""
    wm._expire_old_memories()
    vector_scores = wm._try_tfidf_search(query)
    rows = []
    for memory in wm.memories:
        keyword_score = wm._calculate_keyword_score(query, memory.content)
        vector_score = vector_scores.get(memory.id, 0.0)
        base_relevance = (
            vector_score * 0.7 + keyword_score * 0.3
            if vector_score > 0
            else keyword_score
        )
        time_decay = wm._calculate_time_decay(memory.timestamp)
        importance_weight = 0.8 + (memory.importance * 0.4)
        final_score = base_relevance * time_decay * importance_weight
        rows.append(
            (
                memory,
                final_score,
                {
                    "vector_score": float(vector_score),
                    "keyword_score": keyword_score,
                    "base_relevance": base_relevance,
                    "time_decay": time_decay,
                    "importance_weight": importance_weight,
                },
            )
        )
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:limit]


def _print_ranking(title: str, rows) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if not rows:
        print("（无结果）")
        return
    for i, (memory, final_score, b) in enumerate(rows, 1):
        preview = memory.content if len(memory.content) <= 40 else memory.content[:40] + "..."
        print(f"\n#{i}  final={final_score:.4f}  importance={memory.importance:.2f}")
        print(f"    content : {preview}")
        print(
            f"    vector={b['vector_score']:.4f}  "
            f"keyword={b['keyword_score']:.4f}  "
            f"base={b['base_relevance']:.4f}  "
            f"time={b['time_decay']:.4f}  "
            f"imp_w={b['importance_weight']:.4f}"
        )


def main() -> None:
    config = MemoryConfig(working_memory_capacity=20, working_memory_ttl=120)
    wm = WorkingMemory(config)

    samples = [
        MemoryItem(
            content="用户张三是一名Python开发者，专注于机器学习和数据分析",
            importance=0.8,
            memory_type="working",
        ),
        MemoryItem(
            content="李四是前端工程师，擅长React和Vue.js开发",
            importance=0.7,
            memory_type="working",
        ),
        MemoryItem(
            content="王五是产品经理，负责用户体验设计和需求分析",
            importance=0.6,
            memory_type="working",
        ),
        MemoryItem(
            content="赵六正在准备前端面试，复习JavaScript和CSS基础",
            importance=0.5,
            memory_type="working",
        ),
    ]
    for item in samples:
        mid = wm.add(item)
        print(f"[add] {mid[:8]}... | {item.content[:28]}...")

    print(f"\n当前工作记忆条数: {len(wm.memories)}")

    q1 = "前端工程师"
    rows1 = explain(wm, q1, limit=4)
    _print_ranking(f"查询: 「{q1}」  （预期：李四 ≈ 最前，其次赵六）", rows1)

    if rows1:
        top = rows1[0][0].content
        if "李四" not in top and "前端" not in top:
            print("\n[!] 警告：第一条不像前端相关，请对照上面分项检查 TF-IDF / 关键词。")
        if all(r[2]["vector_score"] == 0 for r in rows1):
            print("[!] 警告：vector_score 全为 0，TF-IDF 可能未生效（检查 sklearn）。")
        else:
            print("\n[ok] 已看到非零 vector_score，混合检索通路正常。")

    q2 = "机器学习数据分析"
    rows2 = explain(wm, q2, limit=4)
    _print_ranking(f"查询: 「{q2}」  （预期：张三 ≈ 最前）", rows2)

    if rows2 and "张三" not in rows2[0][0].content:
        print("\n[!] 警告：第一条不是张三，请看 keyword/vector 分是否符合直觉。")
    elif rows2:
        print("\n[ok] 「机器学习」查询排序符合预期（张三在前）。")

    q3 = "区块链挖矿"
    rows3 = explain(wm, q3, limit=4)
    _print_ranking(f"查询: 「{q3}」  （预期：整体分很低，或结果很少）", rows3)

    print("\n" + "-" * 60)
    print("核对：挑一条用 final = base * time * imp_w 心算是否对得上。")


if __name__ == "__main__":
    main()
