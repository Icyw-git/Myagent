"""
对比用同一组题目 —— 两边只换装配，不换题，才公平。

可按需要往 CASES 里加条目；compare.py 会逐条跑。
"""

from __future__ import annotations

from typing import Dict, List

# id: 方便 --case 筛选
# question: 真正发给 Agent 的问题
# note: 这道题想测什么（给人看的）
CASES: List[Dict[str, str]] = [
    {
        "id": "qa",
        "question": "用三句话介绍一下武汉有哪些值得去的景点。",
        "note": "偏纯问答 / 规划，看两套路子怎么组织答案",
    },
    {
        "id": "search",
        "question": "请搜索一下今天武汉的天气概况，并给出出行建议。",
        "note": "需要调 search 工具（两边都注册了同一 search）",
    },
]
