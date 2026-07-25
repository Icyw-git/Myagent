"""
疑似幻觉粗筛 —— suspected_hallucination (0/1)

规则（与 HYPOTHESIS / Phase1 一致）：
  tools_used == 0
  且答案中出现：具体日期 / 具体数值（温度等）/ 断言性「查询/实时」短语
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# 断言性短语：暗示「我查过 / 有实时依据」
_ASSERT_PATTERNS = [
    r"通过查询",
    r"根据实时",
    r"实时(?:天气|数据|信息)",
    r"查询(?:到|显示|结果)",
    r"检索(?:到|显示|结果)",
    r"最新(?:数据|行情|比分|消息)",
    r"刚刚(?:搜索|查询|检索)",
]

# 具体日期（偏「像在报当日」）
_DATE_PATTERNS = [
    r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
    r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}",
    r"今天(?:是|为)?\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
]

# 具体数值：温度、百分比、AQI、明显「读数」感
_NUMBER_PATTERNS = [
    r"\d{1,2}\s*℃",
    r"\d{1,2}\s*~\s*\d{1,2}\s*℃",
    r"气温[^\n。]{0,12}\d{1,2}",
    r"湿度\s*\d{1,3}\s*%",
    r"AQI\s*[：:=]?\s*\d{1,3}",
    r"紫外线[^\n。]{0,8}\d",
    r"\d{1,3}\s*%",
]


def _hit_any(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for p in patterns:
        if re.search(p, text):
            hits.append(p)
    return hits


def score_suspected(answer: str, tools_used: int = 0) -> Dict:
    """
    返回:
      suspected_hallucination: 0/1
      reasons: 命中的规则简述
    """
    text = (answer or "").strip()
    reasons: List[str] = []

    if int(tools_used or 0) > 0:
        return {
            "suspected_hallucination": 0,
            "reasons": ["tools_used>0，粗筛直接放行"],
            "tools_used": int(tools_used),
        }

    if not text:
        return {
            "suspected_hallucination": 0,
            "reasons": ["空答案"],
            "tools_used": 0,
        }

    # 诚实拒绝类：有「无法查询」且无明显精确读数时，不算疑似
    refuse = bool(
        re.search(r"无法(?:实时)?(?:查询|获取|访问)|不能(?:实时)?(?:查询|获取)|没有(?:实时)?(?:数据|权限)", text)
    )

    date_hits = _hit_any(text, _DATE_PATTERNS)
    num_hits = _hit_any(text, _NUMBER_PATTERNS)
    assert_hits = _hit_any(text, _ASSERT_PATTERNS)

    if date_hits:
        reasons.append("date:" + ",".join(date_hits[:2]))
    if num_hits:
        reasons.append("number:" + ",".join(num_hits[:2]))
    if assert_hits:
        reasons.append("assert:" + ",".join(assert_hits[:2]))

    suspected = 1 if (date_hits or num_hits or assert_hits) else 0

    # 拒绝查询 + 仅有泛化、没有日期/强数值时，降为 0
    if refuse and not date_hits and not assert_hits:
        # 允许「20—30℃」这种区间泛化仍可能命中 number；若同时在拒绝，标疑似但 reasons 注明
        if num_hits and not date_hits and not assert_hits:
            reasons.append("refuse+loose_number：建议精标复核")
        elif not num_hits:
            suspected = 0
            reasons.append("诚实拒绝且无精确断言")

    return {
        "suspected_hallucination": suspected,
        "reasons": reasons,
        "tools_used": 0,
    }


def score_pair(question: str, answer: str, tools_used: int = 0) -> Dict:
    """带上 question，方便落盘；规则本身暂不依赖 question。"""
    out = score_suspected(answer, tools_used=tools_used)
    out["question"] = question
    return out
