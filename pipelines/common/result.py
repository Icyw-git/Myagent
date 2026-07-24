"""
统一跑题结果 —— 两条 pipeline 对比时用同一套字段，方便并排打印。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    pipeline: str          # 标签：旧 A/B，或 factory 的 paradigm|tools|memory
    question: str
    answer: str
    elapsed_sec: float
    steps: int = 0         # 粗略步数（从日志里数）
    tool_calls: int = 0    # 粗略工具调用次数（从日志里数）
    error: str = ""
    raw_trace: str = ""    # 完整 stdout，肉眼看差异用
