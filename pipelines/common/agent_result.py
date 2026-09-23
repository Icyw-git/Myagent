"""Agent 的统一运行结果，不改变现有 ``run() -> str`` 教学接口。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentResult:
    answer: str
    steps: int | None = None
    tool_calls: int | None = None
