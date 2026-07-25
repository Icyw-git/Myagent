"""
Pipeline 三维规格

范式轴 paradigm: simple | react | hybrid | plan | tot | reflection
工具轴 tools:     none | search | calc | search+calc | bench
记忆轴 memory:    off | working | episodic | rag（目前只实现 off）

bench：能力评测用（TicketDesk + calculator），见 eval/bench/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


PARADIGMS = ("simple", "react", "hybrid", "plan", "tot", "reflection")
TOOL_KINDS = ("none", "search", "calc", "search+calc", "bench")
MEMORY_KINDS = ("off", "working", "episodic", "rag")


@dataclass(frozen=True)
class PipelineSpec:
    paradigm: str = "react"
    tools: str = "search"
    memory: str = "off"

    def __post_init__(self) -> None:
        if self.paradigm not in PARADIGMS:
            raise ValueError(f"未知 paradigm={self.paradigm}，可选 {PARADIGMS}")
        if self.tools not in TOOL_KINDS:
            raise ValueError(f"未知 tools={self.tools}，可选 {TOOL_KINDS}")
        if self.memory not in MEMORY_KINDS:
            raise ValueError(f"未知 memory={self.memory}，可选 {MEMORY_KINDS}")

    @property
    def tag(self) -> str:
        """报告里用的短标签，例如 hybrid|search|off"""
        return f"{self.paradigm}|{self.tools}|{self.memory}"


# 兼容旧 A/B 入口的预设
PRESETS: Dict[str, PipelineSpec] = {
    # A：自制 Hybrid + search
    "a": PipelineSpec(paradigm="hybrid", tools="search", memory="off"),
    # B：框架形 ReAct + search
    "b": PipelineSpec(paradigm="react", tools="search", memory="off"),
}
