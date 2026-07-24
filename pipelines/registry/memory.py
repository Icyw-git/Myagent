"""
记忆轴装配（先占位）

目前只支持 off；working / episodic / rag 等 memory_src 接好后再填。
"""

from __future__ import annotations

from typing import Any, Optional


def build_memory(kind: str) -> Optional[Any]:
    """
    返回记忆组件；off 返回 None。
    非 off 先明确报错，避免静默当成没记忆。
    """
    if kind == "off":
        return None
    raise NotImplementedError(
        f"memory={kind} 尚未接入 pipeline 工厂；请先用 memory=off，"
        "等 MemoryTool / RAG 就绪后再扩展本文件。"
    )
