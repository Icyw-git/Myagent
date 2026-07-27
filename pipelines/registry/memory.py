"""
记忆轴装配（先占位）

目前只支持 off；working / episodic / rag 等 memory_src 接好后再填。

旁注（接入思路，暂不改 pipeline 行为）：
- memory_src.manager.MemoryManager 已可按类型开关装配
- memory_src.memory_tool.MemoryTool 可挂到 Agent 工具表
- 等 paradigm.build_* 真正消费 memory 参数时，再在本文件 return Manager/Tool
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
        # 就绪后示例：
        # from memory_src.manager import MemoryManager
        # from memory_src.base import MemoryConfig
        # return MemoryManager(MemoryConfig(), enable_episodic=(kind!="working"), ...)
    )
