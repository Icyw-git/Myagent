"""Pipeline 记忆轴装配。

working 使用纯内存记忆，适合离线运行；其他记忆类型仍保留为后续扩展。
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
    if kind == "working":
        from memory_src import MemoryConfig, MemoryTool

        return MemoryTool(
            user_id="pipeline_user",
            memory_config=MemoryConfig(),
            memory_types=["working"],
        )
    raise NotImplementedError(
        f"memory={kind} 尚未接入 pipeline 工厂；当前仅支持 off / working。"
        # 就绪后示例：
        # from memory_src.manager import MemoryManager
        # from memory_src.base import MemoryConfig
        # return MemoryManager(MemoryConfig(), enable_episodic=(kind!="working"), ...)
    )
