"""
MemoryTool —— Agent 侧记忆工具

原骨架从 hello_agents 导入；改为本地 MemoryManager。
未实现的 action（summary/update/remove/clear_all）保留分支，旁注缺什么。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

try:
    from .base import MemoryConfig
    from .manager import MemoryManager
except ImportError:
    from base import MemoryConfig
    from manager import MemoryManager

try:
    from Tool import Tool, ToolParameter
except ImportError:
    class ToolParameter:  # type: ignore
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Tool:  # type: ignore
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description


class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memory_types: List[str] = None,
    ):
        super().__init__(
            name="memory",
            description="记忆工具 - 可以存储和检索对话历史、知识和经验",
        )

        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]
        self.current_session_id = None  # add 时懒创建
        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types,
        )

    def get_parameters(self) -> List[ToolParameter]:
        # 满足项目 Tool 抽象基类；参数以 action 为主
        return [
            ToolParameter(
                name="action",
                type="string",
                description="add/search/stats/forget/consolidate/…",
                required=True,
            ),
            ToolParameter(name="content", type="string", description="add 正文", required=False),
            ToolParameter(name="query", type="string", description="search 查询", required=False),
            ToolParameter(
                name="memory_type",
                type="string",
                description="working/episodic/semantic/perceptual",
                required=False,
                default="working",
            ),
        ]

    def run(self, parameters: dict) -> str:
        action = (parameters or {}).get("action", "search")
        kwargs = {k: v for k, v in (parameters or {}).items() if k != "action"}
        return self.execute(action, **kwargs)

    def execute(self, action: str, **kwargs):
        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "summary":
            return self._get_summary(**kwargs)
        elif action == "stats":
            return self._get_stats(**kwargs)
        elif action == "update":
            return self._update_memory(**kwargs)
        elif action == "remove":
            return self._remove_memory(**kwargs)
        elif action == "forget":
            return self._forget_memory(**kwargs)
        elif action == "consolidate":
            return self._consolidate_memory(**kwargs)
        elif action == "clear_all":
            return self._clear_all(**kwargs)
        return f"未知 action={action}"

    def _add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: str = None,
        modality: str = None,
        **metadata,
    ):
        try:
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                metadata.setdefault("modality", inferred)
                metadata.setdefault("raw_data", file_path)

            metadata.update(
                {
                    "session_id": self.current_session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=False,
            )

            return f"记忆已添加（ID：{memory_id[:8]}...）"
        except Exception as e:
            return f"添加记忆失败:{str(e)}"

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_type: str = None,
        memory_types: List[str] = None,
        min_importance: float = 0.1,
    ) -> str:
        try:
            if memory_types is None and memory_type:
                memory_types = [memory_type]

            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance,
            )

            if not results:
                return f"未找到与{query}相关的记忆"

            formatted_results = []
            formatted_results.append(f"找到{len(results)}条相关记忆：")

            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆",
                }.get(memory.memory_type, memory.memory_type)

                content_preview = (
                    memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
                )
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性：{memory.importance:.2f})"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"搜索记忆失败：{str(e)}"

    def _forget_memory(
        self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30
    ) -> str:
        # 原骨架方法名 _forget / forget_memomries 有笔误，这里接到 Manager.forget_memories
        # strategy 还可传 smart / capacity_based（见 MemoryManager.forget_memories 注释）
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days,
            )
            return f"已遗忘{count}条记忆（策略：{strategy}）"
        except Exception as e:
            return f"遗忘记忆失败:{str(e)}"

    def _consolidate_memory(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> str:
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
            )
            return (
                f"已整合{count}条记忆为长期记忆"
                f"（{from_type}->{to_type},阈值={importance_threshold}）"
            )
        except Exception as e:
            return f"整合记忆失败：{str(e)}"

    def _get_stats(self, **kwargs) -> str:
        return str(self.memory_manager.stats())

    # ---- 以下原骨架有 action，Manager 侧尚未做；先占位说明缺什么 ----

    def _get_summary(self, **kwargs) -> str:
        # 缺组件：用 LLM 对 retrieve 结果做摘要
        return "summary 尚未实现（思路：retrieve 后交给 LLM 压缩）"

    def _update_memory(self, **kwargs) -> str:
        # 缺组件：按 memory_id 改 content/importance，并同步向量库
        return "update 尚未实现（思路：改 SQLite/Qdrant payload）"

    def _remove_memory(self, **kwargs) -> str:
        # 缺组件：按 id 删除各 store 中的条目
        return "remove 尚未实现（思路：Working 列表删 + doc/vector delete）"

    def _clear_all(self, **kwargs) -> str:
        # 缺组件：危险操作，需按类型清空
        return "clear_all 尚未实现（思路：分类型 clear，并二次确认）"

    def _infer_modality(self, file_path: str) -> str:
        # 简单后缀推断；正式版可接 MIME
        lower = (file_path or "").lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return "image"
        if lower.endswith((".wav", ".mp3", ".flac", ".ogg")):
            return "audio"
        return "text"
