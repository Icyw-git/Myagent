"""
MemoryManager —— 统一调度四种记忆

原骨架从 hello_agents.memory 导入；本仓库改为用 memory_src 本地实现。
教程里的 MemoryStore 共享后端：各类型已自带存储，这里先不建 Store（旁注保留思路）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    from .base import MemoryConfig, MemoryItem
    from .working import WorkingMemory
    from .episodic import EpisodicMemory
    from .semantic import SemanticMemory
    from .perceptual import PerceptualMemory
except ImportError:
    from base import MemoryConfig, MemoryItem
    from working import WorkingMemory
    from episodic import EpisodicMemory
    from semantic import SemanticMemory
    from perceptual import PerceptualMemory


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        # 缺组件思路：self.store = MemoryStore(self.config) 给各类型注入 storage_backend
        # self.store = MemoryStore(self.config)
        self.store = None

        # 原骨架写成 list，但下面按 key 赋值；改为 dict
        self.memory_types: Dict[str, object] = {}

        if enable_working:
            # WorkingMemory 当前构造只收 config（不强制 store）
            self.memory_types["working"] = WorkingMemory(self.config)
        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)
        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)
        if enable_perceptual:
            self.memory_types["perceptual"] = PerceptualMemory(self.config)

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        auto_classify: bool = False,
        **kwargs,
    ) -> str:
        # auto_classify：教程里可按内容自动选类型；此处保留参数，暂不实现
        if memory_type not in self.memory_types:
            raise ValueError(f"未启用记忆类型: {memory_type}，已启用={list(self.memory_types)}")
        meta = dict(metadata or {})
        meta.update(kwargs)
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            importance=importance,
            user_id=self.user_id,
            metadata=meta,
        )
        return self.memory_types[memory_type].add(item)

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        **kwargs,
    ) -> List[MemoryItem]:
        types = memory_types or list(self.memory_types.keys())
        merged: List[MemoryItem] = []
        for t in types:
            store = self.memory_types.get(t)
            if store is None:
                continue
            hits = store.retrieve(query, limit=limit, user_id=self.user_id, **kwargs)
            for m in hits:
                if m.importance >= min_importance:
                    merged.append(m)
        merged.sort(key=lambda m: m.importance, reverse=True)
        return merged[:limit]

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        src = self.memory_types.get(from_type)
        dst = self.memory_types.get(to_type)
        if src is None or dst is None or not hasattr(src, "memories"):
            return 0
        moved = 0
        remain = []
        for m in list(src.memories):
            if m.importance >= importance_threshold:
                dst.add(
                    MemoryItem(
                        content=m.content,
                        memory_type=to_type,
                        importance=m.importance,
                        timestamp=m.timestamp,
                        user_id=m.user_id,
                        metadata=dict(m.metadata or {}),
                    )
                )
                moved += 1
            else:
                remain.append(m)
        src.memories = remain
        return moved

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        # 目前只清 working；episodic/semantic 在外部库，遗忘策略以后再补
        wm = self.memory_types.get("working")
        if wm is None or not hasattr(wm, "memories"):
            return 0
        before = len(wm.memories)
        now = datetime.now()
        kept = []
        for m in wm.memories:
            drop = False
            if strategy == "importance_based" and m.importance < threshold:
                drop = True
            elif strategy == "age_based":
                try:
                    if now - datetime.fromisoformat(m.timestamp) > timedelta(days=max_age_days):
                        drop = True
                except Exception:
                    pass
            if not drop:
                kept.append(m)
        wm.memories = kept
        return before - len(kept)

    def stats(self) -> dict:
        out = {"user_id": self.user_id, "types": {}}
        for name, store in self.memory_types.items():
            if hasattr(store, "memories"):
                out["types"][name] = len(store.memories)
            elif hasattr(store, "_items"):
                out["types"][name] = len(store._items)
            else:
                out["types"][name] = "external"
        return out
