"""
情景记忆 EpisodicMemory —— 对应 md 8.2.5 (2)

特点（原文）：
- SQLite+Qdrant混合存储架构
- 支持时间序列和会话级检索
- 结构化过滤 + 语义向量检索

评分公式（原文）：
    base_relevance = vec_score*0.8 + recency_score*0.2
    importance_weight = 0.8 + (importance * 0.4)
    score = base_relevance * importance_weight

双写思路（本步核心）：
1. 先写 SQLite（结构化：可按时间/session/重要性过滤）
2. 再 encode + 写 Qdrant（向量：语义检索）
3. 若 Qdrant 失败，回滚删掉 SQLite 里刚写的那条，避免「一边有一边没有」
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

try:
    from .base import BaseMemory, MemoryItem, MemoryConfig
    from .storage import (
        SQLiteDocumentStore,
        QdrantVectorStore,
        create_embedding_model_with_fallback,
    )
except ImportError:
    from base import BaseMemory, MemoryItem, MemoryConfig
    from storage import (
        SQLiteDocumentStore,
        QdrantVectorStore,
        create_embedding_model_with_fallback,
    )


@dataclass
class Episode:
    """一条情景事件（add 里构造，再交给 _persist_episode 双写）"""

    episode_id: str
    session_id: str
    timestamp: str
    content: str
    context: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory(BaseMemory):
    """情景记忆实现"""

    # 结构化过滤时认这些 kwargs；都没传则不过滤
    _FILTER_KEYS = ("user_id", "session_id", "min_importance", "start_time", "end_time")

    def __init__(self, config: MemoryConfig):
        # 依赖 storage/ 目录下的 SQLiteDocumentStore、QdrantVectorStore，
        # 以及 embedding.py 里的 create_embedding_model_with_fallback()。
        # 这几个是 Tier 0（理解即可，直接复用/import 你已经跑通的实现）。
        self.config = config
        self.embedder = create_embedding_model_with_fallback()
        # 向量维度跟 embedder 走，避免和集合维度不一致
        vector_size = getattr(self.embedder, "dimension", None) or config.qdrant_vector_size

        self.doc_store = SQLiteDocumentStore(config.database_path)
        self.vector_store = QdrantVectorStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name="episodic_memory",
            vector_size=vector_size,
        )
        self.sessions = {}  # 会话索引: session_id -> [episode_id, ...]

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆

        步骤：
        1. 构造 Episode 对象（episode_id / session_id / timestamp / content / context）
           session_id 从 memory_item.metadata.get("session_id", "default") 取
        2. 更新 self.sessions 索引
        3. self._persist_episode(episode)  写入 SQLite + Qdrant
        4. return memory_item.id
        """
        # importance / user_id 在 MemoryItem 顶层，一并塞进 context，方便持久化
        context = dict(memory_item.metadata or {})
        context.setdefault("importance", memory_item.importance)
        context.setdefault("user_id", memory_item.user_id)

        episode = Episode(
            episode_id=memory_item.id,
            session_id=memory_item.metadata.get("session_id", "default"),
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
        )
        session_id = episode.session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)
        self._persist_episode(episode)
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：结构化过滤 + 语义向量检索

        步骤：
        1. candidate_ids = self._structured_filter(**kwargs)   # 时间范围/重要性等预过滤
        2. hits = self._vector_search(query, limit*5, kwargs.get("user_id"))
        3. 对每个 hit，若 self._should_include(hit, candidate_ids, kwargs) 为真：
             score = self._calculate_episode_score(hit)
             memory_item = self._create_memory_item(hit)
             加入结果列表
        4. 按 score 降序排序，取前 limit 条
        """
        candidate_ids = self._structured_filter(**kwargs)
        hits = self._vector_search(query, limit * 5, kwargs.get("user_id"))
        results = []
        for hit in hits:
            if self._should_include(hit, candidate_ids, kwargs):
                score = self._calculate_episode_score(hit)
                memory_item = self._create_memory_item(hit)
                results.append((score, memory_item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in results[:limit]]

    def _calculate_episode_score(self, hit) -> float:
        """情景记忆评分算法（公式见文件头，已经是完整公式，照抄即可）"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency(hit["metadata"]["timestamp"])
        importance = hit["metadata"].get("importance", 0.5)

        base_relevance = vec_score * 0.8 + recency_score * 0.2
        importance_weight = 0.8 + (importance * 0.4)

        return base_relevance * importance_weight

    # ---- 需要你自己实现的辅助方法 ----

    def _persist_episode(self, episode: Episode) -> None:
        """写入 SQLite（结构化数据）+ Qdrant（向量）

        顺序：先 SQLite，再 Qdrant；Qdrant 失败则回滚 SQLite。
        """
        importance = float(episode.context.get("importance", 0.5))
        user_id = episode.context.get("user_id", "default_user")

        # 1) 结构化写入
        self.doc_store.save_episode(
            episode_id=episode.episode_id,
            content=episode.content,
            timestamp=episode.timestamp,
            session_id=episode.session_id,
            user_id=user_id,
            importance=importance,
            context=episode.context,
        )

        # 2) 向量写入；失败则回滚结构化那一半
        try:
            vector = self.embedder.encode(episode.content)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            metadata = {
                "memory_id": episode.episode_id,
                "memory_type": "episodic",
                "content": episode.content,
                "timestamp": episode.timestamp,
                "session_id": episode.session_id,
                "user_id": user_id,
                "importance": importance,
            }
            self.vector_store.add_vectors(
                vectors=[vector],
                metadata=[metadata],
                ids=[episode.episode_id],
            )
        except Exception:
            self.doc_store.delete_episode(episode.episode_id)
            raise

    def _structured_filter(self, **kwargs) -> set:
        """根据 kwargs 里的时间范围/重要性等条件，从 SQLite 查出候选 id 集合

        若调用方没传任何过滤条件，返回空 set，
        由 _should_include 解释为「不过滤、全部保留」。
        """
        if not self._has_structured_constraints(kwargs):
            return set()
        return self.doc_store.filter_ids(
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            min_importance=kwargs.get("min_importance"),
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
        )

    def _vector_search(self, query: str, limit: int, user_id) -> list:
        """embedder.encode(query) 之后去 self.vector_store 检索，返回原始 hits"""
        query_vector = self.embedder.encode(query)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        where: Dict[str, Any] = {"memory_type": "episodic"}
        if user_id:
            where["user_id"] = user_id

        return self.vector_store.search_similar(
            query_vector=query_vector,
            limit=limit,
            where=where,
        )

    def _should_include(self, hit, candidate_ids, kwargs) -> bool:
        """结合 candidate_ids 判断这条 hit 是否保留"""
        # 没传结构化条件 → 不过滤
        if not self._has_structured_constraints(kwargs):
            return True
        mid = hit.get("id") or (hit.get("metadata") or {}).get("memory_id")
        return mid in candidate_ids

    def _calculate_recency(self, timestamp: str) -> float:
        """时间近因性得分，可以参考感知记忆里的指数衰减实现"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            age_hours = (datetime.now() - memory_time).total_seconds() / 3600
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    def _create_memory_item(self, hit) -> MemoryItem:
        """把向量库返回的 hit 转换回 MemoryItem"""
        meta = dict(hit.get("metadata") or {})
        mid = hit.get("id") or meta.get("memory_id")

        # payload 缺 content 时，回落查 SQLite
        content = meta.get("content")
        if not content and mid:
            row = self.doc_store.get_episode(mid)
            if row:
                content = row["content"]
                meta.setdefault("timestamp", row["timestamp"])
                meta.setdefault("importance", row["importance"])
                meta.setdefault("session_id", row["session_id"])
                meta.setdefault("user_id", row["user_id"])

        return MemoryItem(
            id=mid,
            content=content or "",
            memory_type="episodic",
            importance=float(meta.get("importance", 0.5)),
            timestamp=meta.get("timestamp") or datetime.now().isoformat(),
            metadata=meta,
            user_id=meta.get("user_id", "default_user"),
        )

    def _has_structured_constraints(self, kwargs: Dict[str, Any]) -> bool:
        """kwargs 里是否带了结构化过滤条件"""
        return any(kwargs.get(k) is not None for k in self._FILTER_KEYS)
