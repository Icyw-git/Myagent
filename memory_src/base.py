"""
记忆系统基础设施（对应 HelloAgents 第八章 8.1.3 / 8.1.4 的 base.py）

这一层只做三件事：
1. MemoryItem  —— 所有记忆类型共用的「一条记忆」数据结构
2. MemoryConfig —— 容量、TTL、数据库地址等配置
3. BaseMemory  —— 四种记忆类型必须实现的统一接口（add / retrieve）

为什么先写它？
- WorkingMemory / EpisodicMemory / ... 都 `from base import ...`
- 没有这一层，后面任何类型都 import 失败，谈不上验证

注意：字段是按教程骨架反推的「最小可用版」。
全部四种类型跑通后，再去对照真实 hello_agents 包做字段核对（学习计划第六步）。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _new_memory_id() -> str:
    """生成记忆唯一 ID。用 uuid4 足够；后续若要可追溯可换成更短前缀。"""
    return str(uuid.uuid4())


@dataclass
class MemoryItem:
    """一条标准化记忆。

    字段说明（读代码时重点看这几个）：
    - id:          全局唯一标识，retrieve / remove 都靠它
    - content:     记忆正文（检索主要针对它）
    - memory_type: working / episodic / semantic / perceptual
    - timestamp:   ISO 字符串，用于 TTL 过期和时间衰减
    - importance:  0.0~1.0，影响评分里的 importance_weight
    - metadata:    扩展信息（session_id、modality、user_id 等）
    - user_id:     多用户隔离时用；WorkingMemory 阶段可先默认
    """

    content: str
    memory_type: str = "working"
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=_new_memory_id)
    user_id: str = "default_user"

    def __post_init__(self) -> None:
        # 重要性钳制到合法区间，避免评分公式被异常值带飞
        self.importance = max(0.0, min(1.0, float(self.importance)))


@dataclass
class MemoryConfig:
    """记忆系统配置。

    WorkingMemory 现阶段只用到前两个字段；
    后面 Episodic / Semantic 会用到 database / qdrant / neo4j 相关项。
    先全部留好默认值，避免每种类型各自硬编码。
    """

    # ---- WorkingMemory ----
    working_memory_capacity: int = 50   # 最多保留多少条
    working_memory_ttl: int = 60        # 超过多少分钟视为过期（TTL）

    # ---- 本地文档库（Episodic 等）----
    database_path: str = "./memory_data/memory.db"

    # ---- Qdrant 向量库 ----
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "hello_agents_vectors"
    qdrant_vector_size: int = 384

    # ---- Neo4j 图库（Semantic）----
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: str = "neo4j"


class BaseMemory(ABC):
    """所有记忆类型的抽象基类。

    学习目标：不管底层是纯内存、SQLite+Qdrant 还是 Neo4j，
    对外都只暴露同一套 add / retrieve —— 这样 MemoryManager 才能统一调度。
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        storage_backend: Any = None,
    ) -> None:
        self.config = config or MemoryConfig()
        # storage_backend 留给 Manager 注入共享存储；WorkingMemory 可以不用
        self.storage_backend = storage_backend

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """写入一条记忆，返回 memory_item.id。"""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """按 query 检索，返回最多 limit 条，已按相关性排好序。"""
