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
"""

from typing import List
from base import BaseMemory, MemoryItem, MemoryConfig


class EpisodicMemory(BaseMemory):
    """情景记忆实现"""

    def __init__(self, config: MemoryConfig):
        # 依赖 storage/ 目录下的 SQLiteDocumentStore、QdrantVectorStore，
        # 以及 embedding.py 里的 create_embedding_model_with_fallback()。
        # 这几个是 Tier 0（理解即可，直接复用/import 你已经跑通的实现）。
        self.doc_store = None      # TODO: SQLiteDocumentStore(config.database_path)
        self.vector_store = None   # TODO: QdrantVectorStore(config.qdrant_url, config.qdrant_api_key)
        self.embedder = None       # TODO: create_embedding_model_with_fallback()
        self.sessions = {}         # 会话索引: session_id -> [episode_id, ...]

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆

        步骤：
        1. 构造 Episode 对象（episode_id / session_id / timestamp / content / context）
           session_id 从 memory_item.metadata.get("session_id", "default") 取
        2. 更新 self.sessions 索引
        3. self._persist_episode(episode)  写入 SQLite + Qdrant
        4. return memory_item.id
        """
        episode=Episode(

            episode_id=memory_item.id,
            session_id=memory_item.metadata.get('session_id','default'),
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=memory_item.metadata


        )
        session_id=episode.session_id
        if session_id not in self.sessions:
            self.sessions[session_id]=[]
        self.sessions[session_id].append(episode.episode_id)
        self._persist_episode(episode)
        return  memory_item.id


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
        candidate_ids=self._structured_filter(**kwargs)
        hits=self._vector_search(query,limit*5,kwargs.get('user_id'))
        results=[]
        for hit in hits:
            if self._should_include(hit,candidate_ids,kwargs):
                score=self._calculate_episode_score(hit)
                memory_item=self._create_memory_item(hit)
                results.append((score,memory_item))

        results.sort(key=lambda x:x[0],reverse=True)
        return [m for _,m in results[:limit]]




    def _calculate_episode_score(self, hit) -> float:
        """情景记忆评分算法（公式见文件头，已经是完整公式，照抄即可）"""
        vec_score = float(hit.get("score", 0.0))
        recency_score = self._calculate_recency(hit["metadata"]["timestamp"])
        importance = hit["metadata"].get("importance", 0.5)

        base_relevance = vec_score * 0.8 + recency_score * 0.2
        importance_weight = 0.8 + (importance * 0.4)

        return base_relevance * importance_weight

    # ---- 需要你自己实现的辅助方法 ----

    def _persist_episode(self, episode) -> None:
        """写入 SQLite（结构化数据）+ Qdrant（向量）"""
        raise NotImplementedError

    def _structured_filter(self, **kwargs) -> set:
        """根据 kwargs 里的时间范围/重要性等条件，从 SQLite 查出候选 id 集合"""
        raise NotImplementedError

    def _vector_search(self, query: str, limit: int, user_id) -> list:
        """embedder.encode(query) 之后去 self.vector_store 检索，返回原始 hits"""
        raise NotImplementedError

    def _should_include(self, hit, candidate_ids, kwargs) -> bool:
        """结合 candidate_ids 判断这条 hit 是否保留"""
        raise NotImplementedError

    def _calculate_recency(self, timestamp: str) -> float:
        """时间近因性得分，可以参考感知记忆里的指数衰减实现"""
        raise NotImplementedError

    def _create_memory_item(self, hit) -> MemoryItem:
        """把向量库返回的 hit 转换回 MemoryItem"""
        raise NotImplementedError
