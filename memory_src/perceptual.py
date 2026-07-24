"""
感知记忆 PerceptualMemory —— 对应 md 8.2.5 (4)

特点（原文）：
- 支持多模态数据（文本、图像、音频等）
- 跨模态相似性搜索
- 感知数据的语义理解
- 支持内容生成和检索

评分公式（原文）：
    base_relevance = vector_score*0.8 + recency_score*0.2
    importance_weight = 0.8 + (importance * 0.4)
    combined_score = base_relevance * importance_weight

时间近因性用指数衰减（原文给了完整实现，见 _calculate_recency_score）。
"""

import math
from datetime import datetime
from typing import Dict, List
from base import BaseMemory, MemoryItem, MemoryConfig


class PerceptualMemory(BaseMemory):
    """感知记忆实现"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.text_embedder = None    # TODO: get_text_embedder()
        self._clip_model = None      # TODO: self._init_clip_model()
        self._clap_model = None      # TODO: self._init_clap_model()

        self.vector_dim = 384    # 按你的 text embedder 维度调整
        self._image_dim = 512    # 按 CLIP 维度调整
        self._audio_dim = 512    # 按 CLAP 维度调整

        self.vector_stores: Dict[str, object] = {
            "text": None,   # TODO: QdrantConnectionManager.get_instance(collection_name="perceptual_text", vector_size=self.vector_dim)
            "image": None,  # TODO: collection_name="perceptual_image", vector_size=self._image_dim
            "audio": None,  # TODO: collection_name="perceptual_audio", vector_size=self._audio_dim
        }

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索感知记忆（可筛模态；同模态向量检索+时间/重要性融合）

        流程（原文给了完整实现，直接对照写）：
        1. 取 user_id / target_modality / query_modality（默认 target_modality or "text"）
        2. query_vector = self._encode_data(query, query_modality)
           store = self._get_vector_store_for_modality(target_modality or query_modality)
        3. 构造 where 过滤条件（memory_type=perceptual，可选 user_id / modality）
        4. hits = store.search_similar(query_vector=..., limit=max(limit*5,20), where=where)
           —— 注意用 try/except 包住，查询失败时 hits=[]
        5. 对每个 hit：
             vector_score = hit["score"]
             recency_score = self._calculate_recency_score(hit["metadata"]["timestamp"])
             importance = hit["metadata"].get("importance", 0.5)
             combined_score = (公式见文件头)
        6. 按 combined_score 降序排序，取前 limit 条返回
        """
        try:
            user_id=kwargs.get("user_id")
            target_modality=kwargs.get("target_modality")
            query_modality=kwargs.get("query_modality", target_modality or "text")
            query_vector=self._encode_data(query, query_modality)
            store=self._get_vector_store_for_modality(target_modality or query_modality)
            where={"memory_type":"perceptual"}
            if user_id:
                where["user_id"]=user_id
            if target_modality:
                where["modality"]=target_modality
            hits=store.search_similar(query_vector=query_vector, limit=max(limit*5,20), where=where)
        except Exception:
            hits=[]
        for hit in hits:
            vector_score=hit["score"]
            recency_score=self._calculate_recency_score(hit["metadata"]["timestamp"])
            importance=hit["metadata"].get("importance", 0.5)
            base_relevance=vector_score*0.8 + recency_score*0.2
            importance_weight=0.8 + (importance * 0.4)
            combined_score=base_relevance * importance_weight
            hit["combined_score"]=combined_score
        sorted_hits=sorted(hits, key=lambda x: x["combined_score"], reverse=True)
        top_hits=sorted_hits[:limit]
        return  [item for _,item in top_hits]


    def _calculate_recency_score(self, timestamp: str) -> float:
        """时间近因性得分 —— md 原文给了完整实现，照抄：

        age_hours = (now - memory_time).total_seconds() / 3600
        decay_factor = 0.1
        recency_score = math.exp(-decay_factor * age_hours / 24)
        return max(0.1, recency_score)   # 异常时返回 0.5
        """
        try:
            memory_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            age_hours = (now - memory_time).total_seconds() / 3600
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    # ---- add() md 没有给出，需要你参照 episodic/semantic 的 add() 自己设计 ----

    def add(self, memory_item: MemoryItem) -> str:
        """添加感知记忆

        提示：
        - 根据 memory_item.metadata.get("modality") 判断走哪个 vector_store
        - 用对应的编码器（text/image/audio）生成向量
        - 调用 self.vector_stores[modality].add_vectors(...)
        """
        modality = memory_item.metadata.get("modality", "text")
        vector_store = self._get_vector_store_for_modality(modality)
        vector = self._encode_data(memory_item.content, modality)
        # 假设 vector_store 有 add_vectors 方法，返回 memory_id
        memory_id = vector_store.add_vectors([vector], [memory_item.metadata])
        return memory_id
    # ---- 需要你自己实现的辅助方法 ----

    def _encode_data(self, data: str, modality: str):
        """按 modality 选择 text_embedder / _clip_model / _clap_model 编码"""
        raise NotImplementedError

    def _get_vector_store_for_modality(self, modality: str):
        return self.vector_stores.get(modality, self.vector_stores["text"])

    def _init_clip_model(self):
        raise NotImplementedError

    def _init_clap_model(self):
        raise NotImplementedError
