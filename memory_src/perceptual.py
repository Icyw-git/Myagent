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

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional

try:
    from .base import BaseMemory, MemoryItem, MemoryConfig
    from .storage import QdrantVectorStore, create_embedding_model_with_fallback
except ImportError:
    from base import BaseMemory, MemoryItem, MemoryConfig
    from storage import QdrantVectorStore, create_embedding_model_with_fallback


class PerceptualMemory(BaseMemory):
    """感知记忆实现"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 文本编码器：对应教程 get_text_embedder()，这里复用 storage/embedding
        self.text_embedder = create_embedding_model_with_fallback()
        # 缺组件（思路保留）：图像 CLIP / 音频 CLAP，装好依赖后再 _init_* 赋值
        self._clip_model = None  # TODO: self._init_clip_model()
        self._clap_model = None  # TODO: self._init_clap_model()

        self.vector_dim = getattr(self.text_embedder, "dimension", None) or 384
        self._image_dim = 512  # 按 CLIP 维度调整
        self._audio_dim = 512  # 按 CLAP 维度调整

        # 文本集合先接通；image/audio 集合等编码器就绪后再建（教程 QdrantConnectionManager 同思路）
        self.vector_stores: Dict[str, object] = {
            "text": QdrantVectorStore(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                collection_name="perceptual_text",
                vector_size=self.vector_dim,
            ),
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
            user_id = kwargs.get("user_id")
            target_modality = kwargs.get("target_modality")
            query_modality = kwargs.get("query_modality", target_modality or "text")
            query_vector = self._encode_data(query, query_modality)
            store = self._get_vector_store_for_modality(target_modality or query_modality)
            where = {"memory_type": "perceptual"}
            if user_id:
                where["user_id"] = user_id
            if target_modality:
                where["modality"] = target_modality
            hits = store.search_similar(
                query_vector=query_vector, limit=max(limit * 5, 20), where=where
            )
        except Exception:
            hits = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            vector_score = hit["score"]
            recency_score = self._calculate_recency_score(meta.get("timestamp", ""))
            importance = meta.get("importance", 0.5)
            base_relevance = vector_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight
            hit["combined_score"] = combined_score
        sorted_hits = sorted(hits, key=lambda x: x["combined_score"], reverse=True)
        top_hits = sorted_hits[:limit]
        # 原骨架写成 [item for _, item in top_hits]，但 hit 是 dict；这里转成 MemoryItem
        return [self._hit_to_memory_item(hit) for hit in top_hits]

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
        modality = (memory_item.metadata or {}).get("modality", "text")
        vector_store = self._get_vector_store_for_modality(modality)
        vector = self._encode_data(memory_item.content, modality)
        # 本仓库 QdrantVectorStore.add_vectors(vectors, metadata, ids) 需要三份列表
        meta = {
            "memory_id": memory_item.id,
            "memory_type": "perceptual",
            "modality": modality,
            "content": memory_item.content,
            "timestamp": memory_item.timestamp,
            "importance": memory_item.importance,
            "user_id": memory_item.user_id,
        }
        if memory_item.metadata:
            meta.update({k: v for k, v in memory_item.metadata.items() if k != "raw_data"})
        vector_store.add_vectors([vector], [meta], [memory_item.id])
        return memory_item.id

    # ---- 需要你自己实现的辅助方法 ----

    def _encode_data(self, data: str, modality: str):
        """按 modality 选择 text_embedder / _clip_model / _clap_model 编码"""
        if modality == "text":
            vec = self.text_embedder.encode(data)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)
        if modality == "image":
            # 缺组件：CLIP。思路：self._clip_model.encode_image(path_or_tensor)
            raise NotImplementedError("image 模态待接 CLIP，见 _init_clip_model")
        if modality == "audio":
            # 缺组件：CLAP。思路：self._clap_model.encode_audio(...)
            raise NotImplementedError("audio 模态待接 CLAP，见 _init_clap_model")
        raise ValueError(f"未知 modality={modality}")

    def _get_vector_store_for_modality(self, modality: str):
        store = self.vector_stores.get(modality, self.vector_stores["text"])
        if store is None:
            # 对应上面 image/audio 仍为 None 的占位
            raise NotImplementedError(f"modality={modality} 的 vector_store 尚未创建")
        return store

    def _init_clip_model(self):
        # 缺组件思路：pip 装 open-clip / transformers，加载后 self._clip_model = ...
        raise NotImplementedError("CLIP 未接入（多模态后续步骤）")

    def _init_clap_model(self):
        # 缺组件思路：加载 CLAP 权重后 self._clap_model = ...
        raise NotImplementedError("CLAP 未接入（多模态后续步骤）")

    def _hit_to_memory_item(self, hit: dict) -> MemoryItem:
        """把 Qdrant hit 转回 MemoryItem（retrieve 第 6 步用）"""
        meta = dict(hit.get("metadata") or {})
        mid = hit.get("id") or meta.get("memory_id")
        return MemoryItem(
            id=mid,
            content=meta.get("content") or "",
            memory_type="perceptual",
            importance=float(meta.get("importance", 0.5)),
            timestamp=meta.get("timestamp") or datetime.now().isoformat(),
            metadata=meta,
            user_id=meta.get("user_id", "default_user"),
        )
