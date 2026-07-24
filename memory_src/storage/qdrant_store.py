"""
Qdrant 向量存储 —— 情景记忆的「语义检索」那一半

对外最小接口（给 EpisodicMemory 用）：
- add_vectors(vectors, metadata, ids)
- search_similar(query_vector, limit, where=None) -> List[hit]
  hit 格式：{"id": ..., "score": ..., "metadata": {...}}

注意：新版 qdrant-client 用 query_points，不用已废弃的 client.search。
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional


class QdrantVectorStore:
    """Qdrant 向量库薄封装"""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "episodic_memory",
        vector_size: int = 384,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._qmodels = qmodels

        # 云服务带 api_key；本地可为空
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """集合不存在就按 vector_size 创建；已存在则直接用。
        云端 Qdrant 对 filter 字段要求建 payload index，这里一并补上。
        """
        from qdrant_client.http import models as qmodels

        names = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        # 情景检索常用过滤字段（keyword）
        for field_name in ("memory_type", "user_id", "session_id", "memory_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # 已存在或权限/版本差异时忽略
                pass

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """写入向量 + payload（metadata）"""
        from qdrant_client.http import models as qmodels

        if not (len(vectors) == len(metadata) == len(ids)):
            raise ValueError("vectors / metadata / ids 长度必须一致")

        points = []
        for vec, meta, point_id in zip(vectors, metadata, ids):
            # Qdrant 点 id 可以是 uuid 字符串；非 uuid 时用 uuid5 映射，原 id 放进 payload
            qdrant_id = self._to_point_id(point_id)
            payload = dict(meta)
            payload["memory_id"] = point_id
            points.append(
                qmodels.PointStruct(id=qdrant_id, vector=vec, payload=payload)
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """语义检索，返回统一 hit 列表"""
        query_filter = self._build_filter(where) if where else None

        # 优先新 API；旧版再回退 search
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            raw = response.points
        except AttributeError:
            raw = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )

        hits = []
        for point in raw:
            payload = dict(point.payload or {})
            memory_id = payload.get("memory_id") or str(point.id)
            hits.append(
                {
                    "id": memory_id,
                    "score": float(point.score) if point.score is not None else 0.0,
                    "metadata": payload,
                }
            )
        return hits

    def delete(self, ids: List[str]) -> None:
        """按业务 id 删除（双写回滚用）"""
        from qdrant_client.http import models as qmodels

        point_ids = [self._to_point_id(i) for i in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=point_ids),
        )

    def _build_filter(self, where: Dict[str, Any]):
        """把简单 dict 条件转成 Qdrant Filter（等值匹配）"""
        from qdrant_client.http import models as qmodels

        must = []
        for key, value in where.items():
            must.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=value),
                )
            )
        return qmodels.Filter(must=must) if must else None

    @staticmethod
    def _to_point_id(memory_id: str):
        """Qdrant 要求 id 为无符号 int 或 UUID；统一映射成 uuid5"""
        try:
            return str(uuid.UUID(memory_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))
