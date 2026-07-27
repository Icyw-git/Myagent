"""
语义记忆 SemanticMemory —— 对应 md 8.2.5 (3)

特点（原文）：
- 使用HuggingFace中文预训练模型进行文本嵌入
- 向量检索进行快速相似度匹配
- 知识图谱存储实体和关系
- 混合检索策略：向量+图+语义推理

评分公式（原文）：
    base_relevance = vector_score*0.7 + graph_score*0.3
    importance_weight = 0.8 + (importance * 0.4)   # 范围 [0.8, 1.2]
    combined_score = base_relevance * importance_weight
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import jieba

try:
    from .base import BaseMemory, MemoryItem, MemoryConfig
    from .storage import QdrantVectorStore, create_embedding_model_with_fallback
except ImportError:
    from base import BaseMemory, MemoryItem, MemoryConfig
    from storage import QdrantVectorStore, create_embedding_model_with_fallback


# 教程里的实体/关系结构（最小字段）；接 spaCy/Neo4j 后可再加属性
@dataclass
class Entity:
    entity_id: str
    name: str
    label: str = "CONCEPT"
    memory_ids: Set[str] = field(default_factory=set)


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    memory_id: str = ""


class SemanticMemory(BaseMemory):
    """语义记忆实现"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 对应教程 get_text_embedder() / QdrantConnectionManager
        self.embedding_model = create_embedding_model_with_fallback()
        vector_size = getattr(self.embedding_model, "dimension", None) or config.qdrant_vector_size
        self.vector_store = QdrantVectorStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name="semantic_memory",
            vector_size=vector_size,
        )
        # 缺组件：Neo4jGraphStore(**neo4j_config)。思路：实体/关系双写图库；现用内存 dict 顶上
        self.graph_store = None  # TODO: Neo4jGraphStore(uri=config.neo4j_uri, ...)

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        # 检索时拼 MemoryItem 用（向量 payload 也有 content，这份是热缓存）
        self._items: Dict[str, MemoryItem] = {}

        # 缺组件：spaCy NER。思路：self.nlp = spacy.load("zh_core_web_sm")，失败则 None 降级
        self.nlp = None  # TODO: self._init_nlp()

    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆

        流程（原文，四步）：
        1. embedding = self.embedding_model.encode(memory_item.content)
        2. entities = self._extract_entities(memory_item.content)
           relations = self._extract_relations(memory_item.content, entities)
        3. 遍历 entities/relations，分别调用
           self._add_entity_to_graph(entity, memory_item)
           self._add_relation_to_graph(relation, memory_item)
           写入 Neo4j
        4. 构造 metadata（memory_id/entities/entity_count/relation_count），
           调用 self.vector_store.add_vectors(vectors=[embedding], metadata=[metadata], ids=[memory_item.id])
        """
        embedding = self.embedding_model.encode(memory_item.content)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content, entities)
        for entity in entities:
            self._add_entity_to_graph(entity, memory_item)
        for relation in relations:
            self._add_relation_to_graph(relation, memory_item)

        metadata = {
            "memory_id": memory_item.id,
            "memory_type": "semantic",
            "content": memory_item.content,
            "timestamp": memory_item.timestamp,
            "importance": memory_item.importance,
            "user_id": memory_item.user_id,
            "entities": [entity.entity_id for entity in entities],
            "entity_count": len(entities),
            "relation_count": len(relations),
        }
        self.vector_store.add_vectors(
            vectors=[embedding], metadata=[metadata], ids=[memory_item.id]
        )
        self._items[memory_item.id] = memory_item
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆

        流程：
        1. vector_results = self._vector_search(query, limit*2, user_id)
        2. graph_results = self._graph_search(query, limit*2, user_id)
        3. return self._combine_and_rank_results(vector_results, graph_results, query, limit)[:limit]
        """
        user_id = kwargs.get("user_id")
        vector_results = self._vector_search(query, limit * 2, user_id)
        graph_results = self._graph_search(query, limit * 2, user_id)
        ranked = self._combine_and_rank_results(vector_results, graph_results, query, limit)
        # combine 返回的是 dict 行；转成 MemoryItem 给 Manager 用
        return [self._row_to_item(r) for r in ranked[:limit]]

    def _combine_and_rank_results(self, vector_results, graph_results, query, limit):
        """混合排序结果 —— 这段 md 原文给了完整实现，直接按下面的逻辑填：

        1. combined = {}，先把 vector_results 灌进去，记 vector_score，graph_score=0.0
        2. 再把 graph_results 灌进去：id 已存在就补 graph_score，不存在就新建条目
        3. 对每条计算 combined_score（公式见文件头）
        4. 按 combined_score 降序排序，取前 limit 条返回
        """
        combined = {}
        for result in vector_results:
            combined[result["memory_id"]] = {
                **result,
                "vector_score": result.get("score", 0.0),
                "graph_score": 0.0,
            }
        for result in graph_results:
            memory_id = result["memory_id"]
            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            else:
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0),
                }
        for item in combined.items():
            revelance = item[1]["vector_score"] * 0.7 + item[1]["graph_score"] * 0.3
            importance_weight = 0.8 + (item[1].get("importance", 0.5) * 0.4)
            combined_score = revelance * importance_weight
            item[1]["combined_score"] = combined_score
        sorted_results = sorted(
            combined.values(), key=lambda x: x["combined_score"], reverse=True
        )
        return sorted_results[:limit]

    # ---- 需要你自己设计的部分（md 没给实现，属于真正的算法设计题） ----

    def _extract_entities(self, content: str) -> list:
        """用 self.nlp（spaCy）做 NER，抽取实体列表"""
        # 缺组件：spaCy。有 self.nlp 时走 NER；否则 jieba 词长>=2 顶上（可换回正规 NER）
        if self.nlp is not None:
            doc = self.nlp(content)
            return [
                Entity(entity_id=ent.text, name=ent.text, label=ent.label_)
                for ent in doc.ents
            ]
        tokens = [t.strip() for t in jieba.lcut(content) if len(t.strip()) >= 2]
        seen = set()
        out = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            out.append(Entity(entity_id=tok, name=tok))
            if len(out) >= 12:
                break
        return out

    def _extract_relations(self, content: str, entities: list) -> list:
        """基于句法/共现规则，从 content 和 entities 里抽取关系三元组"""
        # 简易共现：相邻实体 RELATED_TO（正式版可接依存句法 / LLM 抽三元组）
        names = [e.name for e in entities]
        return [
            Relation(head=names[i], relation="RELATED_TO", tail=names[i + 1])
            for i in range(len(names) - 1)
        ]

    def _add_entity_to_graph(self, entity, memory_item: MemoryItem) -> None:
        # 有 Neo4j 时：self.graph_store.merge_entity(...)；现写入内存 self.entities
        existing = self.entities.get(entity.entity_id)
        if existing is None:
            entity.memory_ids.add(memory_item.id)
            self.entities[entity.entity_id] = entity
        else:
            existing.memory_ids.add(memory_item.id)
        if self.graph_store is not None:
            # TODO: self.graph_store.upsert_entity(entity, memory_item)
            pass

    def _add_relation_to_graph(self, relation, memory_item: MemoryItem) -> None:
        self.relations.append(
            Relation(
                head=relation.head,
                relation=relation.relation,
                tail=relation.tail,
                memory_id=memory_item.id,
            )
        )
        if self.graph_store is not None:
            # TODO: self.graph_store.upsert_relation(relation, memory_item)
            pass

    def _vector_search(self, query: str, limit: int, user_id) -> list:
        qv = self.embedding_model.encode(query)
        if hasattr(qv, "tolist"):
            qv = qv.tolist()
        where: Dict[str, Any] = {"memory_type": "semantic"}
        if user_id:
            where["user_id"] = user_id
        hits = self.vector_store.search_similar(query_vector=qv, limit=limit, where=where)
        rows = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            rows.append(
                {
                    "memory_id": hit.get("id") or meta.get("memory_id"),
                    "score": float(hit.get("score", 0.0)),
                    "importance": float(meta.get("importance", 0.5)),
                    "content": meta.get("content", ""),
                    "timestamp": meta.get("timestamp"),
                    "user_id": meta.get("user_id", "default_user"),
                    "metadata": meta,
                }
            )
        return rows

    def _graph_search(self, query: str, limit: int, user_id) -> list:
        """在 Neo4j 里按实体名/关键词匹配，返回带 similarity 字段的结果"""
        # 缺组件：Neo4j。有 graph_store 时改 Cypher；现用内存实体名是否出现在 query
        if self.graph_store is not None:
            # TODO: return self.graph_store.search(query, limit=limit, user_id=user_id)
            pass

        tokens = set(t for t in jieba.lcut(query) if len(t.strip()) >= 2)
        scores: Dict[str, float] = {}
        for name, ent in self.entities.items():
            if name not in tokens and name not in query:
                continue
            for mid in ent.memory_ids:
                item = self._items.get(mid)
                if user_id and item and item.user_id != user_id:
                    continue
                scores[mid] = scores.get(mid, 0.0) + 1.0
        for rel in self.relations:
            if rel.head in tokens or rel.tail in tokens or rel.head in query or rel.tail in query:
                scores[rel.memory_id] = scores.get(rel.memory_id, 0.0) + 0.5

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        max_s = max((s for _, s in ranked), default=1.0) or 1.0
        rows = []
        for mid, s in ranked:
            item = self._items.get(mid)
            rows.append(
                {
                    "memory_id": mid,
                    "similarity": float(s) / float(max_s),
                    "importance": item.importance if item else 0.5,
                    "content": item.content if item else "",
                    "timestamp": item.timestamp if item else datetime.now().isoformat(),
                    "user_id": item.user_id if item else "default_user",
                    "metadata": dict(item.metadata) if item else {},
                }
            )
        return rows

    def _init_nlp(self):
        """加载 spaCy 中英文模型，做好 fallback（模型缺失时返回 None，
        并在 retrieve/_extract_entities 里降级处理，别让整个系统崩掉）"""
        try:
            import spacy

            # 缺模型时 pip install zh_core_web_sm / en_core_web_sm
            for name in ("zh_core_web_sm", "en_core_web_sm"):
                try:
                    self.nlp = spacy.load(name)
                    return self.nlp
                except Exception:
                    continue
        except Exception:
            pass
        self.nlp = None
        return None

    def _row_to_item(self, row: Dict[str, Any]) -> MemoryItem:
        mid = row["memory_id"]
        if mid in self._items:
            return self._items[mid]
        meta = dict(row.get("metadata") or {})
        return MemoryItem(
            id=mid,
            content=row.get("content") or meta.get("content") or "",
            memory_type="semantic",
            importance=float(row.get("importance", 0.5)),
            timestamp=row.get("timestamp") or datetime.now().isoformat(),
            metadata=meta,
            user_id=row.get("user_id", "default_user"),
        )
