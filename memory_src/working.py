"""
工作记忆 WorkingMemory —— 对应 md 8.2.5 (1)

特点（原文）：
- 容量有限（默认50条）+ TTL自动清理
- 纯内存存储，访问速度极快
- 混合检索：TF-IDF向量化 + 关键词匹配

评分公式（原文，写死在这里方便你实现时对照）：
    base_relevance = vector_score*0.7 + keyword_score*0.3   (若 vector_score>0，否则退化为 keyword_score)
    time_decay      = self._calculate_time_decay(memory.timestamp)
    importance_weight = 0.8 + (memory.importance * 0.4)
    final_score = base_relevance * time_decay * importance_weight
"""
import math
from typing import List
from base import BaseMemory, MemoryItem, MemoryConfig
from datetime import datetime, timedelta
import jieba

class WorkingMemory(BaseMemory):
    """工作记忆实现"""

    def __init__(self, config: MemoryConfig):
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_minutes = config.working_memory_ttl or 60
        self.memories: List[MemoryItem] = []

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆

        步骤（原文流程，自己填逻辑）：
        1. self._expire_old_memories()   过期清理
        2. 若 len(self.memories) >= self.max_capacity，
           调用 self._remove_lowest_priority_memory() 腾出空间
        3. self.memories.append(memory_item)
        4. return memory_item.id
        """
        self._expire_old_memories()
        if len(self.memories)>=self.max_capacity:
            self._remove_lowest_priority_memory()

        self.memories.append(memory_item)
        return memory_item.id


    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：TF-IDF向量化 + 关键词匹配

        步骤：
        1. self._expire_old_memories()
        2. vector_scores = self._try_tfidf_search(query)   # dict: memory.id -> score
        3. 对 self.memories 逐条计算 final_score（见文件头公式）
        4. 按 final_score 降序排序，取前 limit 条返回
        """
        self._expire_old_memories()
        vector_scores=self._try_tfidf_search(query)
        final_scores=[]
        for memory in self.memories:
            keyword_score=self._calculate_keyword_score(query,memory.content)
            vector_score=vector_scores.get(memory.id,0.0)
            best_relevance=vector_score*0.7+keyword_score*0.3 if vector_score >0 else keyword_score
            time_decay=self._calculate_time_decay(memory.timestamp)
            importance_weight=0.8+(memory.importance*0.4)
            final_score=best_relevance*time_decay*importance_weight
            final_scores.append((memory,final_score))

        final_scores.sort(key=lambda x:x[1],reverse=True)
        return [memory for memory,_ in  final_scores[:limit]]







    # ---- 以下是 retrieve/add 依赖的私有辅助方法，md 正文没有展开实现，
    #      需要你自己设计。给你留了函数签名和职责说明。 ----

    def _expire_old_memories(self) -> None:
        """删除 timestamp 早于 (now - max_age_minutes) 的记忆"""

        from datetime import timedelta
        memories=[]
        cutoff = datetime.now()-timedelta(self.max_age_minutes)
        for m in self.memories:
            timestamp=datetime.fromisoformat(m.timestamp)
            if timestamp >=cutoff:
                memories.append(m)

        self.memories=memories





    def _remove_lowest_priority_memory(self) -> None:
        """删除 importance 最低（或综合优先级最低）的一条"""
        self.memories.sort(key=lambda x:x.importance)
        self.memories=self.memories[1:]



    def _try_tfidf_search(self, query: str) -> dict:
        """用 sklearn 的 TfidfVectorizer 对 self.memories 的 content 建索引，
        返回 {memory_id: 相似度分数}。
        建议：语料太少（比如少于2条）或 sklearn 未安装时捕获异常，返回空 dict，
        这样上层会自动退化成纯关键词匹配。
        """
        if len(self.memories)<2:
            return {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            contents=[' '.join(jieba.lcut(m.content)) for m in self.memories]
            tfidf_martix=TfidfVectorizer.fit_transform(contents+[' '.join(jieba.lcut(query))])
            query_vec=tfidf_martix[-1]
            memory_vecs=tfidf_martix[:-1]

            scores=cosine_similarity(query_vec,memory_vecs)[0]
            return {
                memory.id:score for memory,score in zip(self.memories,scores)

            }

        except Exception as e:
            return {}






    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """最简单的实现可以是 query 分词后，命中 content 的词数 / query 词数"""
        query_list=jieba.lcut(query)
        content_list=jieba.lcut(content)

        count=0
        for word in query_list:
            if word in content_list:
                count+=1

        return count/len(query_list) if query_list else 0.0






    def _calculate_time_decay(self, timestamp: str,decay_factor:float=0.1) -> float:
        """时间越新分数越高，可以参考感知记忆里给出的指数衰减模型
        (math.exp(-decay_factor * age_hours / 24))，或者设计自己的衰减函数。
        """
        try:
            memory_time=datetime.fromisoformat(timestamp)
            cutoff=datetime.now()-memory_time
            age_hours=cutoff.total_seconds()/3600
            return math.exp(-decay_factor*age_hours/24)
        except Exception as e:
            return 0.5

