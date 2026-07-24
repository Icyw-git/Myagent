"""
统一嵌入服务 —— create_embedding_model_with_fallback()

优先级（跟教程一致的思路）：
1. 读 .env 里的 EMBED_*，走 OpenAI 兼容接口（百炼 / 自定义网关都行）
2. 失败则尝试本地 sentence-transformers
3. 再失败则 TF-IDF 兜底（维度不稳定，仅保证能跑通流程）
"""

from __future__ import annotations

import os
from typing import List, Union


def create_embedding_model_with_fallback():
    """返回一个带 encode() / dimension 的嵌入器实例"""
    # 1) 云端 / 兼容 OpenAI 的 embedding API
    try:
        embedder = _OpenAICompatibleEmbedding.from_env()
        if embedder is not None:
            return embedder
    except Exception:
        pass

    # 2) 本地 sentence-transformers
    try:
        return _LocalTransformerEmbedding()
    except Exception:
        pass

    # 3) 轻量兜底
    return _TFIDFEmbedding()


class _OpenAICompatibleEmbedding:
    """用 OpenAI SDK 调兼容接口（.env 里 EMBED_BASE_URL / EMBED_API_KEY）"""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dimension = None

    @classmethod
    def from_env(cls):
        api_key = os.getenv("EMBED_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("EMBED_BASE_URL")
        model = os.getenv("EMBED_MODEL_NAME") or "text-embedding-v3"
        if not api_key or not base_url:
            return None
        return cls(api_key=api_key, base_url=base_url, model=model)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vec = self.encode("dimension probe")
            self._dimension = len(vec)
        return self._dimension

    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        single = isinstance(text, str)
        inputs = [text] if single else list(text)
        resp = self.client.embeddings.create(model=self.model, input=inputs)
        # 按 index 排序，避免乱序
        data = sorted(resp.data, key=lambda x: x.index)
        vectors = [list(item.embedding) for item in data]
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors[0] if single else vectors


class _LocalTransformerEmbedding:
    """本地 sentence-transformers 兜底"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: Union[str, List[str]]):
        single = isinstance(text, str)
        inputs = [text] if single else list(text)
        vectors = self.model.encode(inputs, normalize_embeddings=True)
        result = [v.tolist() for v in vectors]
        return result[0] if single else result


class _TFIDFEmbedding:
    """最后兜底：保证双写流程能跑，检索质量不作保证"""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._fitted = False
        self._vectorizer = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: Union[str, List[str]]):
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        single = isinstance(text, str)
        inputs = [text] if single else list(text)
        docs = [" ".join(jieba.lcut(t)) for t in inputs]

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(max_features=self._dimension)

        # 每次重新 fit 只适合 smoke；正式环境应换云端/本地模型
        matrix = self._vectorizer.fit_transform(docs).toarray()
        padded = []
        for row in matrix:
            vec = list(map(float, row))
            if len(vec) < self._dimension:
                vec = vec + [0.0] * (self._dimension - len(vec))
            else:
                vec = vec[: self._dimension]
            # L2 归一化
            arr = np.array(vec, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            padded.append(arr.tolist())
        return padded[0] if single else padded
