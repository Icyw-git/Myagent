from dataclasses import dataclass
from typing import Optional,Dict,Any
from datetime import datetime


@dataclass
class ContextPacket:
    """上下文信息包

    Attributes:
        content: 信息内容
        timestamp: 时间戳
        token_count: Token 数量
        relevance_score: 相关性分数(0.0-1.0)
        metadata: 可选的元数据
    """
    content:str
    timestamp:datetime
    token_count:int
    relevance_score:float
    metadata:Optional[Dict[str,Any]]=None


    def __post_init__(self): #这里的__post_init__作用是初始化属性，比如如果metadata为None，则初始化为空字典，和__init__的区别是__init__是初始化属性，__post_init__是初始化后属性
        if self.metadata is None:
            self.metadata = {}
        # 错误记录：曾写 min(1,0,self.relevance_score)——逗号把 1 和 0 当成两个参数，
        # min 永远得到 0，retrieve_score 恒为 0.0。正确：min(1.0, self.relevance_score)
        self.retrieve_score=max(0.0,min(1.0,self.relevance_score))


@dataclass 
class ContextConfig:
    """上下文构建配置

    Attributes:
        max_tokens: 最大 token 数量
        reserve_ratio: 为系统指令预留的比例(0.0-1.0)
        min_relevance: 最低相关性阈值
        enable_compression: 是否启用压缩
        recency_weight: 新近性权重(0.0-1.0)
        relevance_weight: 相关性权重(0.0-1.0)
    """

    max_tokens:int=3000
    reserve_ratio:float=0.2
    min_relevance:float=0.1
    enable_compression:bool=True
    recency_weight:float=0.3
    relevance_weight:float=0.7

    def __post_init__(self): #对上面初始化的属性进行检验
        assert 0.0<=self.reserve_ratio<=1.0, "reserve_ratio must be between 0.0 and 1.0"
        assert 0.0<=self.min_relevance<=1.0, "min_relevance must be between 0.0 and 1.0"

        assert abs(self.recency_weight+self.relevance_weight-1.0)<1e-6, "recency_weight and relevance_weight must sum to 1.0"