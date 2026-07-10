from typing import Literal,Optional,Dict,Any
from pydantic import BaseModel #负责检查数据类型
from datetime import datetime
"""
统一消息格式
"""

class Message(BaseModel):
    role:Literal['user','system','tool','assistant'] #标注身份
    content:str #发送的内容
    timestamp:datetime=None #时间戳
    metadata:Optional[Dict[str,Any]]=None #其他元数据

    def __init__(self,content:str,role:str,**kwargs):
        super().__init__(
            role=role,
            content=content,
            timestamp=kwargs.get('timestamp',datetime.now()),
            metadata=kwargs.get('metadata',{})



        )
        #使用pydantic进行格式检查并初始化

    def to_dict(self)->Dict[str,Any]:
        return {
            'role':self.role,
            'content':self.content,
        }
    #只包含角色和消息内容，这是openai api格式

    def __str__(self):
        return f"[{self.role}] {self.content}" #转换为人类可读的格式
