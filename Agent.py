from typing import Optional,List

from Message import Message
from Config import Config
from abc import ABC, abstractmethod #使用abc模块来定义抽象基类和抽象方法
from my_llm import MyLLM

class Agent(ABC):
    def __init__(self,
                 name:str,
                 llm:MyLLM,
                 system_prompt:Optional[str]=None,
                 config:Optional[Config]=None):
        self.name=name
        self.llm=llm
        self.system_prompt=system_prompt
        self.config=config or Config()
        self._history:List[Message]=[]

    @abstractmethod
    def run(self,input_text:str,**kwargs): #agent的核心方法，接收用户输入并返回响应
        pass

    def add_history(self,history:Message): #添加历史记录，history是Message对象
        self._history.append(history)

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name},provider={self.llm.provider})"
