import os
from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


#pydantic中没有给默认值的参数需要在初始化时传入，否则会报错，给了默认值的参数可以传入，修改默认值
class Config(BaseModel):
    default_model:str='gpt-3.5-turbo'
    default_provider:str='openai'
    temperature:float=0.5
    max_tokens:Optional[int]=None

    debug:bool=False
    log_level:str='INFO'

    max_history:int=100

    @classmethod
    def from_env(cls):
        return cls(
            temperature=float(os.getenv('temperature','0.7')),
            max_tokens=int(os.getenv('max_tokens')) if os.getenv('max_tokens') else None,
            debug=os.getenv('debug','false').lower()=='true', #注意返回的是字符串，需要转换为bool
            log_level=os.getenv('log_level','INFO')

        )

    def to_dict(self):
        return self.model_dump() #这是pydantic v2的用法，返回字典形式的配置

# 小提示：如果你想让某个 Config 全局只有一份、到处 import 都拿到同一个实例（而不是每次 Config() 都新建一份默认值），可以在 config.py 里额外写一行：
# global_config = Config.from_env()


if __name__ =='__main__':
    cfg=Config.from_env() #类方法作用是从环境变量中读取配置并创建Config实例，不需要创建对象再调用实例方法
    print(cfg.to_dict())