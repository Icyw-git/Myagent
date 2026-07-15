from abc import  ABC, abstractmethod
from collections.abc import Callable
from typing import  Dict,List,Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    name: str
    type: str #工具参数类型，可以是str,int,float,bool,list,dict等
    description: str
    required: bool = True #required表示是否必须传入参数，默认是True

    default: Any = None


class Tool(ABC):
    def __init__(self,name:str,description:str):
        self.name=name
        self.description=description

    @abstractmethod
    def run(self,parameters:Dict[List,Any])->str:
        pass

    @abstractmethod
    def get_parameters(self)->List[ToolParameter]:
        pass


class ToolRegistry:
    def __init__(self):
        self._tools:dict[str,Tool]={}
        self._functions:dict[str,dict[str,Any]]={} #存储函数工具，键是函数名，值是包含description和func的字典


    def register_tool(self,tool:Tool): #第一种方式注册工具，传入Tool对象
        if tool.name in self._tools.keys():
            print(f'警告：工具{tool.name}已存在，将被覆盖')
        self._tools[tool.name]=tool
        print(f'工具{tool.name}已注册')

    def register_function(self,name:str,description:str,func:Callable[[str],str]): #第二种方式注册工具，传入函数名、描述和函数对象
        if name in self._functions:
            print(f'警告：工具{name}已存在，将被覆盖')

        self._functions[name]={
            'description':description,
            'func':func,

        }
        print(f'工具{name}已注册')


    def get_tools_description(self)->str: #获取工具列表的描述信息，返回一个字符串，每个工具占一行，格式为"- 工具名:工具描述"
        description=[]

        for tool in self._tools.values():
            description.append(f'- {tool.name}:{tool.description}')

        for name ,info in self._functions.items():
            description.append(f"- {name}:{info['description']}")

        return '\n'.join(description) if description else '暂无可用工具'
