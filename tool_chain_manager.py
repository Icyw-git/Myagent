from typing import List,Dict,Any,Optional

from numpy.distutils.from_template import find_and_remove_repl_patterns

from Tool import ToolRegistry
from test_reflection_agent import result


class ToolChain:
    def __init__(self,name:str,description:str):
        self.name=name
        self.description=description
        self.steps:List[Dict[str,Any]]=[]

    def add_step(self,tool_name:str,input_template:str,output_key:Optional[str]=None):

        self.steps.append(
            {
                'tool_name':tool_name,
                'input_template':input_template,
                'output_key':output_key or f'step_{len(self.steps)}_result'

            }
        )

    def execute(self,registry:ToolRegistry,initial_input:str,context:Dict[str,Any]=None)->str:
        context=context or {}
        context['input']=initial_input
        print(f'开始执行工具链：{self.name}')


        for i,step in enumerate(self.steps,1):
            tool_name=step['tool_name']
            input_template=step['input_template']
            output_key=step['output_key']

            try:
                tool_input=input_template.format(**context)
            except KeyError as e :
                return f'工具链调用失败：模板变量{e}未找到'

            print(f' 步骤{i}:使用{tool_name}处理{tool_input[:50]}...')

            result=registry.execute_tool(tool_name,tool_input)

            context[output_key]=result
            print(f' 步骤{i}完成，结果长度为{len(result)}字符')

        final_result=context[self.steps[-1]['output_key']]

        print(f'工具链{self.name}执行完成')
        return final_result

class ToolChainManager:
    def __init__(self,registry:ToolRegistry):
        self.registry=registry
        self.chains:Dict[str,ToolChain]={}

    def register_chain(self,chain:ToolChain):
        self.chains[chain.name]=chain
        print(f'工具链{chain.name}已注册')

    def execute_chain(self,name:str,input_data:str,context:Dict[str,Any]=None):
        if not name in self.chains:
            return f'工具链{name}不存在'

        chain=self.chains[name]

        result=chain.execute(self.registry,input_data,context)
        return result

    def list_all_chains(self)->list:
        return list(self.chains.keys())



