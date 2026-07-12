from typing import Optional, Dict, Any
import  re
from hello_agents import SimpleAgent,HelloAgentsLLM,ToolRegistry
import inspect
from dotenv import load_dotenv


from Message import Message
from Config import Config


load_dotenv()

class MySimpleAgent(SimpleAgent):
    def __init__(self,
                 name:str,
                 llm: HelloAgentsLLM,
                 system_prompt:Optional[str]=None,
                 tool_registry:Optional[ToolRegistry]=None,
                 enable_tool_calling:Optional[bool]=False,
                 config:Optional[Config]=None


                 ):
        super().__init__(name=name,system_prompt=system_prompt,llm=llm,config=config)
        self.enable_tool_calling=enable_tool_calling and tool_registry is not None #如果启用工具调用但没有提供工具注册表，则禁用工具调用
        self.tool_registry=tool_registry


    def run(self,input_text:str,max_tool_iteration:int,**kwargs) ->str:
        print(f'{self.name}正在处理：{input_text}')

        messages=[]
        enhanced_system_prompt=self._enhance_system_prompt()
        messages.append({'role':'system','content':enhanced_system_prompt})

        for msg in self._history:
            messages.append({'role':msg.role,'content':msg.content})

        messages.append({'role':'user','content':input_text})

        if not self.enable_tool_calling:
            response=self.llm.invoke(messages,**kwargs)
            self.add_message(Message(input_text,'user'))
            self.add_message(Message(response,'assistant'))
            print(f'{self.name}响应成功')
            return response

        return self._run_with_tools(messages,input_text,max_tool_iteration,**kwargs)

    def _enhance_system_prompt(self) ->str:

        base_prompt=self.system_prompt or '你是一个有用的AI助手。'
        if not self.enable_tool_calling or self.tool_registry:
            return base_prompt

        tool_desc=self.tool_registry.get_tools_description()
        if not tool_desc or tool_desc=='暂无可用工具':
            return base_prompt

        tool_prompt_template="""
        --- 可用工具 ---
        你可以使用以下工具来回答问题：
        {tool_desc}
        
        --- 工具使用方法 ---
        当需要调用工具是，请使用以下格式：
        '[TOOL_CALL:{tool_name}:{parameters}]'
        
        例如:'[TOOL_CALL:search:Python编程]' 或 '[TOOL_CALL:memory:recall=用户信息]'
        
        工具调用结果会自动加入到对话中，你可以根据结果更好的回答问题。
        
        
        
        
        
        
        
        
        """

        tool_prompt=tool_prompt_template.format(tool_desc=tool_desc)
        return base_prompt+tool_prompt

    def _run_with_tools(self,messages:list,input_text:str,max_tool_iteration:int,**kwargs)->str:
        current_iteration=0
        final_answer=''
        while current_iteration <=max_tool_iteration:
            current_iteration+=1
            response=self.llm.invoke(messages,**kwargs)
            tools_list=self._parse_tool_calls(response)

            if tools_list:
                print(f'检测到{len(tools_list)}个工具调用')

                tool_results=[]
                clear_response=response

                for tool in tools_list:
                    result=self._execute_tool_call(tool['tool_name'],tool['parameters'])
                    tool_results.append(result)
                    clear_response=clear_response.replace(tool['original'],'')

                messages.append({'role':'assistant','content':clear_response})

                result_texts='\n\n'.join(tool_results)
                messages.append({'role':'user','content':result_texts})

                continue

            final_answer=response
            break

        if current_iteration> max_tool_iteration and not final_answer:
            response=self.llm.invoke(messages,**kwargs)
            final_answer=response

        self.add_message(Message(input_text,'user'))
        self.add_message(Message(response,'assistant'))

        return response

    def _parse_tool_calls(self,texts:str):
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, texts)

        tool_calls=[]

        for tool_name,parameters in matches:
            tool_calls.append(
                {
                    'tool_name':tool_name.strip(),
                    'parameters':parameters.strip(),

                    'original':f'[TOOL_CALL:{tool_name}:{parameters}]'



                }
            )

        return tool_calls











