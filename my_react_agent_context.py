from hello_agents import HelloAgentsLLM, ToolRegistry

from contextbase import ContextConfig, ContextPacket
from ContextBuilder import ContextBuilder
from my_react_agent import MyReActAgent
from typing import Optional, List
from Message import Message
from Config import Config
from datetime import datetime
from memory_src import MemoryTool
import json

# 知识点：ReAct 的 Action 入参是字符串；要用 hello_agents.ToolRegistry.execute_tool(name, str)。
# 本地 Tool.ToolRegistry 期望 Dict，和 search[查询] 对不上。


def _memory_fn(mem_tool: MemoryTool):
    def memory_fn(raw: str) -> str:
        raw = (raw or "").strip()
        try:
            params = json.loads(raw) if raw.startswith("{") else {"action": "search", "query": raw}
            return mem_tool.run(params)
        except Exception as e:
            return f"Memory error: {e}"
    return memory_fn


class ContextAwareAgent(MyReActAgent):
    def __init__(self, name: str, llm: HelloAgentsLLM, tool_registry: ToolRegistry, max_steps: int = 5,system_prompt:Optional[str]=None,config:Optional[Config]=None,custom_prompt:Optional[str]=None,
                context_config: Optional[ContextConfig] = None,custom_context: Optional[str] = None,
                memory_tool: Optional[MemoryTool] = None, register_memory_tool: bool = True):
        super().__init__(name=name, llm=llm, tool_registry=tool_registry, max_steps=max_steps, system_prompt=system_prompt, config=config, custom_prompt=custom_prompt)

        self.context_config= context_config or ContextConfig()
        self.custom_context=custom_context
        # 双轨记忆：被动 build 与主动 memory 工具共用同一 MemoryTool
        self.memory_tool = memory_tool or MemoryTool()
        self.context_builder = ContextBuilder(llm, config=self.context_config, memory_tool=self.memory_tool)
        if register_memory_tool:
            names = set(getattr(tool_registry, "_tools", {}) or {}) | set(getattr(tool_registry, "_functions", {}) or {})
            if "memory" not in names:
                tool_registry.register_function(
                    "memory",
                    '记忆工具。参数必须是 JSON，例如 '
                    '{"action":"add","content":"用户住武汉","memory_type":"episodic","importance":0.8} '
                    '或 {"action":"search","query":"住哪里","limit":3}',
                    _memory_fn(self.memory_tool),
                )
        print(f'{name}初始化完成，启用上下文感知功能')

    def _build_custom_context(self,custom_context:str)->List[ContextPacket]:
        #将自定义上下文字符串转换为ContextPacket列表
        context_packets=[]
        if custom_context:
            context_packets.append(ContextPacket(timestamp=datetime.now(),content=custom_context,token_count=self.context_builder._count_tokens(custom_context),relevance_score=0.5,metadata={"type":"custom_context","source":"custom_context"}))
        return context_packets

    def run(self,input_text:str,**kwargs)->str:
        print(f'{self.name}正在处理：{input_text}')
        self.current_history=[]
        current_step=0

        custom_packets=self._build_custom_context(self.custom_context)
        background=self.context_builder.build(
            input_text,self._history,self.system_prompt,custom_packets,
            include_task_sections=False,
        )

        while current_step<=self.max_steps:
            current_step+=1
            print(f'\n--- 第{current_step}轮 ---\n')

            tool_desc=self.tool_registry.get_tools_description()
            history_str='\n'.join(self.current_history)
            prompt=self.prompt_template.format(
                tools=tool_desc,
                question=input_text,
                history=history_str
            )

            messages=[{'role':'system','content':background},{'role':'user','content':prompt}]
            response=self.llm.invoke(messages,**kwargs) or ''
            print(f'LLM响应：\n{response}\n')

            thought,action=self._parse_output(response)
            print(f'Thought:{thought}')
            print(f'Action:{action}')

            if action and action.startswith('Finish'):
                final_answer=self._parse_action_input(action)
                self.add_message(Message(input_text,'user'))
                self.add_message(Message(final_answer,'assistant'))
                print(f'最终答案：{final_answer}')
                return final_answer

            if action:
                tool_name,tool_input=self._parse_action(action)
                if not tool_name:
                    print(f'无法解析工具调用：{action}')
                    self.current_history.append(f'Action:{action}')
                    self.current_history.append('Observation:Action格式错误，请使用 工具名[参数]')
                    continue

                try:
                    observation=self.tool_registry.execute_tool(tool_name,tool_input)
                except Exception as e:
                    observation=f'工具执行失败:{str(e)}'
                print(f'Observation:{observation}')
                self.current_history.append(f'Action:{action}')
                self.current_history.append(f'Observation:{observation}')
            else:
                print('本步未解析到 Action（LLM 可能没按 Thought/Action 格式输出）')

        final_answer='达到最大步骤数，未能得到最终答案。'
        self.add_message(Message(input_text,'user'))
        self.add_message(Message(final_answer,'assistant'))
        print(f'最终答案：{final_answer}')
        return final_answer
