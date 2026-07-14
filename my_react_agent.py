from typing import Optional,List
import re

from hello_agents import ReActAgent,ToolRegistry,HelloAgentsLLM
from Message import Message
from Config import Config

# ==================== [错误记录 #1] 提示词模板为空 ====================
# 知识点：ReAct Agent 必须通过提示词告诉 LLM 输出格式（Thought/Action/Finish），
# 否则 LLM 不会按格式输出，_parse_output 解析必然失败。
# 错误写法：模板为空字符串 → LLM 收到空指令，自由发挥 → 正则匹配不到 Thought/Action
# 正确写法：填入类似 HybridAgent 中 EXECUTOR_PROMPT_TEMPLATE 的格式规约，
#          明确要求 LLM 输出 Thought / Action / Finish 三段格式。
MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""


class MyReActAgent(ReActAgent):
    def __init__(self,
                 name:str,
                 llm:HelloAgentsLLM,
                 tool_registry:ToolRegistry,
                 system_prompt:Optional[str]=None,
                 config:Optional[Config]=None,
                 max_steps:int=5,
                 custom_prompt:Optional[str]=None, #用户传入的自定义prompt模板
                 ):
        super().__init__(name=name,llm=llm,config=config,system_prompt=system_prompt)
        self.tool_registry=tool_registry
        self.max_steps=max_steps
        self.current_history:List[str]=[]
        self.prompt_template=custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f'{name}初始化完成')

    def run(self,input_text:str,**kwargs)->str: #agent循环类似，每一步都调用llm生成响应，解析响应中的动作，执行工具调用，并将结果添加到历史记录中，直到达到最大步数或完成任务
        self.current_history=[]
        current_step=0

        print(f'{self.name}开始处理问题：{input_text}')
        while current_step<self.max_steps:
            current_step+=1
            print(f'\n--- 第{current_step}步 ---')

            tool_desc=self.tool_registry.get_tools_description()
            history_str='\n'.join(self.current_history)
            prompt=self.prompt_template.format(
                tools=tool_desc,
                question=input_text,
                history=history_str
            )

            messages=[{'role':'user','content':prompt}]
            response_text=self.llm.invoke(messages,**kwargs)

            # ==================== [错误记录 #2] 缺少方法定义与 import ====================
            # 知识点：类中调用的方法必须在类体内定义，使用的标准库模块必须在文件头 import。
            # 错误写法：_parse_output、_parse_action 方法未定义，re 模块未 import
            #          → 运行时抛出 AttributeError
            # 正确写法：在类体中定义 @staticmethod 的 _parse_output / _parse_action 方法，
            #          并在文件头部 import re。
            thought,action=self._parse_output(response_text)

            # ==================== [错误记录 #3] 方法名拼写错误 ====================
            # 知识点：Python 字符串方法名正确拼写为 startswith（含 s），而非 startwith。
            # 错误写法：action.startwith('Finish') → AttributeError
            # 正确写法：action.startswith('Finish')
            if action and action.startwith('Finish'):
                # ==================== [错误记录 #4] 调用了不存在的方法 ====================
                # 知识点：调用方法前必须确认该方法已在类中定义。
                # 错误写法：_parse_action_input 方法未定义 → AttributeError
                # 正确写法：用 re.match(r'Finish\[(.*)\]', action, re.DOTALL) 直接提取，
                #          或定义一个 _parse_action_input 方法。
                final_answer=self._parse_action_input(action)
                self.add_message(Message(input_text,'user'))
                self.add_message((Message(final_answer,'assistant')))
                return final_answer


            if action:
                tool_name,tool_input=self._parse_action(action)
                # ==================== [错误记录 #5] 工具调用缺少异常保护 ====================
                # 知识点：外部调用（工具执行、网络请求等）必须用 try/except 包裹，
                # 否则单次工具失败会导致整个 Agent 崩溃退出。
                # 错误写法：直接调用 execute_tool，信任其永不失败 → 工具异常时 Agent 终止
                # 正确写法：try/except 捕获所有异常，将异常信息转为 observation 追加到 history，
                #          让 LLM 看到错误后可以自行调整策略、选择其他工具或重试。
                observation=self.tool_registry.execute_tool(tool_name,tool_input)
                self.current_history.append(f'Action:{action}')
                self.current_history.append(f'Observation:{observation}')

        final_answer='抱歉，我未能在限定步数内完成该任务。'
        self.add_message(Message(input_text,'user'))
        self.add_message(Message(final_answer,'assistant'))
        return final_answer

    def _parse_output(self,text:str):
        # 使用正则表达式解析 LLM 输出，提取 Thought 和 Action
        thought_match=re.search(r'Thought:(.*)',text,re.DOTALL)
        action_match=re.search(r'Action:(.*)',text,re.DOTALL)

        thought=thought_match.group(1).strip() if thought_match else ''
        action=action_match.group(1).strip() if action_match else ''

        return thought,action

    def _parse_action(self,action:str):
        # 使用正则表达式解析 Action，提取工具名和输入参数
        match=re.match(r'(\w+)\[(.*)\]',action)
        if match:
            tool_name=match.group(1).strip()
            tool_input=match.group(2).strip()
            return tool_name,tool_input
        else:
            return None,None

    def _parse_action_input(self,action:str):
        # 使用正则表达式解析 Finish，提取最终答案
        match=re.match(r'Finish\[(.*)\]',action,re.DOTALL)
        if match:
            final_answer=match.group(1).strip()
            return final_answer
        else:
            return '抱歉，我未能理解最终答案。'


