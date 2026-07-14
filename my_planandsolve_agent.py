from hello_agents import HelloAgentsLLM
from Config import Config
from Message import Message
import ast



from typing import Optional,List,Dict

# 默认规划器提示词模板
DEFAULT_PROMPT = {"planner":"""
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
""",
"executor":"""
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""}


class PlanAndSolveAgent:
    def __init__(self,
                 name:str,
                 llm:HelloAgentsLLM,
                 system_prompt:Optional[str]=None,
                 config:Optional[Config]=None,
                 custom_prompts:Optional[Dict[str,str]]=None,
                 ):
        self.name=name
        self.llm=llm
        self.system_prompt=system_prompt or '你是一个有用的AI助手，能够回答用户的问题并提供帮助。'
        self.config=config or Config()
        self.prompt_template=custom_prompts if custom_prompts else DEFAULT_PROMPT
        self._history:List[Message]=[]
        print(f'{name}初始化完成')
    def planner(self,question:str,**kwargs)->List:
        """
        规划器方法，接收任务描述，返回一个包含子任务的列表。
        """
        prompt=self.prompt_template['planner'].format(question=question)
        response=self.llm.invoke([{'role':'system','content':prompt}],**kwargs)
        try:
            plan_str = response.split("```python")[1].split("```")[0].strip()

            plan = ast.literal_eval(plan_str)  # 这里ast.literal_eval的作用是将字符串解析为Python对象，确保安全性，返回的数据类型是
            return plan if isinstance(plan, list) else []
        except Exception as e:
            print(f'解析计划失败: {e}')
            return []

    def executor(self,question:str,plan:List[str],**kwargs)->str:
        history=[]
        for i,step in enumerate(plan,1):
            history_str='\n'.join(history)
            print(f'\n-> 正在执行步骤 {i}/{len(plan)}: {step}')
            prompt=self.prompt_template['executor'].format(question=question,plan=plan,history=history_str,current_step=step)
            response=self.llm.invoke([{'role':'system','content':prompt}],**kwargs)
            history.append(f'步骤{i}: {step}\n结果: {response}')
            print(f'步骤{i} 已完成，结果为：{response}')
        final_answer=response
        return final_answer

    def run(self,input_text:str,**kwargs)->str:
        print(f'{self.name}正在处理：{input_text}')
        if self._history:
            history_context='\n'.join([f"[{msg.role}]: {msg.content}" for msg in self._history])
            input_text=f'对话历史:\n{history_context}\n\n当前问题: {input_text}'
        plan=self.planner(input_text,**kwargs)
        if not plan:
            print('未能生成有效的计划，无法继续执行。')
            return "未能生成有效的计划，无法继续执行。"
        final_answer=self.executor(input_text,plan,**kwargs)
        print(f'最终答案为：{final_answer}')
        self.add_history(Message(input_text,'user'))
        self.add_history(Message(final_answer,'assistant'))
        return final_answer

    def add_history(self,message:Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history.copy()
