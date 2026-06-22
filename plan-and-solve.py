from ReAct import Myagent
import ast


#为plan-and-solve agent编写的提示词

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner():
    def __init__(self,llm_client:Myagent):
        self.llm_client=llm_client

    def plan(self,question:str)->list:

        prompt=PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages=[{'role':'user','content':prompt}]
        print('---正在生成计划---')
        response=self.llm_client.think(messages)

        print(f'计划已生成:\n{response}')


        #解析计划列表
        try:
            plan_str=response.split("```python")[1].split("```")[0].strip()

            plan=ast.literal_eval(plan_str)
            return plan if isinstance(plan,list) else []
        except Exception as e:
            print(f'解析计划失败: {e}')
            return []


#执行器的提示词
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor():
    def __init__(self,llm_client:Myagent):
        self.llm_client=llm_client

    def execute(self,question:str,plan:list)->str:
        history=''

        for i ,step in enumerate(plan):
            prompt=EXECUTOR_PROMPT_TEMPLATE.format(question=question,plan=plan,history=history,current_step=step)
            messages=[{'role':'user','content':prompt}]
            response=self.llm_client.think(messages)

            history+=f'步骤{i+1}: {step}\n结果: {response}\n\n'

        final_answer=response
        return final_answer
