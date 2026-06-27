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

    def plan(self,question:str)->list: #构建计划生成器，用户输入问题然后调用llm生成计划，返回计划列表

        prompt=PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages=[{'role':'user','content':prompt}]
        print('---正在生成计划---')
        response=self.llm_client.think(messages)

        print(f'计划已生成:\n{response}')


        #解析计划列表
        try:
            plan_str=response.split("```python")[1].split("```")[0].strip()

            plan=ast.literal_eval(plan_str) #这里ast.literal_eval的作用是将字符串解析为Python对象，确保安全性，返回的数据类型是
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

class Executor(): #定义执行器
    def __init__(self,llm_client:Myagent):
        self.llm_client=llm_client

    def execute(self,question:str,plan:list)->str:
        history='' #将每一步的结果写入history中

        print('\n--- 正在执行计划 ---')

        for i ,step in enumerate(plan): #python中for循环不是一个新的作用域，其中的变量保留最终的值，可以在循环外使用
            print(f'\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}')
            prompt=EXECUTOR_PROMPT_TEMPLATE.format(question=question,plan=plan,history=history,current_step=step)
            messages=[{'role':'user','content':prompt}]
            response=self.llm_client.think(messages)

            history+=f'步骤{i+1}: {step}\n结果: {response}\n\n'
            print(f'步骤{i+1} 已完成，结果为：{response}')

        final_answer=response
        return final_answer



class PlanAndSolveAgent:
    def __init__(self,llm_client:Myagent):
        self.llm_client=llm_client
        self.planner=Planner(self.llm_client)
        self.executor=Executor(self.llm_client)

    def run(self,question:str):
        print(f'\n--- 开始处理问题 ---\n问题：{question}')
        plan=self.planner.plan(question)

        if not plan:
            print('错误：未能生成有效的计划！')
            return None

        final_answer=self.executor.execute(question,plan)

        print(f'\n--- 任务完成 ---\n最终答案：{final_answer}')



if __name__ == '__main__':
    llm_client=Myagent()
    plan_and_solve_agent=PlanAndSolveAgent(llm_client=llm_client)

    plan_and_solve_agent.run('计算定积分 I = ∫₀^π e^x * sin x dx （从 0 到 π 的 e^x sin x 的积分）')

