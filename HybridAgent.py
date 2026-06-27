from ReAct import Myagent,ToolExecutor,ReActAgent
from PlanAndSolve import Planner, EXECUTOR_PROMPT_TEMPLATE
from dotenv import load_dotenv
import ast


load_dotenv()
'''
实现PlanAndSolveAgent和，ActAgent的组合使用，创建一个混合智能体HybridAgent。
'''


myPlanner_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列，其中某些任务可能需要调用已有的工具，我会将已有的工具列表给你参考。

问题: {question}
工具列表:
{tool_desc}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
[
  {{"tool":"工具名称", "input":"工具输入", "step":"步骤描述"}},
  {{"step":"纯推理步骤描述"}}
]
```

注意：需要调用工具时，必须包含 tool 和 input 字段；不需要调用工具时，只写 step 字段。 """


EXECUTOR_PROMPT_TEMPLATE="""
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
原始问题: {question}
当前步骤描述: {current_step}
历史结果: {history}
Plan提示（如需调工具）: {tool_hint}
"""

REFLECTION_TEMPLATE="""
你是评审专家。请判断以下答案是否满意。

原始问题: {question}
执行历史: {history_str}
最终答案: {final_answer}

如果答案正确完整，仅回复「无需改进」。
如果答案有误或不完整，回复「需要改进：」并简要说明原因。
"""

class myPlanner(Planner):
    def __init__(self,llm_client:Myagent,tool_executor:ToolExecutor):
        super().__init__(llm_client)
        self.tool_executor=tool_executor
        self.tool_desc=tool_executor.getAvailableTools()

    def plan(self,question:str):
        prompt=myPlanner_PROMPT_TEMPLATE.format(question=question,tool_desc=self.tool_desc)
        messages=[{'role':'user','content':prompt}]
        print('--- 正在生成计划 ---')
        response=self.llm_client.think(messages)

        print(f'计划已生成：\n{response}')

        # 解析计划列表
        try:
            plan_str = response.split("```python")[1].split("```")[0].strip()

            plan = ast.literal_eval(plan_str)  # 这里ast.literal_eval的作用是将字符串解析为Python对象，确保安全性，返回的数据类型是
            return plan if isinstance(plan, list) else []
        except Exception as e:
            print(f'解析计划失败: {e}')
            return []




class HybridAgent:
    def __init__(self,llm_client:Myagent,max_depth:int=3):
        self.llm_client = llm_client
        self.tool_executor = ToolExecutor()
        self.planner = myPlanner(self.llm_client,self.tool_executor)
        self.ReActAgent = ReActAgent(self.llm_client,self.tool_executor)
        self.max_depth = max_depth


    def run(self,question:str,depth:int=1):
        if depth>self.max_depth:
            print('达到递归最大深度，停止运行。')
            return
        print(f'\n--- 开始处理问题 ---\n问题：{question}，第{depth}次尝试')
        plan=self.planner.plan(question)

        if not plan:
            print('错误，未能生成有效的计划！')
            return None
        history=[]

        for i,iteration in enumerate(plan):
            print(f'\n-> 正在执行步骤 {i+1}/{len(plan)}: {iteration["step"]}')
            prompt=EXECUTOR_PROMPT_TEMPLATE.format(question=question,plan=plan,history=history,current_step=iteration.get('step',''),tools=self.tool_executor.getAvailableTools(),tool_hint=f"建议工具：{iteration.get('tool','')},输入：{iteration.get('input','')}" if 'tool' in iteration else "")

            messages=[{'role':'user','content':prompt}]
            if 'tool' not in iteration:
                response=self.llm_client.think(messages)
                history.append(response)
                print(f'无工具调用，步骤{i+1} 已完成，结果为：{response}')
            else:
                history_str='\n'.join(history)
                react_question = (
                    f"原始问题: {question}\n"
                    f"之前步骤的结果: {history_str}\n"
                    f"当前任务: {iteration['step']}\n"
                    f"请使用 {iteration['tool']}[{iteration['input']}] 获取所需信息。"
                )
                response=self.ReActAgent.run(react_question)
                history.append(response)
                print(f'有工具调用，步骤{i+1} 已完成，结果为：{response}')

        final_answer=history[-1]
        history_str = '\n'.join(history)

        prompt=REFLECTION_TEMPLATE.format(question=question,final_answer=final_answer,history_str=history_str)
        messages=[{'role':'user','content':prompt}]
        response=self.llm_client.think(messages)
        if '无需改进' in response:
            print(f'最终答案：{final_answer}')
            return

        self.run(question,depth+1)
        print(f'达到递归最大深度，最终答案为：{final_answer}')



if __name__ == '__main__':
    myagent = Myagent()
    agent=HybridAgent(myagent)
    agent.run("请帮我规划一下去武汉的旅游路线。")




