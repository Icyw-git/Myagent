from ReAct import ToolExecutor,ReActAgent,search
from PlanAndSolve import Planner
from dotenv import load_dotenv
import ast
from llm_client import Myagent



load_dotenv()
'''
实现PlanAndSolveAgent和，ActAgent的组合使用，创建一个混合智能体HybridAgent。
'''


# ==================== [错误记录 #1] str.format() 花括号转义 ====================
# 知识点：Python str.format() 中 { 和 } 是占位符的起始/结束符。
# 如果模板里需要输出字面量的花括号（比如 JSON 示例），必须双写 {{ 和 }} 来转义。
# 错误写法：{"tool": "名称"}  →  KeyError（format() 把 "tool" 当成占位符 key 去找）
# 正确写法：{{"tool": "名称"}}  →  输出 {"tool": "名称"}
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

class myPlanner: #重写planner,使得planner可以使用工具列表
    def __init__(self,llm_client:Myagent,tool_executor:ToolExecutor):
        self.llm_client=llm_client
        self.tool_executor=tool_executor
        self.tool_desc=self.tool_executor.getAvailableTools() #获取工具列表描述

    def plan(self,question:str):
        prompt=myPlanner_PROMPT_TEMPLATE.format(question=question,tool_desc=self.tool_desc)
        messages=[{'role':'user','content':prompt}]
        print('--- 正在生成计划 ---')
        response=self.llm_client.think(messages) or ''

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
    # ==================== [错误记录 #2] 对象初始化完整性（依赖注入） ====================
    # 知识点：对象在构造后必须处于"可用"状态。
    # 如果 Planner/Executor 依赖工具列表，构造函数里就必须注册工具。
    # 错误：先 new ToolExecutor() 传给 Planner，但从未 register_tool() → 工具列表为空，LLM 编造不存在的工具名。
    # 修复：构造函数里调用 _register_tools()，或用 _register_default_tools 抽成独立方法便于扩展。
    def __init__(self,llm_client:Myagent,max_depth:int=3):
        self.llm_client = llm_client
        self.tool_executor = ToolExecutor() #先初始化toolexecutor 之后再统一进行工具注册
        self._register_tools()
        self.planner = myPlanner(self.llm_client,self.tool_executor)
        self.ReActAgent = ReActAgent(self.llm_client,self.tool_executor)
        self.max_depth = max_depth #最大递归深度

    def _register_tools(self):
        self.tool_executor.register_tool(
            'search',
            '一个搜索工具，可以根据用户的查询返回相关的搜索结果。',
            search,
        )


    def run(self,question:str,depth:int=1,current_best:str=None):
        if depth>self.max_depth:
            print('达到递归最大深度，停止运行。')
            return current_best if current_best else None #若达到最大递归深度则返回当前最佳答案
        print(f'\n--- 开始处理问题 ---\n问题：{question}\n--- 第{depth}次尝试 ---\n')
        plan=self.planner.plan(question)

        if not plan:
            print('错误，未能生成有效的计划！')
            return None
        history=[]

        for i,iteration in enumerate(plan):
            print(f'\n-> 正在执行步骤 {i+1}/{len(plan)}: {iteration["step"]}')
            history_str='\n'.join(history) if history else ''
            prompt=EXECUTOR_PROMPT_TEMPLATE.format(question=question,history=history,current_step=iteration.get('step',''),tools=self.tool_executor.getAvailableTools(),tool_hint=f"建议工具：{iteration.get('tool','')},输入：{iteration.get('input','')}" if 'tool' in iteration else "")

            messages=[{'role':'user','content':prompt}]
            # ==================== [错误记录 #3] str.join() 要求所有元素为 str ====================
            # 知识点：'\n'.join(list) 要求 list 的每个元素都是 str，否则抛出 TypeError。
            # 如果 llm_client.think() 或 ReActAgent.run() 返回 None（LLM 异常 / 跑满轮次未收敛），
            # history.append(None) 后下一轮循环 join 时就会报：
            # TypeError: sequence item 0: expected str instance, NoneType found
            # 修复：所有外部返回值后面加 or '' 兜底。

            #None是python中的一种数据类型，None是 Python 的一个特殊常量，表示“没有值”、“空”或“缺失”。
            if 'tool' not in iteration:
                response=self.llm_client.think(messages) or ''  # or ''：防止 None 进入 history
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
                response=self.ReActAgent.run(react_question) or ''  # or ''：防止 ReAct 跑满轮次返回 None
                history.append(response)
                print(f'有工具调用，步骤{i+1} 已完成，结果为：{response}')

        # ==================== [错误记录 #4] str.format() key 必须与模板占位符一一对应 ====================
        # 知识点：format() 传入的 key 必须在模板字符串中有对应的 {key} 占位符。
        # 多传（模板里没有 {plan} 但你传了 plan=plan）→ 某些版本报 KeyError。
        # 少传 → KeyError: '某个占位符名'。
        # 容器对象直接传给 format 时会被转成 __str__ 表示：str([]) = "[]"，LLM 看到"历史结果: []"无意义。
        # 应先 history_str = '\n'.join(history) 再传入。
        final_answer=history[-1]
        history_str = '\n'.join(history)

        prompt=REFLECTION_TEMPLATE.format(question=question,final_answer=final_answer,history_str=history_str)
        messages=[{'role':'user','content':prompt}]
        response=self.llm_client.think(messages) or ''
        if '无需改进' in response:
            print(f'最终答案：{final_answer}')
            return final_answer #无需改进则直接返回答案

        print(f'--- 结果未达到要求，需要改进，重新规划。---')
        # ==================== [错误记录 #5] 递归函数必须 return 递归调用的结果 ====================
        # 知识点：递归函数里写 self.run(...) 而不写 return，递归的返回值不会传递回最外层调用方。
        # 错误写法：self.run(question, depth+1, final_answer) → 结果被丢弃，函数默认返回 None
        # 正确写法：return self.run(question, depth+1, final_answer) → 递归结果逐层上传
        return self.run(question,depth+1,final_answer) #递归至下一层，planner重新规划




if __name__ == '__main__':
    myagent = Myagent()
    agent=HybridAgent(myagent)
    agent.run("请帮我规划一下去武汉的旅游路线。")




