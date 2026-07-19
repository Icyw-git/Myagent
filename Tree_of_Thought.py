"""
贪心版
"""

from typing import Optional
from Agent import Agent
from hello_agents import HelloAgentsLLM
from Config import Config
from Message import Message
import ast
import json
import re



# ==================== [错误记录 #1] 生成 prompt 未约束「列表元素彼此独立」 ====================
# 知识点：ToT 的 generate 必须产出恰好 branches 条独立候选；若 prompt 只写
# 「生成3个答案」并示例 ["答案1","答案2"]，LLM 常把三条方案写进同一个字符串，
# 结果 list 只有 1 个元素 → 只打一次分 → 最终返回答案1+答案2+答案3 整坨文本。
# 错误写法：不强调「每个列表元素只写一条方案」→ 解析出 len(responses)==1
# 正确写法：明确要求恰好 N 个元素，且禁止把多个方案写在同一个字符串里。
GENERATE_THOUGHTS_PROMPT= """
你是具有深度思考能力的AI助手，请根据用户的问题生成多个可能的答案，每个答案都要进行详细的推理和分析，并给出最终的结论。
用户的问题是：{input_text}
上一步的思考是：{previous_thoughts}
请生成{branches}个可能的答案，每个答案都要进行详细的推理和分析，并给出最终的结论。
重要：必须返回恰好{branches}个元素的Python列表，每个列表元素只写一条完整方案，不要把多个方案写在同一个字符串里。
以```python

[

    "方案1的完整内容",

    "方案2的完整内容",

    "方案3的完整内容",

]

```
的形式返回。

"""


# ==================== [错误记录 #2] str.format() 花括号转义 ====================
# 知识点：模板里要输出字面量 JSON 花括号时，必须写成 {{ 和 }}，否则 format()
# 会把 "score" 当成占位符 → KeyError。
# 错误写法：{"score": 8.5}  → KeyError
# 正确写法：{{"score": 8.5}} → 输出 {"score": 8.5}
EVALUATE_THOUGHT_PROMPT="""
请对以下回答的质量打分。
# 原始任务:
{task}
# 当前回答:
{content}
请从准确性、完整性、逻辑性三个维度综合评估，给出 0 到 10 的分数。
必须严格按以下 JSON 格式输出，不要输出其他内容:
{{"score": 8.5, "reason": "简要说明打分理由"}}

"""



class TreeofThought(Agent):
    def __init__(self,
                 name:str,
                 llm:HelloAgentsLLM,
                 config:Optional[Config]=None,
                 system_prompt:Optional[str]=None,

                 ):

        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        print(f'{name}初始化完成')



    def run(self,input_text:str,branches:int=3,max_depth:int=3,quality_threshold:float=8.0,**kwargs):
        print(f'{self.name}正在处理：{input_text}')
        messages=[]
        messages.append({"role":"system","content":self.system_prompt})
        # ==================== [错误记录 #3] Message 对象不能当下标字典访问 ====================
        # 知识点：self._history 里存的是 Message 对象，属性用 .role / .content，
        # 不是 dict 的 msg["role"]。
        # 错误写法：msg["role"] → TypeError: 'Message' object is not subscriptable
        # 正确写法：msg.role、msg.content
        for msg in self._history:
            messages.append({"role":msg.role,"content":msg.content})


        messages.append({'role':'user','content':input_text})
        # ==================== [错误记录 #4] 跑满深度时应用元组记录全局最佳 ====================
        # 知识点：贪心 ToT 每层会选本层 best；最后一层分数不一定是全程最高。
        # 错误写法：循环结束直接 return 最后一层 best_evaluation[0]
        #          → 第1层9分、第2层7分时会错误返回7分答案
        # 正确写法：用 best_overall=(answer, score) 每层比较更新，结束返回全局最佳
        #          （与 MyReflectionAgentplus 的 best_response 元组同思路）
        best_overall = ("", 0.0)



        for depth in range(max_depth):
            print(f'--- 第{depth + 1}层 ---')
            responses=self._generate_thoughts(input_text,messages,branches,**kwargs)
            if not responses:
                print('警告：本层未生成有效候选。')
                continue



            # ==================== [错误记录 #5] tuple 不可变，不能 item[1]=score ====================
            # 知识点：tuple 创建后不能改元素；evaluations 若写成 [(r, 0.0), ...]，
            # 再写 item[1]=score 会 TypeError。
            # 错误写法：evaluations=[(response,0.0) for ...]；item[1]=score
            # 正确写法：用 list [[response,0.0], ...]；或列表推导直接算好分数
            evaluations=[[response,0.0] for response in responses]
            for i, item in enumerate(evaluations, 1):
                item[1]=self._evaluate_thought(input_text,item[0],**kwargs)
                print(f'候选{i}得分：{item[1]:.2f}')



            best_evaluation=max(evaluations,key=lambda x:x[1])
            if best_evaluation[1] > best_overall[1]:
                best_overall = (best_evaluation[0], best_evaluation[1])
            print(f'当前最佳答案: {best_overall[0]}\n得分: {best_overall[1]:.2f}')



            messages.append({'role':'assistant','content':best_evaluation[0]})



            if best_evaluation[1]>=quality_threshold:
                print(f'得分达到阈值,最终答案为：{best_evaluation[0]}')
                # ==================== [错误记录 #6] Message 构造参数顺序 ====================
                # 知识点：Message.__init__(self, content:str, role:str, ...)
                # 位置参数顺序是 content 在前、role 在后；或用关键字时也要对齐。
                # 错误写法：Message(role='user', content=input_text)
                #          → 与自定义 __init__ 位置参数习惯不一致，易传反
                # 正确写法：Message(input_text, 'user')
                self.add_history(Message(input_text,'user'))
                self.add_history(Message(best_evaluation[0],'assistant'))
                return best_evaluation[0]



        print(f'已达到最大深度,返回全局最佳：{best_overall[0]}')
        self.add_history(Message(input_text,'user'))
        self.add_history(Message(best_overall[0],'assistant'))
        return best_overall[0]



    def _generate_thoughts(self,input_text:str,messages:list[dict],branches:int,**kwargs):
        print(f'正在生成{branches}条候选思路...')
        prompt=GENERATE_THOUGHTS_PROMPT.format(input_text=input_text,previous_thoughts=messages[-1]['content'],branches=branches)
        # ==================== [错误记录 #7] invoke 参数类型与返回值兜底 ====================
        # 知识点：HelloAgentsLLM.invoke 需要 List[Dict]（messages），不能直接传 str / {}。
        # 且 invoke 失败可能返回 None，后续 literal_eval(None) 会崩。
        # 错误写法：self.llm.invoke(prompt) 或 self.llm.invoke({})
        # 正确写法：self.llm.invoke([{'role':'user','content':prompt}]) or ''
        responses=self.llm.invoke([{'role':'user','content':prompt}],**kwargs) or ''

        if '```python' in responses:
            responses=responses.split('```python')[1].split('```')[0].strip()
        try:
            responses=ast.literal_eval(responses)
            if not isinstance(responses, list):
                print('警告：返回结果不是列表。')
                return []
            # ==================== [错误记录 #8] 合并候选的兜底拆分 ====================
            # 知识点：即使 prompt 已约束，LLM 仍可能把「答案1/2/3」写进同一个字符串。
            # 此时 len(responses)==1 且文本含「答案2：」，可按「答案N：」拆开再分别打分。
            # 这是补救手段，不能替代错误记录 #1 的 prompt 约束。
            if len(responses) == 1 and re.search(r'答案2[：:]', str(responses[0])):
                parts = re.split(r'答案\d+[：:]', str(responses[0]))
                responses = [p.strip() for p in parts if p.strip()]

            print(f'共解析出 {len(responses)} 条候选')
            return responses

        except (SyntaxError, ValueError) as e:
            print(f'解析候选失败：{e}')
            return []



    def _evaluate_thought(self,input_text:str,thought:str,**kwargs):
        prompt=EVALUATE_THOUGHT_PROMPT.format(task=input_text,content=thought)
        response=self.llm.invoke([{'role':'user','content':prompt}],**kwargs) or ''
        # ==================== [错误记录 #9] JSON 评分应用 json.loads，不能整段 literal_eval ====================
        # 知识点：evaluate prompt 要求返回 {"score": ...}；LLM 常带前后说明文字。
        # 错误写法：ast.literal_eval(response) 或 json.loads(整段原文)
        #          → 解析失败 / 格式不匹配
        # 正确写法：re.search 抽出 \{.*\} 再用 json.loads；失败则返回 0.0
        #          （与 MyReflectionAgentplus._parse_score 同思路）
        match=re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                score_data = json.loads(match.group())
                return float(score_data['score'])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                print('警告：分数解析失败，返回 0.0')

        return 0.0





if __name__ =='__main__':
    agent=TreeofThought('TOT',HelloAgentsLLM())
    agent.run('请帮我规划一下去武汉的旅游路线。',branches=3,max_depth=3,quality_threshold=8.0)


