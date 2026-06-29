import os 
from dotenv import load_dotenv
from typing import List ,Dict,Any,Optional,Literal
from serpapi import SerpApiClient
from llm_client import Myagent #使用通用的llm_client模块
import re
from pydantic import BaseModel #使用pydantic进行json格式的规定和检查



load_dotenv() #使用load_dotenv()函数加载环境变量，该环境变量作用域为当前进程，且只在当前进程中有效。该函数会从当前目录下的.env文件中读取环境变量，并将其加载到系统环境变量中。

        





#定义搜索工具 这里使用了google SerpApi来进行搜索，用户需要在.env文件中设置SERPAPI-API-KEY环境变量。
def search(query:str)->str:

    print(f'正在执行{query}的搜索...')
    try:
        api_key=os.getenv('SERPAPI-API-KEY')
        if not api_key:
            raise ValueError("SERPAPI-API-KEY未设置，请在.env文件中设置该环境变量。")
        
        params={
            'engine':'google',
            'q':query,
            'api_key':api_key,
            'gl':'cn',
            'hl':'zh-cn',



        
        }

        client=SerpApiClient(params)
        results=client.get_dict()


        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f'搜索失败：{e}'
    

#构建一个通用的工具执行器
class ToolExecutor:

    def __init__(self):
        self.tools:Dict[str,Dict[str,Any]]={}

    def register_tool(self,name:str,description:str,func:callable): #工具注册器

        if name in self.tools:
            print(f'警告：工具{name}已存在，将被覆盖。')

        self.tools[name]={'description':description,'func':func}

        print(f'工具{name}注册成功。')

    def getTool(self,name:str)->callable: #获取工具的函数
        if name not in self.tools:
            raise ValueError(f'工具{name}未注册。')
        return self.tools[name]['func']
    
    def getAvailableTools(self)->str: #获取可用的工具表，这是给llm进行工具选择，在调用Llm时让llm选择适合的工具

        return '\n'.join([f"{name}:{info['description']}" for name,info in self.tools.items()])
    




#ReAct agent的编码实现
#系统提示词设计：这里的提示词设计是为了让llm能够理解它的角色和任务，并且能够按照指定的格式进行输出。提示词中包含了可用工具的描述、用户的问题以及对话历史，以便llm能够根据这些信息进行思考和行动。

REACT_PROMPT_TEMPLATE='''
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
Question: {question}
History: {history}
'''

#实现ReAct核心循环

class ReActAgent:
    def __init__(self,llm_client:Myagent,tool_executor:ToolExecutor,max_iters:int=5):
        self.llm_client=llm_client #llm客户端
        self.tool_executor=tool_executor #工具管理器
        self.max_iters=max_iters #最大循环轮数
        self.history=[] #对话历史



    def run (self,question:str):
        self.history=[]

        current_step=0
        while current_step<self.max_iters:
            current_step+=1
            print(f'---循环轮数：第{current_step}轮---')

            tool_desc=self.tool_executor.getAvailableTools()
            history_str='\n'.join(self.history)
            prompt=REACT_PROMPT_TEMPLATE.format(question=question,tools=tool_desc,history=history_str) #使用提示词将question,history,tools进行格式化


            messages=[{'role':'user','content':prompt}]
            response_text=self.llm_client.think(messages=messages) or '' #调用Llm客户端


            if not response_text:
                print('错误：LLM未能返回有效结果。')
                break

            thought,action=self._parse_output(response_text) #使用文本解析器将返回的内容转换为action和thought


            if thought:
                print(f'思考：{thought}')
            if not action:
                print(f'警告：未能解析有效的action，流程中断。')
                break

            if action.startswith('Finish'): #终止条件，当llm认为已经得出答案的时候终止
                # ==================== [错误记录 #6] re.DOTALL 让 . 匹配换行符 ====================
                # 知识点：re.match/re.search 默认模式下，. 不匹配换行符 \n（等价于 [^\n]）。
                # LLM 输出的 Finish[答案] 中答案经常包含多行（如带换行的穿搭建议），
                # 没有 re.DOTALL 时 (.*) 遇到第一个 \n 就停止，匹配不到结尾的 ]，返回 None →
                # AttributeError: 'NoneType' object has no attribute 'group'
                # 修复：加 re.DOTALL（或 re.S），让 . 匹配包括 \n 在内的所有字符。
                final_answer=re.match(r"Finish\[(.*)\]",action,re.DOTALL).group(1)
                print(f'最终答案：{final_answer}')
                return final_answer

            tool_name,tool_input=self._parse_action(action) #获取llm所需的工具和工具输入
            if not tool_name or not tool_input:
                continue

            print(f'行动：{tool_name}[{tool_input}]')
            tool_function=self.tool_executor.getTool(tool_name) #根据工具名寻找工具函数
            if not tool_function:
                observation=f'错误：未找到名为 {tool_name} 的工具。'
            else:
                observation=tool_function(tool_input)

            print(f'观察：{observation}')
            #将该步所用的action和observation添加到历史中，以便下一轮的prompt能够包含这些信息
            self.history.append(f'Action: {action}')
            self.history.append(f'Observation: {observation}')

        print('已达最大循环步数，循环结束。')
        # ==================== [错误记录 #7] ReActAgent.run 返回 None 的连锁反应 ====================
        # 知识点：返回值类型一致性——如果函数正常路径返回 str，异常/兜底路径返回 None，
        # 调用方若不加判断就把返回值直接塞进 list，后续 str.join() 会报 TypeError。
        # 错误链：ReAct 跑满轮次 → return None → HybridAgent 里 history.append(None) → '\n'.join(history) 崩溃
        # 修复方向（二选一）：
        #   a) 这里改为 return ''（让 ReActAgent 保证返回 str）
        #   b) 调用方加 or '' 兜底（HybridAgent 已做）
        return None #不需要返回值，这里使用的是return none




    @staticmethod
    def _parse_output(text: str):
            """解析LLM的输出，提取Thought和Action。
            """
            # Thought: 匹配到 Action: 或文本末尾
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
            # Action: 匹配到文本末尾
            action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else None
            action = action_match.group(1).strip() if action_match else None
            return thought, action

    @staticmethod
    def _parse_action(action_text: str):
            """解析Action字符串，提取工具名称和输入。
            """
            match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
            if match:
                return match.group(1), match.group(2)
            return None, None


#以上是使用正则表达式和llm prompt进行约束生成的指令 对llm的指令遵循能力要求高，而且解析正则表达式的bug较多，下面使用pydantic进行约束生成和解析，pydantic是一个数据验证和设置管理的库，它使用Python的类型注解来定义数据模型，并提供了数据验证、序列化和反序列化等功能。



class ReActOutput(BaseModel):
    thought: str
    action_type:Literal['tool_call','finish']
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    final_answer: Optional[str] = None

#这样规定的json格式可以让llm在输出时遵循这个格式，方便后续的解析和处理。下面是使用pydantic进行约束生成和解析的ReActAgentplus类。

REACT_PROMPT_TEMPLATE1="""
你是一个能调用外部工具的助手。

可用工具：
{tools}

你必须**只返回一个 JSON 对象**，格式如下：

如果是调用工具：
{{"thought": "你的推理", "action_type": "tool_call", "tool_name": "工具名", "tool_input": "工具输入"}}

如果是输出最终答案：
{{"thought": "你的推理", "action_type": "finish", "final_answer": "最终答案"}}

现在回答：
Question: {question}
History: {history}
"""

class ReActAgentplus:
    def __init__(self,llm_client:Myagent,tool_executor:ToolExecutor,max_iters:int=5):
        self.llm_client=llm_client #llm客户端
        self.tool_executor=tool_executor #工具管理器
        self.max_iters=max_iters #最大循环轮数
        self.history=[] #对话历史

        self.consecutive_tool_failures = 0  # 连续工具调用失败计数器
        self.max_consecutive_errors = 3  # 最大连续工具调用失败次数
        self.last_action = None

    def _handle_tool_error(self, error_type: str, tool_name: str,tool_input: Optional[str] = None) -> str: #用于处理tool可能的调用失败的情况

        self.consecutive_tool_failures += 1
        if error_type == 'not found':
            return f'{tool_name} 工具错误[{self.consecutive_tool_failures}/{self.max_consecutive_errors}]:未找到名为 {tool_name} 的工具。'

        elif error_type == 'execution_error':
            return f'{tool_name} 工具执行异常[{self.consecutive_tool_failures}/{self.max_consecutive_errors}]:请检查输入参数「{tool_input}」是否合理，或尝试换一个参数。'
        elif error_type == 'repeated_call':
            return f'{tool_name} 重复调用[{self.consecutive_tool_failures}/{self.max_consecutive_errors}]:你刚才已用「{tool_name}[{tool_input}]」调过该工具，请换一个搜索词或换一种思路。'

        return '未知错误'

    def run(self,question:str):
        self.history=[]
        self.consecutive_tool_failures=0
        self.max_consecutive_errors=3
        self.last_action=None
        current_step=0
        while current_step<self.max_iters:
            current_step+=1
            print(f'---循环轮数：第{current_step}轮---')

            tool_desc=self.tool_executor.getAvailableTools()
            history_str='\n'.join(self.history)
            prompt=REACT_PROMPT_TEMPLATE1.format(question=question,tools=tool_desc,history=history_str) #使用提示词将question,history,tools进行格式化


            messages=[{'role':'user','content':prompt}]
            response_json= self.llm_client.think_json(messages=messages) or '' #调用Llm客户端


            if not response_json:
                print('错误：LLM未能返回有效结果。')
                break

                
            try:
                result=ReActOutput.model_validate(response_json) #使用pydantic进行json格式的验证和解析

            except Exception as e:
                print(f'错误：解析LLM响应时出错。错误信息：{e}')
                break

            thought=result.thought #直接使用pydantic模型的属性来获取thought,action_type,tool_name,tool_input,final_answer
            action_type=result.action_type #注意不能使用result['action_type']，因为pydantic模型的属性是通过点操作符访问的，而不是字典的键访问。
            tool_name=result.tool_name
            tool_input=result.tool_input

            if thought:
                print(f'思考：{thought}')
            if not action_type:
                print('错误，未能解析有效的action_type，流程中断。')
                break

            if action_type=='finish':
                final_answer=result.final_answer
                action_text=f"Finish[{final_answer}]"

                print(f'最终答案：{final_answer}')
                return final_answer

            if not tool_name or not tool_input:
                continue

            action_text=f"{tool_name}[{tool_input}]"

            if (tool_name,tool_input)==self.last_action: #出现重复调用同一个工具的情况，增加错误处理机制
                observation=self._handle_tool_error('repeated_call',tool_name,tool_input)
                self.last_action=(tool_name,tool_input)
            else:

                tool_function=self.tool_executor.getTool(tool_name)
                print(f'行动：{tool_name}[{tool_input}]')
                if not tool_function:
                    observation=self._handle_tool_error('not found',tool_name,tool_input)
                else:
                    try:
                        observation=tool_function(tool_input)
                        self.consecutive_tool_failures=0
                    except Exception as e:
                        observation=self._handle_tool_error('execution_error',tool_name,tool_input)
                self.last_action=(tool_name,tool_input)

            if self.consecutive_tool_failures>=self.max_consecutive_errors:
                print(f'连续{self.max_consecutive_errors}次调用工具失败，已强制退出。')
                break

            print(f'观察：{observation}')
            self.history.append(f'Action: {action_text}')
            self.history.append(f'Observation: {observation}')
        if current_step>=self.max_iters:
            print('已达到最大循环步数，循环结束。')
        return None

#json格式的ReActAgentplus类相比于文本格式的ReActAgent类，具有更强的结构化和可验证性，能够更好地约束llm的输出格式，减少解析错误的可能性。
#缺点是需要llm能够严格遵守json格式输出，否则会导致解析失败.




#TODO:在工具调用失败时，应该有一个机制让llm能够重新选择工具或者修改输入，而不是直接中断整个流程。可以考虑在工具调用失败时，将错误信息作为Observation的一部分返回给llm，让llm根据新的信息进行下一步的思考和行动.
#TODO:多工具的时候，搜索工具的结果可能不够准确，应该更新搜索方式







if __name__=='__main__':
    agent=Myagent()
    # Messages=[
    #     {"role":"system","content":"你是一个有用的助手"},
    #     {"role":"user","content":"一阶线性电路的全响应怎么求？"}
    # ]
    # agent.think(messages=Messages)

    toolExecutor=ToolExecutor()


    search_description='一个搜索工具，可以根据用户的查询返回相关的搜索结果。'
    toolExecutor.register_tool(name='Search',description=search_description,func=search)

    print(toolExecutor.getAvailableTools())

    # tool_name='Search'
    # tool_input='英伟达的最新显卡是什么？'
    #
    # tool_function=toolExecutor.getTool(tool_name)
    # if tool_function:
    #     observation=tool_function(tool_input)
    #     print(f'工具{tool_name}的输出结果为：\n{observation}')
    #
    # else:
    #     print(f'工具{tool_name}未注册，无法执行。')



    ReAct=ReActAgentplus(agent,toolExecutor,max_iters=5)
    ReAct.run('今天上海的天气如何？')