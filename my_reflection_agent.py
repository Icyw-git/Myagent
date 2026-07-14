from hello_agents import HelloAgentsLLM
from Config import Config
from Message import Message
from typing import Optional,List,Dict

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

class MyReflectionAgent:
    def __init__(self,
                 name:str,
                 llm:HelloAgentsLLM,
                 system_prompt:Optional[str]=None,
                 config:Optional[Config]=None,
                 custom_prompts:Optional[Dict[str,str]]=None,
                 max_iterations:int=3

                 ):
        self.name=name
        self.llm=llm
        self.system_prompt=system_prompt or '你是一个有用的AI助手，能够回答用户的问题并提供帮助。'
        self.config=config or Config()
        self.prompt_template=custom_prompts if custom_prompts else DEFAULT_PROMPTS
        self.max_iterations=max_iterations
        self._history:List[Message]=[]
        print(f'{name}初始化完成')

    def run(self,input_text:str,**kwargs)->str:
        print(f'{self.name}正在处理：{input_text}')

        messages=[]
        messages.append({'role':'system','content':self.system_prompt})
        for msg in self._history:
            messages.append({'role':msg.role,'content':msg.content})



        initial_prompt=self.prompt_template['initial'].format(task=input_text)
        messages.append({'role':'user','content':initial_prompt})

        response=self.llm.invoke(messages,**kwargs) or ''
        current_iteration=0
        while current_iteration<self.max_iterations:
            current_iteration+=1
            print(f'--- 第{current_iteration}轮反思 ---')
            reflect_prompt=self.prompt_template['reflect'].format(task=input_text,content=response)
            messages.append({'role':'user','content':reflect_prompt})
            reflect_response=self.llm.invoke(messages,**kwargs) or ''
            messages.append({'role':'assistant','content':reflect_response})


            if '无需改进' in reflect_response:
                print(f'反思结果: 无需改进，最终答案为：{response}')
                final_answer=response
                self.add_history(Message(input_text,'user'))
                self.add_history(Message(final_answer,'assistant'))
                return final_answer

            refine_prompt=self.prompt_template['refine'].format(task=input_text,last_attempt=response,feedback=reflect_response)
            messages.append({'role':'user','content':refine_prompt})
            response=self.llm.invoke(messages,**kwargs) or ''

        final_answer=response
        print(f'已达到最大反思轮次,最终答案为：{final_answer}')
        self.add_history(Message(input_text,'user'))
        self.add_history(Message(final_answer,'assistant'))
        return final_answer

    def add_history(self,message:Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history.copy()




