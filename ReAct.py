import os 
from openai import OpenAI
from dotenv import load_dotenv
from typing import List ,Dict


load_dotenv() #使用load_dotenv()函数加载环境变量，该环境变量作用域为当前进程，且只在当前进程中有效。该函数会从当前目录下的.env文件中读取环境变量，并将其加载到系统环境变量中。

class Myagent:
    '''
    Myagent类用于与openai的api进行交互，返回模型的响应结果。
    这是自定义的一个类，主要用于封装与openai的api交互的逻辑，方便在其他地方调用。
    '''

    def __init__(self,api_key:str =None,base_url:str=None,model_id:str=None,timeout:int=60):
        self.api_key= os.getenv('LLM-API-KEY') if os.getenv('LLM-API-KEY') else api_key
        self.base_url= os.getenv('LLM-BASE-URL') if os.getenv('LLM-BASE-URL') else base_url
        self.model_id= os.getenv('LLM-MODEL-ID') if os.getenv('LLM-MODEL-ID') else model_id
        self.timeout= os.getenv('LLM-TIMEOUT') if os.getenv('LLM-TIMEOUT') else timeout
        self.client= OpenAI(api_key=self.api_key,base_url=self.base_url,timeout=self.timeout)
        
    def think(self,messages:List[Dict[str,str]],temperature:float=0):
        print(f'正在调用{self.model_id}进行思考...')
        try:
            response=self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                stream=True,

            ) #使用流式输出，模型会返回内容块，而不是一次性返回完整的响应，这样可以更快地获取到模型的响应内容，并且在控制台上实时显示出来。

            print(f'LLM响应成功：')
            collected_content=[]
            for chunk in response:
                if not chunk.choices:
                    continue
                content=chunk.choices[0].delta.content or ''
                print(content,end='',flush=True) #end=''取消自动换行，flush=True强制刷新输出缓冲区，使得内容能够立即显示在控制台上。
                collected_content.append(content)

            print() #默认的end参数是换行符，这里调用print()函数来输出一个换行符，以确保在输出完内容后换行。
            return ''.join(collected_content) #将收集到的内容块连接成一个完整的字符串，并返回给调用者。
        
        except Exception as e:
            print(f'LLM响应失败：{e}')
            return 
        


if __name__=='__main__':
    agent=Myagent()
    Messages=[
        {"role":"system","content":"你是一个有用的助手"},
        {"role":"user","content":"写一个快速排序算法"}
    ]
    agent.think(messages=Messages)