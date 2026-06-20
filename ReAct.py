import os 
from openai import OpenAI
from dotenv import load_dotenv
from typing import List ,Dict,Any
from serpapi import SerpApiClient



load_dotenv() #使用load_dotenv()函数加载环境变量，该环境变量作用域为当前进程，且只在当前进程中有效。该函数会从当前目录下的.env文件中读取环境变量，并将其加载到系统环境变量中。

class Myagent:
    '''
    Myagent类用于与openai的api进行交互，返回模型的响应结果。
    这是自定义的一个类，主要用于封装与openai的api交互的逻辑，方便在其他地方调用。
    '''

    def __init__(self,api_key:str =None,base_url:str=None,model_id:str=None,timeout:int=60):
        self.api_key= api_key if api_key else os.getenv('LLM-API-KEY')
        self.base_url= base_url if base_url else os.getenv('LLM-BASE-URL')
        self.model_id= model_id if model_id else os.getenv('LLM-MODEL-ID')
        self.timeout= timeout if timeout else int(os.getenv('LLM-TIMEOUT'))
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
        





#定义搜索工具
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

    def register_tool(self,name:str,description:str,func:callable):

        if name in self.tools:
            print(f'警告：工具{name}已存在，将被覆盖。')

        self.tools[name]={'description':description,'func':func}

        print(f'工具{name}注册成功。')

    def getTool(self,name:str)->callable:
        if name not in self.tools:
            raise ValueError(f'工具{name}未注册。')
        return self.tools[name]['func']
    
    def getAvailableTools(self)->str:

        return '\n'.join([f"{name}:{info['description']}" for name,info in self.tools.items()])
    
    

if __name__=='__main__':
    agent=Myagent()
    Messages=[
        {"role":"system","content":"你是一个有用的助手"},
        {"role":"user","content":"一阶线性电路的全响应怎么求？"}
    ]
    agent.think(messages=Messages)

    toolExecutor=ToolExecutor()


    search_description='一个搜索工具，可以根据用户的查询返回相关的搜索结果。' 
    toolExecutor.register_tool(name='Search',description=search_description,func=search)

    print(toolExecutor.getAvailableTools())

    tool_name='Search'
    tool_input='英伟达的最新显卡是什么？'

    tool_function=toolExecutor.getTool(tool_name)
    if tool_function:
        observation=tool_function(tool_input)
        print(f'工具{tool_name}的输出结果为：\n{observation}')

    else:
        print(f'工具{tool_name}未注册，无法执行。')










