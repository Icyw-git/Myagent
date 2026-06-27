import os
from typing import List,Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Myagent:
    '''
    Myagent类用于与openai的api进行交互，返回模型的响应结果。
    这是自定义的一个类，主要用于封装与openai的api交互的逻辑，方便在其他地方调用。
    '''

    def __init__(self, api_key: str = None, base_url: str = None, model_id: str = None, timeout: int = 60):
        self.api_key = api_key if api_key else os.getenv('LLM-API-KEY')
        self.base_url = base_url if base_url else os.getenv('LLM-BASE-URL')
        self.model_id = model_id if model_id else os.getenv('LLM-MODEL-ID')
        self.timeout = timeout if timeout else int(os.getenv('LLM-TIMEOUT'))
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)  # 初始化llm客户端

    def think(self, messages: List[Dict[str, str]], temperature: float = 0):
        print(f'正在调用{self.model_id}进行思考...')  # 使用openai的格式
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,  # 使用的模型名称
                messages=messages,
                temperature=temperature,
                stream=True,

            )  # 使用流式输出，模型会返回内容块，而不是一次性返回完整的响应，这样可以更快地获取到模型的响应内容，并且在控制台上实时显示出来。

            print(f'LLM响应成功：')
            collected_content = []
            for chunk in response:
                if not chunk.choices:  # 若无内容就跳过
                    continue
                content = chunk.choices[0].delta.content or ''
                print(content, end='', flush=True)  # end=''取消自动换行，flush=True强制刷新输出缓冲区，使得内容能够立即显示在控制台上。
                collected_content.append(content)

            print()  # 默认的end参数是换行符，这里调用print()函数来输出一个换行符，以确保在输出完内容后换行。
            return ''.join(collected_content)  # 将收集到的内容块连接成一个完整的字符串，并返回给调用者。

        except Exception as e:
            print(f'LLM响应失败：{e}')
            return

if __name__ == '__main__':
    llm_client = Myagent()
    prompt=[{'role':'user','content':'请帮我规划一下去武汉的旅游路线。'}]
    llm_client.think(prompt)