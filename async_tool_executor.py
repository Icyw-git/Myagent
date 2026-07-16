import asyncio #作用是提供对异步编程的支持，允许你编写非阻塞的代码，从而在等待I/O操作（如网络请求、文件读写等）时不会阻塞整个程序的执行。通过使用async/await语法，你可以定义异步函数，并在需要等待的地方使用await关键字，从而实现高效的并发处理。

import concurrent.futures #作用是提供一个高级接口，用于异步执行可调用对象（如函数、方法等），并在后台线程或进程中运行它们，从而实现并发执行。它允许你提交任务，并在任务完成时获取结果，而无需阻塞主线程。
from typing import Dict,Any,List,Callable
from Tool import ToolRegistry

class AsyncToolExecutor:

    def __init__(self,registry:ToolRegistry,max_workers:int=4):
        self.registry=registry
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self,tool_name:str,input_data:str)->str:

        loop=asyncio.get_event_loop()
        def _execute():
            return self.registry.execute_tool(tool_name,input_data)

        result=await loop.run_in_executor(self.executor,_execute)
        return result

    async def execute_task_async(self,tasks:List[Dict[str,str]]):
        async_tasks=[]
        for task in tasks:
            tool_name=task['tool_name']
            input_data=task['input_data']
            async_task=self.execute_tool_async(tool_name,input_data)
            async_tasks.append(async_task)

        results=await asyncio.gather(*async_tasks)
        return results

    def __del__(self):
        if hasattr(self,'executor'):
            self.executor.shutdown(wait=True)







