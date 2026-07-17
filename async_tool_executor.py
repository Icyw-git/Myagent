import asyncio #作用是提供对异步编程的支持，允许你编写非阻塞的代码，从而在等待I/O操作（如网络请求、文件读写等）时不会阻塞整个程序的执行。通过使用async/await语法，你可以定义异步函数，并在需要等待的地方使用await关键字，从而实现高效的并发处理。

import concurrent.futures #作用是提供一个高级接口，用于异步执行可调用对象（如函数、方法等），并在后台线程或进程中运行它们，从而实现并发执行。它允许你提交任务，并在任务完成时获取结果，而无需阻塞主线程。
from typing import Dict,Any,List,Callable
from Tool import ToolRegistry

class AsyncToolExecutor:

    def __init__(self,registry:ToolRegistry,max_workers:int=4):
        self.registry=registry
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) #作用是创建一个线程池执行器，允许你在后台线程中并发执行任务，从而提高程序的性能和响应能力。通过指定max_workers参数，你可以控制线程池中同时运行的最大线程数，以便根据任务的性质和系统资源进行优化。

    async def execute_tool_async(self,tool_name:str,input_data:str)->str:

        loop=asyncio.get_event_loop() #作用是获取当前线程的事件循环对象，它是异步编程的核心，用于调度和管理异步任务。通过事件循环，你可以注册协程、回调函数和I/O操作，从而实现非阻塞的并发执行。在异步函数中，通常使用await关键字等待协程的结果，而事件循环负责在后台处理这些任务。
        def _execute():
            return self.registry.execute_tool(tool_name,input_data)

        result=await loop.run_in_executor(self.executor,_execute) #run_in_executor 方法的作用是将一个阻塞的函数（在这里是 _execute）提交给指定的执行器（executor）在后台线程中运行，并返回一个协程对象。通过使用 await 关键字，你可以在异步函数中等待该协程完成，从而实现非阻塞的并发执行。这种方式允许你在等待 I/O 操作或其他耗时任务时，不会阻塞主线程，从而提高程序的性能和响应能力。
        return result

    async def execute_task_async(self,tasks:List[Dict[str,str]]):
        async_tasks=[]
        for task in tasks:
            tool_name=task['tool_name']
            input_data=task['input_data']
            async_task=self.execute_tool_async(tool_name,input_data)
            async_tasks.append(async_task)

        results=await asyncio.gather(*async_tasks) #作用是并发执行多个异步任务，并在所有任务完成后返回它们的结果。通过使用 asyncio.gather()，你可以将一个包含协程对象的可迭代对象（如列表）传递给它，从而同时运行这些协程，而不会阻塞主线程。函数会返回一个包含所有协程结果的列表，保持与输入顺序一致。这种方式允许你高效地处理多个异步操作，提高程序的性能和响应能力。
        return results

    def __del__(self):
        if hasattr(self,'executor'):
            self.executor.shutdown(wait=True)







