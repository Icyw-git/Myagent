from typing import Optional

from hello_agents import SimpleAgent,HelloAgentsLLM,ToolRegistry
import inspect
from dotenv import load_dotenv
from Config import Config
load_dotenv()

class MySimpleAgent(SimpleAgent):
    def __init__(self,
                 name:str,
                 llm: HelloAgentsLLM,
                 system_prompt:Optional[str]=None,
                 tool_registry:Optional[ToolRegistry]=None,
                 enable_tool_calling:Optional[bool]=False,
                 config:Optional[Config]=None


                 ):
        super().__init__(name=name,system_prompt=system_prompt,llm=llm,config=config)
        self.enable_tool_calling=enable_tool_calling and tool_registry is not None
        self.tool_registry=tool_registry
