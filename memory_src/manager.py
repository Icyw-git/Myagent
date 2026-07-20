from typing import Optional

from hello_agents.memory import MemoryConfig, WorkingMemory, EpisodicMemory, SemanticMemory, PerceptualMemory


class MemoryManager:
    def  __init__(self,
                  config:Optional[MemoryConfig]=None,
                  user_id:str='default_user',
                  enable_working:bool=True,
                  enable_episodic:bool=True,
                  enable_semantic:bool=True,
                  enable_perceptual:bool=False
    ):

        self.config=config or MemoryConfig()
        self.user_id=user_id
        self.store=MemoryStore(self.config)

        self.memory_types=[]

        if enable_working:
            self.memory_types['working']=WorkingMemory(self.config,self.store)
        if enable_episodic:
            self.memory_types['episodic']=EpisodicMemory(self.config,self.store)

        if enable_semantic:
            self.memory_types['semantic']=SemanticMemory(self.config,self.store)
        if enable_perceptual:
            self.memory_types['perceptual']=PerceptualMemory(self.config,self.store)
