from importlib.metadata import metadata

from hello_agents.memory import MemoryConfig, MemoryManager
from torch.ao.nn.quantized.functional import threshold

from Tool import Tool
from datetime import datetime
from typing import List
class MemoryTool(Tool):
    def __init__(self,
                 user_id:str='default_user',
                 memory_config:MemoryConfig=None,
                 memory_types:List[str]=None):
        super().__init__(
            name='memory',
            description='记忆工具 - 可以存储和检索对话历史、知识和经验'
        )

        self.memory_config=memory_config or MemoryConfig()
        self.memory_types=memory_types or ['working','episodic','semantic']
        self.memory_manager=MemoryManager(
            config=self.memory_config,

            user_id=user_id,

            enable_working='working' in self.memory_types,
            enable_episodic='episodic' in self.memory_types,
            enable_semantic='semantic' in self.memory_types,
            enable_perceptual='perceptual' in self.memory_types
        )




    def execute(self,action:str,**kwargs):
        if action=='add':
            return self._add_memory(**kwargs)
        elif action=='search':
            return self._search_memory(**kwargs)
        elif action=='summary':
            return self._get_summary(**kwargs)
        elif action=='stats':
            return self._get_stats(**kwargs)
        elif action=='update':
            return self._update_memory(**kwargs)
        elif action=='remove':
            return self._remove_memory(**kwargs)
        elif action =='forget':
            return self._forget_memory(**kwargs)
        elif action=='consolidate':
            return self._consolidate_memory(**kwargs)
        elif action=='clear_all':
            return self._clear_all(**kwargs)


    def _add_memory(self,
                    content:str,
                    memory_type:str='working',
                    importance:float=0.5,
                    file_path:str=None,
                    modality:str=None,
                    **metadata):
        try:
            if self.current_session_id is None:
                self.current_session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if memory_type=='perceptual' and file_path:
                inferred=modality or self._infer_modality(file_path)
                metadata.setdefault('modality',inferred)
                metadata.setdefault('raw_data',file_path)

            metadata.update({
                'session_id':self.current_session_id,
                'timestamp':datetime.now().isoformat()
            })

            memory_id=self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,

                metadata=metadata,
                auto_classify=False
            )

            return f'记忆已添加（ID：{memory_id[:8]}...）'
        except Exception as e:
            return f'添加记忆失败:{str(e)}'



    def _search_memory(self,
                       query:str,
                       limit:int=5,

                       memory_type:str=None,
                       memory_types:List[str]=None,
                       min_importance:float=0.1)->str:
        try:
            if memory_types is None and memory_type:
                memory_types=[memory_type]

            results=self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance
            )

            if not results:
                return f'未找到与{query}相关的记忆'

            formatted_results=[]
            formatted_results.append(f'找到{len(results)}条相关记忆：')

            for i,memory in enumerate(results,1):
                memory_type_label={
                    'working':'工作记忆',
                    'episodic':'情景记忆',
                    'semantic':'语义记忆',
                    'perceptual':'感知记忆'

                }.get(memory.memory_type,memory.memory_type)


                content_preview= memory.content[:80] + '...' if len(memory.content) > 80 else memory.content
                formatted_results.append(f"{i}. [{memory_type_label}] {content_preview} (重要性：{memory.importance:.2f})")

            return "\n".join(formatted_results)

        except Exception as e:
            return f'搜索记忆失败：{str(e)}'

    def _forget(self,strategy:str='importance_based',threhold:float=0.1,max_age_days:int=30)->str:
        try:
            count= self.memory_manager.forget_memomries(

                strategy=strategy,
                threshold=threshold,

                max_age_days=max_age_days

            )
            return f'已遗忘{count}条记忆（策略：{strategy}）'
        except Exception as e:
            return f'遗忘记忆失败:{str(e)}'

    def _consolidate(self,
                     from_type:str='working',
                     to_type:str='episodic',
                     importance_threshold:float=0.7)->str:
        try:
            count=self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold
            )

            return f'已整合{count}条记忆为长期记忆（{from_type}->{to_type},阈值={importance_threshold}）'
        except Exception as e:
            return f'整合记忆失败：{str(e)}'



