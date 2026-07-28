from contextbase import ContextPacket, ContextConfig
from typing import List, Optional, Dict, Any
from Message import Message
from datetime import datetime
from memory_src.memory_tool import MemoryTool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from hello_agents import HelloAgentsLLM

class ContextBuilder:

    def __init__(self,llm:HelloAgentsLLM,config:Optional[ContextConfig]=None):
        self.config=config or ContextConfig()
        self.memory_tool=MemoryTool()
        self.rag_tool=None
        self.llm=llm



    def  _gather(self,user_query:str,conversation_history:Optional[List[Message]]=None,system_instructions:Optional[str]=None,custom_packets:Optional[List[ContextPacket]]=None,custom_context:Optional[List[ContextPacket]]=None):
        """汇集所有候选信息

    Args:
        user_query: 用户查询
        conversation_history: 对话历史
        system_instructions: 系统指令
        custom_packets: 自定义信息包

    Returns:
        List[ContextPacket]: 候选信息列表
    """
        packets=[]

        if system_instructions:
            packets.append(
                ContextPacket(
                    content=system_instructions,
                    timestamp=datetime.now(),
                    token_count=self._count_tokens(system_instructions),
                    relevance_score=1.0,
                    metadata={'type':'system_instructions','priority':'high'}

                    
                )
            )
        if self.memory_tool:
            try:
                # 错误记录：曾写 memory_tool.run({'action':'search',...})，run() 返回给人看的 str，
                # 不能当 List[Dict] 去 _parse_memory_results。结构化检索要用 search_items() → List[MemoryItem]。
                memory_results = self.memory_tool.search_items(
                    query=user_query,
                    limit=10,
                    min_importance=0.3,
                )
                memory_packets = self._parse_memory_results(memory_results, user_query)
                packets.extend(memory_packets)
            except Exception as e:
                print(f'[Warning] Failed to gather memory context: {e}')

        if self.rag_tool:
            try:
                rag_results=self.rag_tool.run(
                    {
                        'action':'search',
                        'query':user_query,
                        'limit':5,
                        'min_score':0.3
                    }
                )
                rag_packets=self._parse_rag_resylts(rag_results,user_query)
                packets.extend(rag_packets)
            except Exception as e :
                print(f'[Warning] Failed to gather RAG context: {e}')

        if conversation_history:
            # 错误记录：这里用了 self.config.max_history，但 ContextConfig 尚未定义该字段，
            # 跑到对话历史分支会 AttributeError；需要在 ContextConfig 里补 max_history 或改成常量。
            recent_history=conversation_history[-self.config.max_history:]
            for msg in recent_history:
                packets.append(

                    ContextPacket(
                        content=f'{msg.role}:{msg.content}',
                        timestamp=msg.timestamp if hasattr(msg,'timestamp') else datetime.now(),
                        token_count=self._count_tokens(msg.content),
                        relevance_score=0.6,
                        metadata={'type':'conversation_history','role':msg.role}

                    )
                )
        if custom_packets:
            packets.extend(custom_packets)

        print(f'[ContextBuilder] Gathered {len(packets)} candidate context packets')
        return packets

    def _count_tokens(self, text: str) -> int:
        """估算文本 token 数（用于上下文预算，不要求与模型完全一致）。

        优先用 tiktoken（若已安装）；否则按中英混合启发式：
        - 中日韩等宽字符约 1.5 字/token
        - 其余（英文等）约 4 字符/token
        """
        if not text:
            return 0
        try:
            import tiktoken
            # cl100k_base 覆盖 GPT-4/多数 OpenAI 兼容场景；非 OpenAI 模型也够做预算
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # 无 tiktoken 或编码失败时走启发式兜底
            cjk = 0
            other = 0
            for ch in text:
                # CJK 统一表意文字 + 常用标点/假名等大致区间
                if (
                    "\u4e00" <= ch <= "\u9fff"
                    or "\u3400" <= ch <= "\u4dbf"
                    or "\uf900" <= ch <= "\ufaff"
                    or "\u3040" <= ch <= "\u30ff"
                    or "\uac00" <= ch <= "\ud7af"
                ):
                    cjk += 1
                else:
                    other += 1
            return max(1, int(cjk / 1.5 + other / 4.0))

    def _parse_memory_results(self, memory_results, user_query: str) -> List[ContextPacket]:
        """将 MemoryItem 列表（search_items）解析为 ContextPacket。

        MemoryItem 字段：content / timestamp(ISO str) / importance / memory_type
        没有 relevance_score、source —— 用 importance 作相关性近似，memory_type 作 source。

        错误记录：曾按 List[Dict] 写 result['content']/['relevance_score']/['source']，
        且用 hasattr(result,'timestamp') 判断 dict（拿不到键）。应对齐 MemoryItem 属性。
        """
        packets = []
        if not memory_results:
            return []

        for item in memory_results:
            # timestamp 在 MemoryItem 里是 ISO 字符串，转成 datetime 供 ContextPacket 用
            ts = datetime.now()
            raw_ts = getattr(item, "timestamp", None)
            if isinstance(raw_ts, datetime):
                ts = raw_ts
            elif isinstance(raw_ts, str) and raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts)
                except ValueError:
                    pass

            content = getattr(item, "content", "") or ""  #getattr方法，如果item有content属性，则返回content，否则返回空字符串
            importance = float(getattr(item, "importance", 0.5) or 0.5)
            memory_type = getattr(item, "memory_type", "unknown")

            packets.append(
                ContextPacket(
                    content=content,
                    timestamp=ts,
                    token_count=self._count_tokens(content),
                    relevance_score=importance,  # 暂无独立相关分，用 importance 顶上
                    metadata={
                        "type": "memory_result",
                        "source": memory_type,
                        "memory_id": getattr(item, "id", None),
                        "query": user_query,
                    },
                )
            )

        return packets

    def _select(self,
    packets: List[ContextPacket],
    user_query: str,
    available_tokens: int
    )->List[ContextPacket]:

        """选择最相关的信息包

        Args:
            packets: 候选信息包列表
            user_query: 用户查询(用于计算相关性)
            available_tokens: 可用的 token 数量

        Returns:
            List[ContextPacket]: 选中的信息包列表
        """
        # 错误记录：metadata 是 dict，不能用 getattr(p.metadata,'type')（属性不存在会当 None）。
        # 应写 p.metadata.get('type')。另外上面写入的是 'system_instructions'，这里却比
        # 'system_instruction'，类型名不一致会导致系统包选不出来。
        system_packets=[p for p in packets if getattr(p.metadata,'type')=='system_instruction']
        other_packets=[p for p in packets if getattr(p.metadata,'type')!='system_instruction']

        system_tokens=sum(p.token_count for p in system_packets)
        remaining_tokens=available_tokens-system_tokens

        if remaining_tokens<=0:
            print('[Warning] 系统指令已占满所有token预算')
            return system_packets

        scored_packets=[]
        for packet in other_packets:
            if packet.relevance_score==0.5:
                relevance=self._calculate_relevance(packet.content,user_query)
                packet.relevance_score=relevance

            recency=self._calculate_recency(packet.timestamp)

            combined_score=(
                self.config.relevance_weight*packet.relevance_score+self.config.recency_weight*recency
            )

            if packet.relevance_score >=self.config.min_relevance:
                scored_packets.append((combined_score,packet))

        # 错误记录：下面 sort / 装填 / return 曾缩进进 for 循环内，导致只处理完第一条
        # other_packet 就 return，选择逻辑错误。应与 for 同级（只排序、装填一次）。
        scored_packets.sort(key=lambda x:x[0],reverse=True)

        selected=system_packets.copy()
        current_tokens=system_tokens

        for score,packet in scored_packets:
            if current_tokens+packet.token_count <=available_tokens:
                selected.append(packet)
                current_tokens+=packet.token_count
            else:
                break

        print(f'[ContextBuilder] 选择了{len(selected)}个信息包，共{current_tokens} tokens')
        return selected

    def _calculate_relevance(self,content:str,query:str)->float:
        sentences=[content,query]
        model=SentenceTransformer('all-MiniLM-L6-v2')
        embeddings=model.encode(sentences)

        similarity=cosine_similarity([embeddings[0],embeddings[1]])[0][0]
        return similarity

    def _calculate_recency(self,timestamp:datetime)->float:
        """计算时间近因性分数

            使用指数衰减模型,24小时内保持高分,之后逐渐衰减。

            Args:
                timestamp: 信息的时间戳

            Returns:
                float: 新近性分数(0.0-1.0)
        """
        import math
        age_hours=(datetime.now()-timestamp).total_seconds()/3600
        decay_factor=0.1
        recency_score=math.exp(-decay_factor*age_hours/24)
        return max(0.1,min(1.0,recency_score))

    def _structure(self,selected_packets:List[ContextPacket],user_query:str)->str:
        """将选中的信息包组织成结构化的上下文模板

            Args:
                selected_packets: 选中的信息包列表
                user_query: 用户查询

            Returns:
                str: 结构化的上下文字符串
        """
        system_instructions=[]
        evidence=[]
        context=[]

        for packet in selected_packets:
            packet_type=packet.metadata.get('type','general')

            if packet_type=='system_instruction':
                system_instructions.append(packet.content)
            elif packet_type in ['rag_result','knowledge']:
                evidence.append(packet.content)
            else:
                context.append(packet.content)

        sections=[]

        if system_instructions:
            sections.append("[Role & Policies]\n"+'\n'.join(system_instructions))

        sections.append(f'[Task]\n{user_query}')

        if evidence:
            sections.append("[Evidence]\n"+"\n---\n".join(evidence))

        if context:
            sections.append("[Context]\n"+'\n'.join(context))

        sections.append("[Output]\n请根据以上信息，提供准确有据的答案。")

        return '\n\n'.join(sections)

    def _compress(self,context:str,max_tokens:int)->str:
        """压缩超限的上下文

            Args:
                context: 原始上下文
                max_tokens: 最大 token 限制

            Returns:
                str: 压缩后的上下文
        """
        # 错误记录：曾写 self._count_tokens() 漏传 context，TypeError。必须传要统计的字符串。
        current_tokens=self._count_tokens(context)
        if current_tokens<=max_tokens:
            return context

        print(f'[ContextBuilder] 上下文超限（{current_tokens}>{max_tokens}）,执行压缩')

        sections=context.split('\n\n')
        compressed_sections=[]
        current_total=0

        for section in sections:
            section_tokens=self._count_tokens(section)

            if current_total+section_tokens <=max_tokens:
                compressed_sections.append(section)
                current_total+=section_tokens
            else:
                remaining_tokens=max_tokens-current_total

                if remaining_tokens>50:
                    # 错误记录：曾 truncate 到 remaining_tokens 后再拼 '\n[...内容已压缩...]'，
                    # 后缀也占 token，总长会略超 max_tokens（测出 154>150）。要先扣后缀预算。
                    suffix='\n[...内容已压缩...]'
                    suffix_tokens=self._count_tokens(suffix)
                    body_budget=max(1,remaining_tokens-suffix_tokens)
                    truncated=self._truncate_text(section,body_budget)
                    compressed_sections.append(truncated+suffix)

                    break
        compressed_context='\n\n'.join(compressed_sections)
        final_tokens=self._count_tokens(compressed_context)
        print(f'[ContextBuilder] 压缩完成：{current_tokens}->{final_tokens} tokens')
        return compressed_context

    def _truncate_text(self,text:str,max_tokens:int)->str:
        # 错误记录：曾一上来就调 LLM，短文本也会被「摘要」改写。不超限应直接原样返回。
        if not text or max_tokens<=0:
            return ""
        if self._count_tokens(text)<=max_tokens:
            return text

        max_words=int(1.5*max_tokens)
        prompt=f"""
        你是一个专业的文本压缩专家，请根据以下上下文，摘要出最核心的信息，并返回摘要后的文本。要尽可能保留原文意思，不要丢失重要信息。最多保留{max_tokens}个token。约{max_words}个字。
        上下文：{text}
        注意只返回摘要后的文本，不要返回任何其他内容。

        
        """
        message=[{'role':'system','content':prompt}]
        if self.llm:
            try:
                response=self.llm.invoke(message)
                if self._count_tokens(response)<=max_tokens:
                    return response
                else:
                    # 错误记录：摘要仍超长时曾对原文 text 做 tiktoken 硬截，丢掉了 LLM 摘要结果。
                    # 应硬截 response（摘要），保留更多关键信息。
                    import tiktoken
                    enc=tiktoken.get_encoding("cl100k_base")
                    encoded=enc.encode(response)
                    truncated=encoded[:max_tokens]
                    return enc.decode(truncated)
                
            except Exception as e:
                print(f'[Warning] Failed to truncate text: {e}')
                length=int(len(text)*max_tokens/self._count_tokens(text))
                return text[:length]
        else:
            length = int(len(text) * max_tokens / self._count_tokens(text))
            return text[:length]






        




