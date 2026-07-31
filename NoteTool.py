from typing import List,Optional,Dict,Any,Tuple
from datetime import datetime
import yaml
import os



class NoteTool:
    def __init__(self,workspace:str='./notes'):
        # 1) 工作目录：笔记 .md 与 index.yaml 都落在这里
        self.workspace=workspace
        os.makedirs(self.workspace,exist_ok=True)
        # 2) 索引文件：用 YAML 存 {note_id: metadata}，避免每次全目录扫
        self.index_file=os.path.join(self.workspace,'index.yaml')
        # 3) 启动时把索引读进内存；无文件则空 dict（见 _load_index）
        self.index=self._load_index()

    def _load_index(self)->Dict[str,Any]:
        """从 index.yaml 加载笔记索引。

        错误记录：__init__ 直接 self.index=self._load_index()，若未实现本方法会 AttributeError。
        文件不存在 / 空文件 / YAML 非法时，应回退为空 dict，不要让初始化炸掉。
        """
        if not os.path.exists(self.index_file):
            return {}
        try:
            with open(self.index_file,'r',encoding='utf-8') as f:
                data=yaml.safe_load(f)
            # safe_load 空文件会返回 None；索引约定是 dict
            return data if isinstance(data,dict) else {}
        except Exception as e:
            print(f'[Warning] 加载笔记索引失败，使用空索引：{e}')
            return {}

    def _save_index(self)->None:
        """把内存中的 self.index 写回 index.yaml。

        create/update/delete 后都要调，否则重启后索引与磁盘 .md 不一致。
        """
        with open(self.index_file,'w',encoding='utf-8') as f:
            yaml.dump(self.index,f,allow_unicode=True,sort_keys=False)



    def _create_note(self,
                     title:str,
                     content:str,
                     note_type:str='general',
                     tags:Optional[List[str]]=None
                     )->str:
        """创建笔记

            Args:
                title: 笔记标题
                content: 笔记内容(Markdown格式)
                note_type: 笔记类型(task_state/conclusion/blocker/action/reference/general)
                tags: 标签列表

            Returns:
                str: 笔记ID

        """
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        note_id=f'note_{timestamp}_{len(self.index)}'
        metadata={
            'id':note_id,
            'title':title,
            'type':note_type,
            'tags':tags or [],
            'created_at':datetime.now().isoformat(),
            'updated_at':datetime.now().isoformat()
        }

        md_content=self._build_markdown(metadata,content)


        file_path=os.path.join(self.workspace,f'{note_id}.md')
        with open(file_path,'w',encoding='utf-8') as f:
            f.write(md_content)


        metadata['file_path']=file_path #元数据中存储路径
        self.index[note_id]=metadata #元数据存储到内存中
        self._save_index()

        return note_id

    def _build_markdown(self,metadata:dict,content:str)->str:
        """构建Markdown格式的笔记内容"""
        yaml_header=yaml.dump(metadata,allow_unicode=True,sort_keys=False)

        return f'---\n{yaml_header}---\n\n{content}'


    def _read_note(self,note_id:str)->Dict:
        """读取笔记内容

            Args:
                note_id: 笔记ID

            Returns:
                Dict: 包含元数据和内容的字典
        """

        if note_id not in self.index:
            raise ValueError(f'笔记ID {note_id} 不存在')
        file_path=self.index[note_id]['file_path'] #读取元数据

        with open(file_path,'r',encoding='utf-8') as f:
            raw_content=f.read()

        metadata,content=self._parse_markdown(raw_content) #解析 Markdown 文件，分离 YAML 和正文

        return {
            'metadata':metadata,
            'content':content
        }

    def _parse_markdown(self,raw_content:str)->Tuple[Dict,str]:
        """解析 Markdown 文件(分离 YAML 和正文)"""
        parts=raw_content.split('---\n',2) #这里使用 2 作为 maxsplit，确保只分割前两个 '---\n'，避免正文中出现 '---\n' 时被误分割

        if len(parts)>=3:
            yaml_str=parts[1]
            content=parts[2].strip()
            metadata=yaml.safe_load(yaml_str) #safe_load是解析字符串然后返回一个Python对象，通常是字典或列表

        else:
            metadata={}

            content=raw_content.strip()

        return metadata,content


    def _update_note(self,
                     note_id:str,
                     title:Optional[str]=None,
                     content:Optional[str]=None,
                     note_type:Optional[str]=None,
                     tags:Optional[List[str]]=None)->str:
        """更新笔记

            Args:
                note_id: 笔记ID
                title: 新标题(可选)
                content: 新内容(可选)
                note_type: 新类型(可选)
                tags: 新标签(可选)

            Returns:
                str: 操作结果消息
        """
        if note_id not in self.index:
            raise ValueError(f'笔记不存在：{note_id}')

        note=self._read_note(note_id)
        metadata=note['metadata']
        old_content=note['content']


        if title:
            metadata['title']=title
        if note_type:
            metadata['type']=note_type
        if tags is not None:
            metadata['tags']=tags
        if content is not None:
            old_content=content

        metadata['updated_at']=datetime.now().isoformat()

        # 错误记录：file_path 只写在 self.index 里，_build_markdown 时尚未写入 .md 的 YAML；
        # 若用 note['metadata']['file_path'] 会 KeyError。路径一律从索引取，再写回 metadata。
        file_path=self.index[note_id]['file_path']
        metadata['file_path']=file_path

        md_content=self._build_markdown(metadata,old_content) #构建新的 Markdown 内容
        with open(file_path,'w',encoding='utf-8') as f:
            f.write(md_content)

        self.index[note_id]=metadata
        self._save_index()

        return f"笔记已更新：{metadata['title']}"


    def _search_notes(self,
                      query:str,
                      limit:int=10,

                      note_type:Optional[str]=None,
                      tags:Optional[List[str]]=None)->List[Dict]:
        """搜索笔记

            Args:
                query: 搜索关键词
                limit: 返回数量限制
                note_type: 按类型过滤(可选)
                tags: 按标签过滤(可选)

            Returns:
                List[Dict]: 匹配的笔记列表
        """
        results=[]
        query_lower=query.lower()

        for note_id ,metadata in self.index.items():

            if note_type and metadata.get('type') != note_type:
                continue

            if tags:
                note_tags=set(metadata.get('tags',[]))
                if not note_tags.intersection(tags): #使用 set.intersection() 检查是否有交集
                    continue

            try:
                note=self._read_note(note_id)
                content=note['content']
                title=metadata.get('title','')

                if query_lower in title.lower() or query_lower in content.lower():

                    results.append({
                        'note_id':note_id,
                        'title':title,
                        'type':metadata.get('type'),
                        'tags':metadata.get('tags',[]),
                        'content':content,
                        'updated_by':metadata.get('updated_at')
                    })

            except Exception as e:
                print(f'[Warning] 读取笔记{note_id} 失败：{e}')
                continue
        results.sort(key=lambda x:x['updated_by'],reverse=True)
        return results[:limit]

    def _list_notes(self,
                    note_type:Optional[str]=None,
                    tags:Optional[List[str]]=None,
                    limit:int=20)->List[Dict]:
        """列出笔记(按更新时间倒序)

            Args:
                note_type: 按类型过滤(可选)
                tags: 按标签过滤(可选)
                limit: 返回数量限制

            Returns:
                List[Dict]: 笔记元数据列表
        """
        results=[]
        for note_id ,metadata in self.index.items():
            if note_type and metadata.get('type') != note_type:
                continue

            if tags:
                note_tags=set(metadata.get('tags',[]))
                if not note_tags.intersection(tags):
                    continue

            results.append(metadata)
        results.sort(key=lambda x:x.get('updated_at',''),reverse=True)
        return results[:limit]

    def _summary(self)->Dict[str,Any]:
        """生成笔记摘要统计

            Returns:
                Dict: 统计信息
        """
        total_count=len(self.index)
        type_counts={}
        for metadata in self.index.values():
            note_type=metadata.get('type','general')
            type_counts[note_type]=type_counts.get(note_type,0)+1

        recent_notes=sorted(self.index.values(),key=lambda x:x.get('updated_at',''),
                            reverse=True)[:5]

        return {
            'total_count':total_count,
            'type_distribution':type_counts,
            'recent_notes':[
                {
                    'id':note['id'],
                    'title':note.get('title',''),
                    'type':note.get('type'),
                    'updated_at':note.get('updated_at')

                }
                for note in recent_notes
            ]
        }

    def _delete_note(self,note_id:str)->str:
        """删除笔记

            Args:
                note_id: 笔记ID

            Returns:
                str: 操作结果消息
        """
        if note_id not in self.index:
            raise ValueError(f'笔记不存在：{note_id}')
        file_path=self.index[note_id]['file_path']


        if os.path.exists(file_path):
            os.remove(file_path)

        title=self.index[note_id].get('title',note_id)
        del self.index[note_id]
        self._save_index()

        return f'笔记已删除：{title}'
