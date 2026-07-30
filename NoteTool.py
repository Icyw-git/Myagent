from importlib.metadata import metadata
from typing import List,Optional,Dict,Any,Tuple
from datetime import datetime
import yaml
import os

from click.termui import raw_terminal


class NoteTool:



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


        metadata['file_path']=file_path
        self.index[note_id]=metadata
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
        file_path=self.index[note_id]['file_path']

        with open(file_path,'r',encoding='utf-8') as f:
            raw_content=f.read()

        metadata,content=self._parse_markdown(raw_content)

        return {
            'metadata':metadata,
            'content':content
        }

    def _parse_markdown(self,raw_content:str)->Tuple[Dict,str]:
        """解析 Markdown 文件(分离 YAML 和正文)"""
        parts=raw_content.split('---\n',2)

        if len(parts)>=3:
            yaml_str=parts[1]
            content=parts[2].strip()
            metadata=yaml.safe_load(yaml_str)

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

        md_content=self._build_markdown(metadata,old_content)
        file_path=metadata['file_path']
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
                if not note_tags.intersection(tags):
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
