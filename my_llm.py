'''
在llm_client基础上进行扩展，支持本地与更多平台和自动识别，
'''
from typing import Optional

from openai import OpenAI

from llm_client import Myagent
import os


class My_llm(Myagent):
    def __init__(self,
                 model:Optional[str] = None,
                 api_key:Optional[str] = None,
                 base_url:Optional[str] = None,
                 provider:Optional[str] = 'auto',
                 **kwargs


                 ):
        if provider =='modelscope':
            print('正在使用自定义的modelscope平台')

            self.provider='modelscope'

            self.base_url=base_url or os.getenv('MODELSCOPE_BASE_URL')
            self.api_key=api_key or os.getenv('MODELSCOPE_API_KEY')

            if not api_key:
                raise ValueError('没有找到modelscope的api_key')

            self.model_id=model or os.getenv('MODELSCOPE_MODEL_ID')
            self.temperature=kwargs.get('temperature',0.7)
            self.max_tokens=kwargs.get('max_tokens',100)
            self.timeout=kwargs.get('timeout',10)

            self._client=OpenAI(api_key=self.api_key,base_url=self.base_url,timeout=self.timeout)

        else:
            super().__init__(model_id=model,api_key=api_key,base_url=base_url,**kwargs)

