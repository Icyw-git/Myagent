from ReAct import Myagent
from dotenv import load_dotenv
import os
load_dotenv()
def test_init_with_env_vars():
    myagent = Myagent()

    assert myagent.api_key==os.getenv("LLM-API-KEY")
    assert myagent.base_url==os.getenv("LLM-BASE-URL")
    assert myagent.timeout==int(os.getenv("TIMEOUT"))


def test_init_with_args():
    myagent = Myagent(api_key='24324',base_url='http://localhost:8000',timeout=1)
    assert myagent.api_key=='24324'
    assert myagent.base_url=='http://localhost:8000'
    assert myagent.timeout==1


if __name__ == '__main__':
    test_init_with_env_vars()


