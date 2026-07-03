from hello_agents import SimpleAgent,HelloAgentsLLM

from dotenv import load_dotenv

load_dotenv()

llm=HelloAgentsLLM()

agent=SimpleAgent(
    name='AI助手',
    llm=llm,
    system_prompt='你是一个有用的AI助手，能够回答用户的问题并提供帮助。',

)

response=agent.run('你好，请介绍一下自己。')
print(response)