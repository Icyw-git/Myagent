import json
from hello_agents import ToolRegistry, HelloAgentsLLM
from memory_src import MemoryTool, MemoryConfig
from my_react_agent import MyReActAgent

mem_tool = MemoryTool(
    user_id="default_user",
    memory_config=MemoryConfig(),
    memory_types=["working", "episodic"],
)


def memory_fn(raw: str) -> str:
    raw = (raw or "").strip()
    try:
        params = json.loads(raw) if raw.startswith("{") else {"action": "search", "query": raw}
        return mem_tool.run(params)
    except Exception as e:
        return f"Memory error: {e}"


registry = ToolRegistry()
registry.register_function(
    "memory",
    '记忆工具。参数必须是 JSON，例如 '
    '{"action":"add","content":"用户住武汉","memory_type":"episodic","importance":0.8} '
    '或 {"action":"search","query":"住哪里","limit":3}',
    memory_fn,
)

agent = MyReActAgent(
    name="MyReActAgentWithMemory",
    llm=HelloAgentsLLM(),
    tool_registry=registry,
    max_steps=5,
)

agent.run("先记住我住在武汉，再告诉我我住哪。")
