# test_reflection_agent.py
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_reflection_agent.py
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_reflection_agent import MyReflectionAgent, MyReflectionAgentplus

load_dotenv()
llm = HelloAgentsLLM()

# 使用默认通用提示词
general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

# 使用自定义代码生成提示词（类似第四章）
code_prompts = {
    "initial": "你是Python专家，请编写函数:{task}",
    "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
    "refine": "请根据反馈优化代码:\n任务:{task}\n反馈:{feedback}"
}
code_agent = MyReflectionAgent(
    name="我的代码生成助手",
    llm=llm,
    custom_prompts=code_prompts
)

# 测试使用
result = general_agent.run("写一篇关于人工智能发展历程的文章,要求结构清晰，内容丰富，字数不少于500字。")
print(f"最终结果: {result}")

# ========== MyReflectionAgentplus 测试（质量评分 + best_response 元组） ==========
print("\n=== 测试 MyReflectionAgentplus ===")
score_agent = MyReflectionAgentplus(
    name="我的评分反思助手",
    llm=llm,
    max_iterations=3,
)

# 评分模板为 0-10 分，quality_threshold 需与之对应（如 8.0）
result_plus = score_agent.run(
    "用三句话介绍机器学习的基本概念",
    quality_threshold=8.0,
)
print(f"评分反思最终结果: {result_plus}")
print(f"对话历史: {len(score_agent.get_history())} 条消息")

# 测试 _parse_score 解析
print("\n=== 测试 _parse_score ===")
assert score_agent._parse_score('{"score": 8.5, "reason": "ok"}') == 8.5
assert score_agent._parse_score("无效内容") == 0.0
print("_parse_score 测试通过")
