# test_context_aware_multiturn.py
# 多轮对话：覆盖 ContextAwareAgent 的上下文与记忆能力
#
# 运行（项目根目录，需 .env LLM）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_context_aware_multiturn.py
#
# ┌──────┬────────────────────────────────────────┬─────────────────────────────┐
# │ 轮次 │ 用户话                                 │ 考察能力                    │
# ├──────┼────────────────────────────────────────┼─────────────────────────────┤
# │ 预热 │ （代码写入住址）                       │ MemoryTool 存储 / 被动检索  │
# │  1   │ 你的回答风格？                         │ custom_context（静态背景）  │
# │  2   │ 我住在哪个小区？                       │ 被动记忆 build→system       │
# │  3   │ 请记住我喜欢美式咖啡                   │ 主动 memory[add]            │
# │  4   │ 刚才记的饮品是什么？                   │ _history 多轮会话           │
# │  5   │ 算 15*8                                │ ReAct 工具轨 / current 隔离 │
# │  6   │ 改喝拿铁，请记下                       │ 主动 memory 更新            │
# │  7   │ 综合：住址+饮品+风格                   │ 被动+会话+静态 综合召回     │
# └──────┴────────────────────────────────────────┴─────────────────────────────┘
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class Turn:
    """一轮对话定义。"""
    id: str
    user: str
    capability: str  # 给人看的考察点
    expect: List[str]  # 答案里至少出现其一（软断言）
    forbid: List[str] = field(default_factory=list)  # 答案里不应出现
    extra_check: Optional[Callable] = None  # 额外断言 (agent, answer) -> None


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _any_in(text: str, keys: List[str]) -> bool:
    return any(k in text for k in keys)


def _build_agent():
    from dotenv import load_dotenv
    from hello_agents import HelloAgentsLLM
    from memory_src import MemoryConfig, MemoryTool
    from my_react_agent_context import ContextAwareAgent
    from pipelines.registry.tools import build_hello_registry
    from contextbase import ContextConfig

    load_dotenv()
    llm = HelloAgentsLLM()
    mem_tool = MemoryTool(
        user_id="multiturn_ctx_test",
        memory_config=MemoryConfig(),
        memory_types=["working"],
    )
    cfg = ContextConfig(
        max_tokens=2800,
        enable_compression=False,
        reserve_ratio=0.0,
        max_history=10,
    )
    agent = ContextAwareAgent(
        name="多轮上下文助手",
        llm=llm,
        tool_registry=build_hello_registry("calc"),  # calc + 自动挂 memory
        max_steps=6,
        system_prompt=(
            "你是带长期记忆的多轮助手。"
            "用户明确要求「记住/更新」时，用 memory 工具 add（JSON）；"
            "需要计算时用 calculator；不要编造未记录事实；中文简洁回答。"
        ),
        context_config=cfg,
        custom_context="用户偏好：回答务必简洁，每条不超过两句话。",
        memory_tool=mem_tool,
    )
    agent.context_builder._calculate_relevance = lambda c, q: 0.85
    return agent, mem_tool


TURNS: List[Turn] = [
    Turn(
        id="T1",
        user="用一句话说明：你给我的回答应该是什么风格？",
        capability="custom_context 静态背景",
        expect=["简洁", "简短", "两句话", "精简"],
    ),
    Turn(
        id="T2",
        user="不要编造。根据你已知的长期记忆：我住在哪个小区？说出小区名即可。",
        capability="被动记忆（build 时 search_items→system [Context]）",
        expect=["紫荆花园"],
    ),
    Turn(
        id="T3",
        user=(
            "请记住：我平时喜欢喝美式咖啡。"
            "请用 memory 工具写入后 Finish[已记住]。"
            '示例：memory[{"action":"add","content":"用户喜欢喝美式咖啡","memory_type":"working","importance":0.9}]'
        ),
        capability="主动 memory[add]",
        expect=["记住", "已记住", "美式", "好的"],
    ),
    Turn(
        id="T4",
        user="刚才对话里我让你记住的饮品是什么？只答饮品名。",
        capability="_history 多轮会话（上轮 user/assistant 进 build）",
        expect=["美式"],
    ),
    Turn(
        id="T5",
        user="帮我算 15*8，只给最终数字。",
        capability="ReAct 工具轨（calculator）；current_history 每轮清空",
        expect=["120"],
    ),
    Turn(
        id="T6",
        user=(
            "我改喝拿铁了。请用 memory 记下新偏好，然后 Finish[已更新]。"
            'memory[{"action":"add","content":"用户现在喜欢喝拿铁","memory_type":"working","importance":0.9}]'
        ),
        capability="主动 memory 更新（新事实写入同一库）",
        expect=["拿铁", "更新", "记住", "已更新"],
    ),
    Turn(
        id="T7",
        user="综合回答：我住哪个小区？现在喜欢喝什么？你的回答风格应该怎样？各用很短一句。",
        capability="被动记忆 + _history + custom_context 综合（饮品可能命中多条 working 记忆）",
        expect=["紫荆花园", "简洁"],
        # working 不自动删旧条，模型可能答「拿铁」或「美式」；优先期望拿铁
        extra_check=lambda agent, ans: _assert(
            _any_in(ans, ["拿铁", "美式", "咖啡"]),
            f"T7 应提到饮品偏好，实际={ans[:160]}",
        ),
    ),
]


def _warmup(mem_tool) -> None:
    fact = "用户住在武汉市武昌区紫荆花园7栋502室"
    ret = mem_tool.run(
        {
            "action": "add",
            "content": fact,
            "memory_type": "working",
            "importance": 0.95,
        }
    )
    hits = mem_tool.search_items("住在哪个小区", limit=3, min_importance=0.3)
    print(f"[预热] {ret}")
    _assert(any("紫荆花园" in (h.content or "") for h in hits), "预热记忆应可检索")
    print("[预热] 被动检索 OK\n")


def run_multiturn_dialogue() -> None:
    agent, mem_tool = _build_agent()
    _warmup(mem_tool)

    transcript: List[dict] = []
    print("=" * 60)
    print("多轮对话开始（同一 Agent 实例，_history 跨轮累积）")
    print("=" * 60)

    for turn in TURNS:
        print(f"\n{'─' * 60}")
        print(f"[{turn.id}] 考察：{turn.capability}")
        print(f"用户：{turn.user}")
        print(f"{'─' * 60}")

        answer = agent.run(turn.user)
        hist_len = len(agent.get_history())
        bg_preview = agent.context_builder.build(
            turn.user,
            agent._history,
            agent.system_prompt,
            agent._build_custom_context(agent.custom_context),
            include_task_sections=False,
        )
        mem_hits = len(
            mem_tool.search_items(turn.user[:20] or "用户", limit=5, min_importance=0.3)
        )

        print(f"\n助手：{answer}")
        print(f"[状态] _history={hist_len} 条 | 本轮被动检索命中≈{mem_hits} | system 背景 {len(bg_preview)} 字")

        _assert(bool(answer.strip()), f"{turn.id} 答案不应为空")
        if turn.expect:
            _assert(
                _any_in(answer, turn.expect),
                f"{turn.id} 期望含 {turn.expect} 之一，实际={answer[:160]}",
            )
        for bad in turn.forbid:
            _assert(bad not in answer, f"{turn.id} 不应含「{bad}」，实际={answer[:160]}")

        if turn.extra_check:
            turn.extra_check(agent, answer)

        transcript.append(
            {
                "id": turn.id,
                "capability": turn.capability,
                "user": turn.user,
                "assistant": answer,
                "history_len": hist_len,
            }
        )

    print("\n" + "=" * 60)
    print("对话摘要")
    print("=" * 60)
    for row in transcript:
        print(f"{row['id']} ({row['history_len']} hist): {row['assistant'][:80]}...")

    # T6 后库中应有拿铁（美式可能仍在，working 不自动覆盖）
    store = " ".join(h.content for h in mem_tool.search_items("喝什么", limit=5, min_importance=0.1))
    _assert("拿铁" in store, f"长期库应含拿铁，store={store[:120]}")
    _assert("紫荆花园" in store, f"长期库应含住址，store={store[:120]}")
    print("\n全部多轮测试通过。")


if __name__ == "__main__":
    run_multiturn_dialogue()
