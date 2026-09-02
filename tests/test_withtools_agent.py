# test_withtools_agent.py
# ContextToolsAgent 离线冒烟：note 项目笔记工具 + terminal 即时上下文 + 笔记背景注入
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_withtools_agent.py
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory_src import MemoryConfig, MemoryTool
from NoteTool import NoteTool
from TerminalTool import TerminalTool
from my_react_agent_withtools import _note_fn, _terminal_fn, ContextToolsAgent
from contextbase import ContextConfig


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _DummyLLM:
    """离线：不调真 LLM，收到任何 prompt 都直接 Finish。"""

    def invoke(self, messages, **kwargs):
        return "Thought: 已知答案\nAction: Finish[武汉东湖]"  # type: ignore[no-any-return]


def _make_agent(tmp: str, inject_notes_context: bool = True) -> ContextToolsAgent:
    """构造只依赖临时目录的 ContextToolsAgent（不联网）。"""
    from hello_agents import ToolRegistry

    cfg = ContextConfig(
        max_tokens=1500,
        enable_compression=False,
        reserve_ratio=0.0,
        relevance_weight=0.7,
        recency_weight=0.3,
    )
    note_tool = NoteTool(workspace=os.path.join(tmp, "notes"))
    terminal_tool = TerminalTool(workspace=os.path.join(tmp, "term"))
    # 仅 working 记忆：内存检索，不依赖 Qdrant/网络（与 test_context_aware_agent --memory 一致）
    mem_tool = MemoryTool(
        user_id=f"test_withtools_{os.getpid()}",
        memory_config=MemoryConfig(),
        memory_types=["working"],
    )
    agent = ContextToolsAgent(
        name="工具上下文助手",
        llm=_DummyLLM(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
        max_steps=3,
        system_prompt="你是带项目笔记和终端能力的 ReAct 助手。",
        context_config=cfg,
        memory_tool=mem_tool,
        note_tool=note_tool,
        terminal_tool=terminal_tool,
        inject_notes_context=inject_notes_context,
    )
    # 避免 custom 包 relevance=0.5 触发 SentenceTransformer 加载
    agent.context_builder._calculate_relevance = lambda c, q: 0.85
    return agent


def test_note_tool_active() -> None:
    print("=== 1) note 工具主动调用：create → search（结构化长期项目记忆） ===")
    with tempfile.TemporaryDirectory(prefix="withtools_note_") as tmp:
        agent = _make_agent(tmp)
        note_fn = _note_fn(agent.note_tool)

        ret = json.loads(
            note_fn(
                '{"action":"create","title":"东湖项目","content":"本周目标：完成东湖天气接入","note_type":"task_state","tags":["weather"]}'
            )
        )
        print(f"create -> {ret}")
        _assert(ret.get("ok") and ret.get("note_id"), "create 应返回 note_id")

        hits = json.loads(note_fn('{"action":"search","query":"东湖","limit":5}'))
        print(f"search -> 命中 {hits.get('count')} 条")
        _assert(hits.get("count", 0) >= 1, "search 东湖 应命中刚建的笔记")
        _assert("天气接入" in hits["results"][0]["content"], "命中正文应含天气接入")

        # 非 JSON 入参回退为 search 查询
        fallback = json.loads(note_fn("东湖"))
        _assert(fallback.get("count", 0) >= 1, "非 JSON 入参应回退为 search")

        # 笔记已注册进 registry
        names = set(getattr(agent.tool_registry, "_tools", {}) or {}) | set(
            getattr(agent.tool_registry, "_functions", {}) or {}
        )
        _assert("note" in names, "note 工具应注册进 tool_registry")
        print("通过\n")


def test_terminal_tool_active() -> None:
    print("=== 2) terminal 工具主动调用：echo 即时上下文 ===")
    with tempfile.TemporaryDirectory(prefix="withtools_term_") as tmp:
        agent = _make_agent(tmp)
        terminal_fn = _terminal_fn(agent.terminal_tool)

        out = terminal_fn("echo hello_withtools")
        print(f"terminal -> {out!r}")
        _assert("hello_withtools" in out, "echo 输出应含 hello_withtools")

        names = set(getattr(agent.tool_registry, "_tools", {}) or {}) | set(
            getattr(agent.tool_registry, "_functions", {}) or {}
        )
        _assert("terminal" in names, "terminal 工具应注册进 tool_registry")
        print("通过\n")


def test_notes_injected_to_background() -> None:
    print("=== 3) 被动注入：相关项目笔记进入背景 [Context] ===")
    with tempfile.TemporaryDirectory(prefix="withtools_inject_") as tmp:
        agent = _make_agent(tmp)

        # 先写入一条相关笔记
        note_fn = _note_fn(agent.note_tool)
        note_fn(
            '{"action":"create","title":"服务器信息","content":"部署服务器IP: 10.0.0.8","note_type":"reference","tags":["ops"]}'
        )

        # 模拟 run 前记录当前 task，再走 build 背景
        agent._last_input_text = "服务器部署在哪里？"
        packets = agent._build_custom_context("用户偏好：回答用中文。")

        note_packets = [p for p in packets if (p.metadata or {}).get("type") == "note_context"]
        print(f"packets 中 note_context 数量: {len(note_packets)}")
        _assert(len(note_packets) >= 1, "相关笔记应注入背景")

        content = note_packets[0].content
        print(f"注入内容: {content}")
        _assert("10.0.0.8" in content, "注入的笔记内容应含服务器 IP")

        # 完整 run 应不炸，并命中笔记背景
        answer = agent.run("服务器部署在哪里？")
        print(f"agent.run 返回: {answer}")
        _assert(bool(answer and answer.strip()), "run 应返回非空答案")

        # 关闭注入开关后不再出现 note_context
        agent2 = _make_agent(tmp, inject_notes_context=False)
        agent2._last_input_text = "服务器部署在哪里？"
        packets2 = agent2._build_custom_context("用户偏好：回答用中文。")
        note_packets2 = [p for p in packets2 if (p.metadata or {}).get("type") == "note_context"]
        _assert(len(note_packets2) == 0, "inject_notes_context=False 时不应注入笔记")
        print("通过\n")


def main() -> None:
    test_note_tool_active()
    test_terminal_tool_active()
    test_notes_injected_to_background()
    print("全部 ContextToolsAgent 离线冒烟测试通过。")


if __name__ == "__main__":
    main()