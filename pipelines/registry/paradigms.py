"""
范式轴装配

已接通：simple | react | hybrid | plan | tot | reflection
- simple / react / hybrid：会吃工具轴
- plan / tot / reflection：当前实现不调外部工具，tools!=none 时仅打印提示
"""

from __future__ import annotations

from typing import Any

from pipelines.registry.tools import build_hello_registry, build_tool_executor
from pipelines.specs import PipelineSpec

# 会真正注册工具的范式
_TOOL_AWARE = frozenset({"simple", "react", "hybrid"})


def build_paradigm(spec: PipelineSpec, memory: Any = None):
    """
    按 paradigm 构造 Agent。
    memory 第一版未使用（memory 轴仅 off）；预留参数避免以后改签名。
    """
    if memory is not None:
        # 占位：以后把 MemoryTool 塞进 registry
        pass

    builders = {
        "simple": _build_simple,
        "react": _build_react,
        "hybrid": _build_hybrid,
        "plan": _build_plan,
        "tot": _build_tot,
        "reflection": _build_reflection,
    }
    if spec.paradigm not in builders:
        raise ValueError(f"未知 paradigm={spec.paradigm}")

    if spec.paradigm not in _TOOL_AWARE and spec.tools != "none":
        print(
            f"[注意] paradigm={spec.paradigm} 暂不使用工具，"
            f"tools={spec.tools} 将被忽略（对比工具轴请用 simple/react/hybrid）"
        )

    if spec.paradigm == "react":
        return _build_react(spec, memory=memory)
    return builders[spec.paradigm](spec)


def _build_simple(spec: PipelineSpec):
    from hello_agents import HelloAgentsLLM
    from my_simple_agent import MySimpleAgent

    llm = HelloAgentsLLM()
    registry = build_hello_registry(spec.tools)
    enable = spec.tools != "none"
    return MySimpleAgent(
        name=f"pipe_{spec.tag}",
        llm=llm,
        system_prompt="你是一个有用的AI助手。",
        tool_registry=registry if enable else None,
        enable_tool_calling=enable,
    )


def _build_react(spec: PipelineSpec, memory: Any = None):
    from hello_agents import HelloAgentsLLM
    from my_react_agent import MyReActAgent

    llm = HelloAgentsLLM()
    registry = build_hello_registry(spec.tools)
    if memory is not None:
        from my_react_agent_context import ContextAwareAgent

        return ContextAwareAgent(
            name=f"pipe_{spec.tag}",
            llm=llm,
            tool_registry=registry,
            max_steps=5,
            memory_tool=memory,
            register_memory_tool=True,
        )
    return MyReActAgent(
        name=f"pipe_{spec.tag}",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )


def _build_hybrid(spec: PipelineSpec):
    from HybridAgent import HybridAgent
    from llm_client import Myagent

    llm = Myagent()
    executor = build_tool_executor(spec.tools)
    # max_depth=2：对比时别递归太深
    return HybridAgent(llm, max_depth=2, tool_executor=executor)


def _build_plan(spec: PipelineSpec):
    """PlanAndSolve：只规划+逐步执行，不调 search/calc。"""
    from hello_agents import HelloAgentsLLM
    from my_planandsolve_agent import PlanAndSolveAgent

    llm = HelloAgentsLLM()
    return PlanAndSolveAgent(
        name=f"pipe_{spec.tag}",
        llm=llm,
        system_prompt="你是一个擅长拆解任务并逐步执行的AI助手。",
    )


def _build_tot(spec: PipelineSpec):
    """Tree-of-Thought：多分支打分选优。"""
    from hello_agents import HelloAgentsLLM
    from Tree_of_Thought import TreeofThought

    llm = HelloAgentsLLM()
    agent = TreeofThought(
        name=f"pipe_{spec.tag}",
        llm=llm,
        system_prompt="你是一个善于多路径思考的AI助手。",
    )
    # 包一层：对比时默认少分支、浅深度，省时间和费用
    return _TotRunner(agent)


def _build_reflection(spec: PipelineSpec):
    """Reflection：生成 → 反思/打分 → 改进（用 plus 版）。"""
    from hello_agents import HelloAgentsLLM
    from my_reflection_agent import MyReflectionAgentplus

    llm = HelloAgentsLLM()
    return MyReflectionAgentplus(
        name=f"pipe_{spec.tag}",
        llm=llm,
        system_prompt="你是一个会自我反思并改进答案的AI助手。",
        max_iterations=2,
    )


class _TotRunner:
    """给 ToT 设更省的默认 branches/max_depth，仍可用 run(..., branches=3) 覆盖。"""

    def __init__(self, agent):
        self._agent = agent

    def run(self, question: str, **kwargs):
        kwargs.setdefault("branches", 2)
        kwargs.setdefault("max_depth", 2)
        kwargs.setdefault("quality_threshold", 8.0)
        return self._agent.run(question, **kwargs)
