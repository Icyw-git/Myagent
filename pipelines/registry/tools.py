"""
工具轴装配

产出一份与范式无关的「工具清单」，再分别灌进：
- hello_agents.ToolRegistry（simple / react）
- ReAct.ToolExecutor（hybrid）
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from ReAct import ToolExecutor, search

# (name, description, func)
ToolFn = Tuple[str, str, Callable[[str], str]]


def _calculate(expression: str) -> str:
    """计算器：入参是表达式字符串，和 search 一样走 register_function 风格。"""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


_CATALOG: Dict[str, ToolFn] = {
    "search": (
        "search",
        "一个搜索工具，可以根据用户的查询返回相关的搜索结果。",
        search,
    ),
    "calc": (
        "calculator",
        "一个计算器工具，输入数学表达式（如 15*8+32），返回计算结果。",
        _calculate,
    ),
}


def list_tool_fns(kind: str) -> List[ToolFn]:
    """按工具轴枚举展开成函数列表。"""
    if kind == "none":
        return []
    if kind == "search":
        return [_CATALOG["search"]]
    if kind == "calc":
        return [_CATALOG["calc"]]
    if kind == "search+calc":
        return [_CATALOG["search"], _CATALOG["calc"]]
    if kind == "bench":
        # 能力评测：TicketDesk + calculator（不挂联网 search，结果可复现）
        from eval.bench.env_ticket import get_active_desk, list_ticket_tool_fns

        def _bench_calculate(expression: str) -> str:
            get_active_desk().note_tool_call()
            return _calculate(expression)

        calc_fn: ToolFn = (
            "calculator",
            "一个计算器工具，输入数学表达式（如 15*8+32），返回计算结果。",
            _bench_calculate,
        )
        return list(list_ticket_tool_fns()) + [calc_fn]
    raise ValueError(f"未知 tools={kind}")


def build_hello_registry(kind: str):
    """给 MySimpleAgent / MyReActAgent 用的 ToolRegistry。"""
    from hello_agents import ToolRegistry

    registry = ToolRegistry()
    for name, desc, func in list_tool_fns(kind):
        registry.register_function(name, desc, func)
    return registry


def build_tool_executor(kind: str) -> ToolExecutor:
    """给 HybridAgent 用的 ToolExecutor。"""
    executor = ToolExecutor()
    for name, desc, func in list_tool_fns(kind):
        executor.register_tool(name, desc, func)
    return executor
