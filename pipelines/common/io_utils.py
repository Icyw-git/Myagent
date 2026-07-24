"""
跑题时的小工具：截 stdout、从日志粗估 steps/tool_calls。

不改 Agent 内部实现，只在外面数打印出来的标记。
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from typing import Iterator, Tuple


@contextmanager
def capture_stdout() -> Iterator[io.StringIO]:
    """临时劫持 stdout，把 Agent 里的 print 收进 buffer。"""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def estimate_metrics(trace: str, paradigm: str) -> Tuple[int, int]:
    """从日志里粗估步数和工具调用次数（只求对比相对高低，不求绝对精确）。

    paradigm: simple / react / hybrid / plan / tot / reflection（也兼容旧标签 A/B）
    """
    steps = 0
    tools = 0
    label = paradigm.lower()

    # 旧 compare 曾用 A=hybrid、B=react
    if label in ("a", "hybrid"):
        steps = trace.count("正在执行步骤")
        tools = trace.count("正在执行")
        tools = max(0, tools - steps)
    elif label == "plan":
        steps = trace.count("正在执行步骤")
        tools = 0
    elif label == "reflection":
        steps = trace.count("轮反思")
        tools = 0
    elif label == "tot":
        steps = max(1, trace.count("正在处理"))
        tools = 0
    elif label in ("b", "react", "simple"):
        steps = trace.count("--- 第")
        if steps == 0 and "正在处理" in trace:
            steps = 1
        tools = sum(
            1
            for line in trace.splitlines()
            if line.strip().startswith("Action:") and "Finish" not in line
        )
        if tools == 0:
            tools = max(0, trace.count("正在执行") - steps)
    else:
        steps = (
            trace.count("--- 第")
            + trace.count("正在执行步骤")
            + trace.count("轮反思")
        )
        tools = max(0, trace.count("正在执行") - trace.count("正在执行步骤"))

    return steps, tools
