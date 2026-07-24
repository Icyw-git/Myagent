"""
三维 Pipeline 工厂

用法：
    from pipelines.factory import build_pipeline, run_pipeline
    from pipelines.specs import PipelineSpec, PRESETS

    handle = build_pipeline(PRESETS["b"])
    result = run_pipeline("你好", handle)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from pipelines.common.io_utils import capture_stdout, estimate_metrics
from pipelines.common.result import RunResult
from pipelines.registry.memory import build_memory
from pipelines.registry.paradigms import build_paradigm
from pipelines.specs import PRESETS, PipelineSpec


@dataclass
class PipelineHandle:
    """装配结果：带着 spec，方便报告打标签。"""

    spec: PipelineSpec
    agent: Any

    def run(self, question: str) -> str:
        return self.agent.run(question)


def build_pipeline(spec: PipelineSpec) -> PipelineHandle:
    """按三维规格装配一条 pipeline。"""
    memory = build_memory(spec.memory)
    agent = build_paradigm(spec, memory=memory)
    return PipelineHandle(spec=spec, agent=agent)


def build_preset(name: str) -> PipelineHandle:
    """按预设名装配，例如 a / b。"""
    key = name.lower()
    if key not in PRESETS:
        raise KeyError(f"未知 preset={name}，可选 {list(PRESETS)}")
    return build_pipeline(PRESETS[key])


def run_pipeline(
    question: str,
    handle: Optional[PipelineHandle] = None,
    spec: Optional[PipelineSpec] = None,
) -> RunResult:
    """跑一道题。可传现成 handle，或传 spec 现场 build。"""
    if handle is None:
        if spec is None:
            raise ValueError("必须提供 handle 或 spec")
        handle = build_pipeline(spec)

    t0 = time.time()
    answer = ""
    error = ""
    trace = ""

    try:
        with capture_stdout() as buf:
            raw = handle.agent.run(question)
            trace = buf.getvalue()
        answer = raw if raw is not None else ""
    except Exception as e:
        error = str(e)
        answer = ""

    elapsed = time.time() - t0
    steps, tools = estimate_metrics(trace, handle.spec.paradigm)
    return RunResult(
        pipeline=handle.spec.tag,
        question=question,
        answer=answer or "",
        elapsed_sec=elapsed,
        steps=steps,
        tool_calls=tools,
        error=error,
        raw_trace=trace,
    )
