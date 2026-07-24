"""
Pipeline A —— 预设 a 的薄封装（hybrid | search | off）

实际装配走 pipelines.factory，这里只保留旧 import 路径。
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from pipelines.factory import PipelineHandle, build_preset, run_pipeline
from pipelines.common.result import RunResult


def build_agent(max_depth: int = 2) -> PipelineHandle:
    """返回 Handle（带 .agent / .run），兼容旧「build 一次多题复用」。"""
    # max_depth 已在工厂 hybrid 里写死为 2；参数保留以免旧调用报错
    _ = max_depth
    return build_preset("a")


def run(question: str, agent: PipelineHandle | None = None) -> RunResult:
    handle = agent or build_agent()
    return run_pipeline(question, handle=handle)


if __name__ == "__main__":
    q = "用一句话介绍武汉。"
    r = run(q)
    print(f"[A/{r.pipeline}] {r.elapsed_sec:.1f}s steps={r.steps} tools={r.tool_calls}")
    print(r.answer or r.error)
