"""
Pipeline 工厂 / 规格 / 工具轴 / 报告 —— 单元测试（不打真实 LLM）

运行（项目根目录）：
    D:\\Anaconda_envs\\envs\\aitest01_py310\\python.exe tests/test_pipelines.py
若已装 pytest：
    python -m pytest tests/test_pipelines.py -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.common.io_utils import estimate_metrics
from pipelines.common.report import (
    _status,
    format_results_table,
    print_case_report,
    print_final_summary,
)
from pipelines.common.result import RunResult
from pipelines.registry.memory import build_memory
from pipelines.registry.tools import list_tool_fns
from pipelines.specs import PRESETS, PipelineSpec


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---- specs ----

def test_spec_tag():
    s = PipelineSpec("react", "search+calc", "off")
    _assert(s.tag == "react|search+calc|off", s.tag)


def test_spec_rejects_bad_paradigm():
    try:
        PipelineSpec(paradigm="nope", tools="none", memory="off")
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass


def test_spec_rejects_bad_tools():
    try:
        PipelineSpec(paradigm="react", tools="web", memory="off")
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass


def test_presets_a_b():
    _assert(PRESETS["a"].paradigm == "hybrid", "preset a")
    _assert(PRESETS["b"].paradigm == "react", "preset b")
    _assert(PRESETS["a"].tools == "search", "preset a tools")
    _assert(PRESETS["b"].memory == "off", "preset b memory")


# ---- tools ----

def test_list_tool_fns_none():
    _assert(list_tool_fns("none") == [], "none")


def test_list_tool_fns_search_calc():
    names = [n for n, _, _ in list_tool_fns("search+calc")]
    _assert(names == ["search", "calculator"], str(names))


def test_list_tool_fns_calc_runs():
    _, _, fn = list_tool_fns("calc")[0]
    _assert(fn("1+2*3") == "7", fn("1+2*3"))


# ---- memory ----

def test_memory_off():
    _assert(build_memory("off") is None, "off")


def test_memory_working_ready():
    memory = build_memory("working")
    assert memory is not None
    assert "working" in memory.memory_manager.memory_types


def test_memory_not_ready():
    try:
        build_memory("episodic")
        raise AssertionError("应抛 NotImplementedError")
    except NotImplementedError:
        pass


# ---- metrics ----

def test_estimate_metrics_hybrid():
    trace = "正在执行步骤 1/2\n正在执行天气\n正在执行步骤 2/2\n"
    steps, tools = estimate_metrics(trace, "hybrid")
    _assert(steps == 2 and tools == 1, f"{steps},{tools}")


def test_estimate_metrics_react():
    trace = "--- 第1步 ---\nAction:search[武汉]\n--- 第2步 ---\nAction:Finish[ok]\n"
    steps, tools = estimate_metrics(trace, "react")
    _assert(steps == 2 and tools == 1, f"{steps},{tools}")


def test_estimate_metrics_plan_and_reflection():
    _assert(estimate_metrics("-> 正在执行步骤 1/3\n-> 正在执行步骤 2/3\n", "plan")[0] == 2, "plan")
    _assert(estimate_metrics("--- 第1轮反思 ---\n--- 第2轮反思 ---\n", "reflection")[0] == 2, "refl")


# ---- report ----

def _sample_results():
    return [
        RunResult("react|search|off", "q", "答案甲", 1.2, steps=3, tool_calls=1),
        RunResult("plan|none|off", "q", "", 0.5, steps=2, tool_calls=0, error="boom"),
    ]


def test_status_ok_err_empty():
    ok, err = _sample_results()
    _assert(_status(ok) == "OK", "OK")
    _assert(_status(err) == "ERR", "ERR")
    _assert(_status(RunResult("x", "q", "  ", 0.1)) == "EMPTY", "EMPTY")


def test_format_results_table_has_headers_and_rows():
    table = format_results_table(_sample_results())
    _assert("pipeline" in table, "header")
    _assert("react|search|off" in table, "row")
    _assert("ERR" in table and "OK" in table, "status")
    _assert(table.count("\n") >= 3, "lines")


def test_print_case_and_summary():
    results = _sample_results()
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        print_case_report("qa", "纯问答", "武汉有什么好玩的？", results)
        print_final_summary({"qa": results})
    finally:
        sys.stdout = old
    out = buf.getvalue()
    _assert("CASE  qa" in out, "case")
    _assert("SUMMARY" in out, "summary")
    _assert("OK=1" in out and "ERR=1" in out, out)


# ---- compare CLI 规格展开 ----

def test_specs_from_args_grid_and_preset():
    from pipelines import compare as compare_mod

    ns = SimpleNamespace(
        grid="paradigm",
        tools="none",
        memory="off",
        paradigm=None,
        only=None,
        preset="a,b",
    )
    specs = compare_mod._specs_from_args(ns)
    _assert(
        [s.paradigm for s in specs]
        == ["simple", "react", "hybrid", "plan", "tot", "reflection"],
        str([s.paradigm for s in specs]),
    )

    ns2 = SimpleNamespace(
        grid=None, tools="search", memory="off",
        paradigm="plan", only=None, preset="a,b",
    )
    _assert(compare_mod._specs_from_args(ns2)[0].tag == "plan|search|off", "plan")

    ns3 = SimpleNamespace(
        grid=None, tools="search", memory="off",
        paradigm=None, only="a", preset="a,b",
    )
    _assert(compare_mod._specs_from_args(ns3)[0].paradigm == "hybrid", "only a")


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print("-" * 40)
    print(f" total={len(tests)}  fail={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
