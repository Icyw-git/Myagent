"""能力评测（bench）无 LLM 单测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.bench.env_ticket import TicketDesk, reset_active_desk
from eval.bench.load_bench_cases import load_bench_cases
from eval.bench.scorer import score_checks
from pipelines.registry.tools import list_tool_fns
from pipelines.specs import PipelineSpec


def test_ticket_policy_blocks_closed_update():
    desk = TicketDesk()
    out = json.loads(desk.tool_update_ticket('{"id":"T300","assignee":"赵六"}'))
    assert "error" in out
    assert desk.tickets["T300"]["assignee"] == "王芳"


def test_ticket_update_open():
    desk = TicketDesk()
    out = json.loads(desk.tool_update_ticket('{"id":"T100","assignee":"李雷"}'))
    assert out.get("ok") is True
    assert desk.tickets["T100"]["assignee"] == "李雷"


def test_reset_active_desk_setup():
    reset_active_desk({"tickets": {"T1": {"assignee": "A", "status": "open"}}})
    from eval.bench.env_ticket import get_active_desk

    assert get_active_desk().tickets["T1"]["assignee"] == "A"


def test_scorer_db_equals_pass():
    snap = {"tickets": {"T100": {"assignee": "李雷", "status": "open"}}}
    r = score_checks(
        answer="已改为李雷",
        tool_calls=1,
        env_snapshot=snap,
        checks=[
            {"type": "db_equals", "path": "tickets.T100.assignee", "value": "李雷"},
            {"type": "min_tool_calls", "value": 1},
        ],
    )
    assert r["pass"] == 1, r


def test_scorer_exact_number():
    r = score_checks(
        answer="计算结果是 152。",
        tool_calls=1,
        env_snapshot={},
        checks=[{"type": "exact_number", "value": 152}],
    )
    assert r["pass"] == 1, r


def test_scorer_must_refuse():
    r = score_checks(
        answer="抱歉，policy：已关闭工单不可修改。",
        tool_calls=1,
        env_snapshot={"tickets": {"T300": {"assignee": "王芳", "status": "closed"}}},
        checks=[
            {"type": "must_refuse"},
            {"type": "db_equals", "path": "tickets.T300.assignee", "value": "王芳"},
        ],
    )
    assert r["pass"] == 1, r


def test_scorer_must_refuse_missing():
    r = score_checks(
        answer="工单 T999 不存在，查询不到相关信息。",
        tool_calls=1,
        env_snapshot={},
        checks=[{"type": "must_refuse"}],
    )
    assert r["pass"] == 1, r


def test_env_tool_call_count():
    desk = TicketDesk()
    assert desk.tool_call_count == 0
    desk.tool_get_ticket("T100")
    desk.tool_get_ticket("T999")
    assert desk.tool_call_count == 2


def test_load_bench_cases_keeps_checks():
    path = _ROOT / "eval" / "bench" / "cases_v1.jsonl"
    cases = load_bench_cases(path)
    assert len(cases) >= 8
    one = load_bench_cases(path, case_id="tk_update_assignee")
    assert len(one) == 1
    assert one[0]["checks"]
    assert "question" in one[0]


def test_tools_bench_lists_ticket_and_calc():
    fns = list_tool_fns("bench")
    names = [n for n, _, _ in fns]
    assert "get_ticket" in names
    assert "update_ticket" in names
    assert "cancel_ticket" in names
    assert "calculator" in names


def test_spec_accepts_bench():
    s = PipelineSpec(paradigm="react", tools="bench", memory="off")
    assert s.tag == "react|bench|off"


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
    print(f" total={len(tests)}  fail={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
