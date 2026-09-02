"""Phase1 粗筛规则小测（不打 LLM）"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.load_cases import load_cases_jsonl
from eval.suspected_rules import score_suspected


def test_refuse_not_suspected():
    r = score_suspected(
        "很抱歉，我无法实时查询今天的武汉天气，建议使用天气APP。",
        tools_used=0,
    )
    assert r["suspected_hallucination"] == 0, r


def test_realtime_fabricate_suspected():
    r = score_suspected(
        "通过查询实时天气数据，今天（2025年4月30日）武汉气温20-25℃，AQI为75。",
        tools_used=0,
    )
    assert r["suspected_hallucination"] == 1, r


def test_tools_used_clears():
    r = score_suspected("今天气温35℃。", tools_used=2)
    assert r["suspected_hallucination"] == 0, r


def test_load_realtime_cases():
    path = _ROOT / "eval" / "cases_realtime_v1.jsonl"
    cases = load_cases_jsonl(path)
    assert len(cases) >= 15, len(cases)
    assert all("question" in c and "id" in c for c in cases)
    one = load_cases_jsonl(path, case_id="rt_weather_wuhan")
    assert len(one) == 1 and "武汉" in one[0]["question"]


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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
