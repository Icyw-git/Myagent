"""
能力评测确定性打分 —— 二元 pass（τ-bench / SWE-bench 思路：自动检查器）。

评分原则（相对旧版）：
  1. 以终态 / 答案为主：db_equals、answer_*、exact_number、must_refuse
  2. min_tool_calls 使用「环境侧真实调用次数」，不用 stdout 粗估
  3. must_refuse 覆盖「不存在 / 查不到」等常见拒答说法

支持的 check.type：
  - db_equals:       path + value（对照 TicketDesk 终态）
  - answer_contains: value 子串（大小写敏感）
  - exact_number:    答案中抽出的数字（优先最后一个）等于 value
  - min_tool_calls:  tool_calls >= value（应为 env.tool_call_count）
  - must_refuse:     答案像在拒绝 / 说明办不到
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_REFUSE_RE = re.compile(
    r"无法|不能|不可|拒绝|不允许|policy|已关闭|抱歉|对不起|"
    r"不存在|没有找到|找不到|查不到|查询不到|无此工单|未找到",
    re.I,
)


def _get_path(root: Dict[str, Any], path: str) -> Any:
    cur: Any = root
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _extract_numbers(text: str) -> List[float]:
    nums = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", text or ""):
        try:
            nums.append(float(m.group()))
        except ValueError:
            continue
    return nums


def score_checks(
    *,
    answer: str,
    tool_calls: int,
    env_snapshot: Optional[Dict[str, Any]],
    checks: List[Dict[str, Any]],
    error: str = "",
) -> Dict[str, Any]:
    """
    返回:
      pass: 0/1
      check_results: [{type, ok, detail}, ...]
    """
    results = []
    all_ok = True
    snap = env_snapshot or {}
    ans = answer or ""

    if error:
        for c in checks:
            results.append(
                {"type": c.get("type"), "ok": False, "detail": f"run_error: {error}"}
            )
        return {"pass": 0, "check_results": results}

    for c in checks:
        ctype = c.get("type")
        ok = False
        detail = ""

        if ctype == "db_equals":
            path = c["path"]
            expected = c["value"]
            got = _get_path(snap, path)
            ok = got == expected
            detail = f"path={path} expected={expected!r} got={got!r}"

        elif ctype == "answer_contains":
            needle = str(c["value"])
            ok = needle in ans
            detail = f"need={needle!r}"

        elif ctype == "exact_number":
            expected = float(c["value"])
            nums = _extract_numbers(ans)
            ok = bool(nums) and abs(nums[-1] - expected) < 1e-6
            detail = f"expected={expected} numbers={nums}"

        elif ctype == "min_tool_calls":
            need = int(c["value"])
            got = int(tool_calls or 0)
            ok = got >= need
            detail = f"tool_calls={got} need>={need} (env-counted)"

        elif ctype == "must_refuse":
            ok = bool(_REFUSE_RE.search(ans))
            detail = "refuse_pattern"

        else:
            ok = False
            detail = f"unknown check type: {ctype}"

        results.append({"type": ctype, "ok": bool(ok), "detail": detail})
        if not ok:
            all_ok = False

    return {"pass": 1 if all_ok and checks else 0, "check_results": results}
