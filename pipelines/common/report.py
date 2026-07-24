"""
对比报告打印 —— 表格汇总 + 分条详情。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .result import RunResult


def _status(r: RunResult) -> str:
    if r.error:
        return "ERR"
    if not (r.answer or "").strip():
        return "EMPTY"
    return "OK"


def _preview(text: str, limit: int = 48) -> str:
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return "-"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_results_table(results: Sequence[RunResult]) -> str:
    """并排指标表（等宽，适合终端）。"""
    headers = ("pipeline", "status", "sec", "steps", "tools", "answer")
    rows = [
        (
            r.pipeline,
            _status(r),
            f"{r.elapsed_sec:.1f}",
            str(r.steps),
            str(r.tool_calls),
            _preview(r.answer if not r.error else r.error, 40),
        )
        for r in results
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: Sequence[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def print_result_detail(r: RunResult, answer_limit: int = 200) -> None:
    """单条详情（答案截断；完整过程在 raw_trace）。"""
    print(f"  · [{r.pipeline}] {_status(r)}  {r.elapsed_sec:.1f}s  "
          f"steps≈{r.steps}  tools≈{r.tool_calls}")
    if r.error:
        print(f"    error : {r.error}")
    else:
        print(f"    answer: {_preview(r.answer, answer_limit)}")


def print_case_report(
    case_id: str,
    note: str,
    question: str,
    results: List[RunResult],
    *,
    show_detail: bool = True,
) -> None:
    """一道题的完整输出：表头 + 汇总表 + 可选详情。"""
    print()
    print("=" * 72)
    print(f" CASE  {case_id}")
    print(f" note  {note}")
    print(f" Q     {question}")
    print("-" * 72)
    print(format_results_table(results))
    if show_detail:
        print("-" * 72)
        print(" details")
        for r in results:
            print_result_detail(r)
    print("=" * 72)


def print_pair(case_id: str, note: str, question: str, results: List[RunResult]) -> None:
    """兼容旧调用名。"""
    print_case_report(case_id, note, question, results, show_detail=True)


def print_run_line(spec_tag: str, case_id: str, r: RunResult) -> None:
    """跑完一条后的一行进度。"""
    print(
        f"  [done] {spec_tag} | {case_id} | {_status(r)} | "
        f"{r.elapsed_sec:.1f}s | steps={r.steps} tools={r.tool_calls}"
    )


def print_final_summary(case_results: Dict[str, List[RunResult]]) -> None:
    """全部 case 结束后的总表。"""
    print()
    print("#" * 72)
    print(" SUMMARY")
    print("#" * 72)
    if not case_results:
        print("(no results)")
        return

    # 展平：一行一个 (case, pipeline)
    flat: List[RunResult] = []
    labels: List[str] = []
    for case_id, results in case_results.items():
        for r in results:
            flat.append(r)
            labels.append(case_id)

    headers = ("case", "pipeline", "status", "sec", "steps", "tools")
    rows = []
    for case_id, r in zip(labels, flat):
        rows.append(
            (
                case_id,
                r.pipeline,
                _status(r),
                f"{r.elapsed_sec:.1f}",
                str(r.steps),
                str(r.tool_calls),
            )
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: Sequence[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))

    ok = sum(1 for r in flat if _status(r) == "OK")
    err = sum(1 for r in flat if _status(r) == "ERR")
    empty = sum(1 for r in flat if _status(r) == "EMPTY")
    print("-" * 72)
    print(f" total={len(flat)}  OK={ok}  ERR={err}  EMPTY={empty}")
    print("#" * 72)
