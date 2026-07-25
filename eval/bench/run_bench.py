"""
能力评测入口（τ-bench 风格缩小版）

用法（项目根目录，建议 aitest01_py310）：
  python eval/bench/run_bench.py --paradigm react
  python eval/bench/run_bench.py --case tk_get --paradigm react
  python eval/bench/run_bench.py --grid paradigm --stamp

与幻觉 Phase1 平行：本入口写 eval_results/bench_*.jsonl，主指标 pass_rate。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS = _ROOT / "eval_results"
_DEFAULT_CASES = Path(__file__).resolve().parent / "cases_v1.jsonl"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from eval.bench.env_ticket import reset_active_desk
from eval.bench.load_bench_cases import load_bench_cases
from eval.bench.scorer import score_checks
from pipelines.common.tee_utf8 import TeeUtf8
from pipelines.factory import build_pipeline, run_pipeline
from pipelines.specs import PARADIGMS, PipelineSpec


def _out_paths(stamp: bool) -> Path:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    if stamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return _RESULTS / f"bench_{ts}.jsonl"
    return _RESULTS / "bench_latest.jsonl"


def _specs(args) -> List[PipelineSpec]:
    if args.grid == "paradigm":
        ready = ("simple", "react", "hybrid", "plan", "tot", "reflection")
        return [PipelineSpec(paradigm=p, tools="bench", memory="off") for p in ready]
    return [
        PipelineSpec(paradigm=args.paradigm, tools="bench", memory="off"),
    ]


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    print()
    print("#" * 72)
    print(" BENCH SUMMARY")
    print("#" * 72)
    print(f"{'case':<28} {'pipeline':<22} {'pass':>4} {'sec':>6} {'tools':>5}")
    print("-" * 72)
    n_pass = 0
    for r in rows:
        print(
            f"{r['id']:<28} {r['pipeline']:<22} {r['pass']:>4} "
            f"{r['elapsed_sec']:>6.1f} {r['tool_calls']:>5}"
        )
        n_pass += int(r["pass"])
    print("-" * 72)
    print(f" total={len(rows)}  pass={n_pass}  pass_rate={n_pass / max(len(rows), 1):.2%}")
    print("#" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="myagent 能力评测（bench）")
    parser.add_argument("--cases-file", default=str(_DEFAULT_CASES))
    parser.add_argument("--case", default=None, help="只跑一个 case id")
    parser.add_argument("--paradigm", default="react", choices=list(PARADIGMS))
    parser.add_argument("--grid", choices=["paradigm"], default=None)
    parser.add_argument("--stamp", action="store_true")
    parser.add_argument("--out", default=None, help="jsonl 输出路径")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    cases_path = Path(args.cases_file)
    if not cases_path.is_absolute():
        cases_path = _ROOT / cases_path
    cases = load_bench_cases(cases_path, case_id=args.case)
    if not cases:
        print(f"[error] 无题目: {cases_path} case={args.case}")
        sys.exit(1)

    specs = _specs(args)
    jsonl_path = Path(args.out) if args.out else _out_paths(stamp=args.stamp)
    if not jsonl_path.is_absolute():
        jsonl_path = _ROOT / jsonl_path
    txt_path = jsonl_path.with_suffix(".txt")

    tee = None
    real_stdout = sys.stdout
    if not args.no_save:
        tee = TeeUtf8(real_stdout, txt_path)
        sys.stdout = tee

    rows: List[Dict[str, Any]] = []
    try:
        print()
        print("*" * 72)
        print(" PIPELINE BENCH")
        print("*" * 72)
        print(f" cases-file: {cases_path}")
        print(f" specs: {[s.tag for s in specs]}")
        print(f" n_cases: {len(cases)}")
        print("*" * 72)

        handles = []
        for spec in specs:
            try:
                handles.append(build_pipeline(spec))
                print(f"  [build] OK   {spec.tag}")
            except Exception as e:
                print(f"  [build] FAIL {spec.tag}  ({e})")

        if not handles:
            print("[error] 无可用 pipeline")
            sys.exit(1)

        for case in cases:
            cid = case["id"]
            q = case["question"]
            setup = case.get("setup")
            checks = case["checks"]
            print()
            print(f">> case={cid}")

            for h in handles:
                desk = reset_active_desk(setup)
                print(f"  [run ] {h.spec.tag} ...")
                result = run_pipeline(q, handle=h)
                snap = desk.snapshot()
                # 评分用环境真实调用次数，不用 stdout 粗估
                env_tools = int(getattr(desk, "tool_call_count", 0))
                scored = score_checks(
                    answer=result.answer,
                    tool_calls=env_tools,
                    env_snapshot=snap,
                    checks=checks,
                    error=result.error,
                )
                row = {
                    "id": cid,
                    "pipeline": result.pipeline,
                    "paradigm": h.spec.paradigm,
                    "tools": h.spec.tools,
                    "pass": scored["pass"],
                    "check_results": scored["check_results"],
                    "answer": result.answer,
                    "tool_calls": env_tools,
                    "tool_calls_trace_est": result.tool_calls,
                    "steps": result.steps,
                    "elapsed_sec": round(result.elapsed_sec, 3),
                    "error": result.error,
                    "question": q,
                }
                rows.append(row)
                mark = "PASS" if row["pass"] else "FAIL"
                print(
                    f"  [done] {h.spec.tag} | {cid} | {mark} | "
                    f"{result.elapsed_sec:.1f}s | tools={env_tools}"
                )
                if not row["pass"]:
                    for cr in scored["check_results"]:
                        if not cr["ok"]:
                            print(f"         x {cr['type']}: {cr['detail']}")

        _print_summary(rows)
    finally:
        if tee is not None:
            sys.stdout = real_stdout
            tee.close()

    if not args.no_save:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        latest = _RESULTS / "bench_latest.jsonl"
        if jsonl_path.resolve() != latest.resolve():
            latest.write_text(jsonl_path.read_text(encoding="utf-8"), encoding="utf-8")
            latest_txt = _RESULTS / "bench_latest.txt"
            if txt_path.exists():
                latest_txt.write_bytes(txt_path.read_bytes())
        print(f"[saved] jsonl -> {jsonl_path}")
        print(f"[saved] txt   -> {txt_path}")


if __name__ == "__main__":
    main()
