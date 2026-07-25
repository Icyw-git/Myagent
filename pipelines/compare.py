"""
Pipeline 对比入口（三维工厂）

用法（项目根目录，建议 aitest01_py310）：
    python pipelines/compare.py --case qa
    python pipelines/compare.py --preset a,b --case qa
    python pipelines/compare.py --paradigm plan --tools none --case qa
    python pipelines/compare.py --grid paradigm --tools none --memory off --case qa
    python pipelines/compare.py --case qa --stamp

说明：
- 预设 a = hybrid|search|off ； b = react|search|off
- 六种范式均已接通；plan/tot/reflection 暂不用工具轴
- 记忆轴目前仅 off
- 评测结果默认写到 eval_results/（与 memory_data 分离），UTF-8 BOM
- 幻觉 Phase1 case：`--cases-file eval/cases_realtime_v1.jsonl --tools none`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ROOT / "eval_results"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from pipelines.common.cases import CASES
from pipelines.common.report import print_case_report, print_final_summary, print_run_line
from pipelines.common.result import RunResult
from pipelines.common.tee_utf8 import TeeUtf8
from pipelines.factory import build_pipeline, run_pipeline
from pipelines.specs import PARADIGMS, PipelineSpec, PRESETS, TOOL_KINDS


def _default_out_path(stamp: bool) -> Path:
    """评测结果统一落在 eval_results/，不混进 memory_data。"""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if stamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return _RESULTS_DIR / f"compare_{ts}.txt"
    return _RESULTS_DIR / "compare_latest.txt"


def _also_update_latest(saved: Path) -> None:
    """--stamp 时同步一份到 compare_latest.txt，方便直接打开最新全量结果。"""
    if saved.name == "compare_latest.txt":
        return
    latest = _RESULTS_DIR / "compare_latest.txt"
    latest.write_bytes(saved.read_bytes())


def _parse_presets(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _specs_from_args(args) -> List[PipelineSpec]:
    """根据 CLI 参数展开要跑的 spec 列表。"""
    if args.grid == "paradigm":
        ready = ("simple", "react", "hybrid", "plan", "tot", "reflection")
        return [
            PipelineSpec(paradigm=p, tools=args.tools, memory=args.memory)
            for p in ready
        ]

    if args.paradigm:
        return [
            PipelineSpec(
                paradigm=args.paradigm,
                tools=args.tools,
                memory=args.memory,
            )
        ]

    if args.only and args.only != "both":
        return [PRESETS[args.only]]
    names = _parse_presets(args.preset)
    return [PRESETS[n] for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="对比多条 Pipeline")
    parser.add_argument("--only", choices=["a", "b", "both"], default=None)
    parser.add_argument("--preset", default="a,b", help="预设列表，默认 a,b")
    parser.add_argument("--paradigm", default=None, choices=list(PARADIGMS))
    parser.add_argument("--tools", default="search", choices=list(TOOL_KINDS))
    parser.add_argument("--memory", default="off")
    parser.add_argument("--grid", choices=["paradigm"], default=None)
    parser.add_argument("--case", default=None, help="case id，如 qa / search / rt_weather_wuhan")
    parser.add_argument(
        "--cases-file",
        default=None,
        help="可选 jsonl case 文件（默认用 pipelines 内置 CASES）；见 eval/cases_realtime_v1.jsonl",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="只打汇总表，不打印每条答案预览",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="按时间戳另存 eval_results/compare_YYYYMMDD_HHMMSS.txt",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="自定义结果路径；默认 eval_results/compare_latest.txt（可用 --stamp）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不写结果文件，只打控制台",
    )
    args = parser.parse_args()

    # 控制台 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    tee = None
    real_stdout = sys.stdout
    out_path = None
    if not args.no_save:
        if args.out:
            out_path = Path(args.out)
            if not out_path.is_absolute():
                out_path = _ROOT / out_path
        else:
            out_path = _default_out_path(stamp=args.stamp)
        tee = TeeUtf8(real_stdout, out_path)
        sys.stdout = tee

    try:
        _run_compare(args)
    finally:
        if tee is not None:
            sys.stdout = real_stdout
            tee.close()
            _also_update_latest(tee.path)
            print(f"[saved] UTF-8 -> {tee.path}")
            if tee.path.name != "compare_latest.txt":
                print(f"[saved] also  -> {_RESULTS_DIR / 'compare_latest.txt'}")


def _run_compare(args) -> None:
    if args.cases_file:
        from eval.load_cases import load_cases_jsonl

        cases_path = Path(args.cases_file)
        if not cases_path.is_absolute():
            cases_path = _ROOT / cases_path
        cases = load_cases_jsonl(cases_path, case_id=args.case)
        if not cases:
            print(f"[error] cases-file 无可用题目: {cases_path}")
            return
    else:
        cases = CASES
        if args.case:
            cases = [c for c in CASES if c["id"] == args.case]
            if not cases:
                print(f"[error] 找不到 case id={args.case}，可选: {[c['id'] for c in CASES]}")
                return

    try:
        specs = _specs_from_args(args)
    except (KeyError, ValueError, NotImplementedError) as e:
        print(f"[error] 规格错误: {e}")
        return

    print()
    print("*" * 72)
    print(" PIPELINE COMPARE")
    print("*" * 72)
    if args.cases_file:
        print(f" cases-file: {args.cases_file}")

    handles = []
    for spec in specs:
        try:
            handles.append(build_pipeline(spec))
            print(f"  [build] OK   {spec.tag}")
        except NotImplementedError as e:
            print(f"  [build] SKIP {spec.tag}  ({e})")
        except Exception as e:
            print(f"  [build] FAIL {spec.tag}  ({e})")

    if not handles:
        print("[error] 没有可跑的 pipeline。")
        return

    print("-" * 72)
    print(f" specs : {[h.spec.tag for h in handles]}")
    print(f" cases : {[c['id'] for c in cases]}")
    print("*" * 72)

    case_results: Dict[str, List[RunResult]] = {}

    for case in cases:
        results: List[RunResult] = []
        q = case["question"]
        print()
        print(f">> case={case['id']}  running {len(handles)} pipeline(s)...")

        for h in handles:
            print(f"  [run ] {h.spec.tag} ...", flush=True)
            r = run_pipeline(q, handle=h)
            print_run_line(h.spec.tag, case["id"], r)
            results.append(r)

        case_results[case["id"]] = results
        print_case_report(
            case["id"],
            case["note"],
            q,
            results,
            show_detail=not args.no_detail,
        )

    print_final_summary(case_results)
    print("提示：答案质量需人工看 details；完整 trace 在 RunResult.raw_trace。")


if __name__ == "__main__":
    main()
