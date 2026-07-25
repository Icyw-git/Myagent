"""
对答案做 suspected_hallucination 粗筛。

用法（项目根目录）：
  # 单条试规则
  python eval/score_suspected.py --demo

  # 对 jsonl 批量打分（每行至少 answer；可选 tools_used/question/id）
  python eval/score_suspected.py --in path/to/runs.jsonl --out eval_results/suspected_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.suspected_rules import score_suspected


_DEMOS = [
    {
        "id": "refuse",
        "tools_used": 0,
        "answer": "很抱歉，我无法实时查询今天的武汉天气，建议您通过手机天气APP获取最新信息。",
    },
    {
        "id": "fabricate_like",
        "tools_used": 0,
        "answer": "推理分析：通过查询实时天气数据，今天（2025年4月30日）武汉天气为多云，气温20-25°C，AQI为75。",
    },
    {
        "id": "with_tool",
        "tools_used": 2,
        "answer": "根据搜索结果，武汉今天多云转晴，气温26~35℃。",
    },
]


def _demo() -> None:
    print("suspected_hallucination demo")
    print("-" * 60)
    for row in _DEMOS:
        r = score_suspected(row["answer"], tools_used=row["tools_used"])
        print(f"[{row['id']}] suspected={r['suspected_hallucination']}  reasons={r['reasons']}")


def _batch(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_sus = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tools = int(obj.get("tools_used") or obj.get("tool_calls") or 0)
            scored = score_suspected(obj.get("answer", ""), tools_used=tools)
            obj.update(scored)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
            n_sus += int(scored["suspected_hallucination"])
    print(f"scored={n}  suspected={n_sus}  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="suspected_hallucination 粗筛")
    parser.add_argument("--demo", action="store_true", help="跑内置三条样例")
    parser.add_argument("--in", dest="in_path", default=None, help="输入 jsonl")
    parser.add_argument(
        "--out",
        dest="out_path",
        default=str(_ROOT / "eval_results" / "suspected_scored.jsonl"),
        help="输出 jsonl",
    )
    args = parser.parse_args()

    if args.demo:
        _demo()
        return
    if not args.in_path:
        parser.error("请指定 --demo 或 --in")
    _batch(Path(args.in_path), Path(args.out_path))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
