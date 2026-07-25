"""
评测材料（两条轨道，互不混用）

## 1) 幻觉 Phase 1（tools=none）
- HYPOTHESIS.md / power_notes.md / cases_realtime_v1.jsonl
- suspected_rules.py / score_suspected.py / labels/judge_rubric.md
- load_cases.py → 给 pipelines/compare.py --cases-file 用

试跑：
  python eval/score_suspected.py --demo
  python pipelines/compare.py --cases-file eval/cases_realtime_v1.jsonl --case rt_weather_wuhan --grid paradigm --tools none --stamp

## 2) 能力 Bench（τ-lite，tools=bench）
- 见 eval/bench/（DESIGN.md / cases_v1.jsonl / run_bench.py）
- 主指标：pass_rate（确定性检查器）

试跑：
  python eval/bench/run_bench.py --case tk_get --paradigm react
  python tests/test_eval_bench.py
"""
