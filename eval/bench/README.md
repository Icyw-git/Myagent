# 能力评测（bench）

τ-bench 风格缩小版：固定环境工具 + 确定性 pass/fail。

## 怎么跑

```bash
# 单题
python eval/bench/run_bench.py --case tk_get --paradigm react

# 全量（默认 react|bench|off）
python eval/bench/run_bench.py --stamp

# 六范式对照（plan/tot/reflection 预期多 fail）
python eval/bench/run_bench.py --grid paradigm --stamp
```

结果：`eval_results/bench_*.jsonl` + 同名 `.txt`；最新一份同步为 `bench_latest.*`。

## 文件

- `DESIGN.md` — 与幻觉轨边界、设计映射
- `env_ticket.py` — TicketDesk 环境
- `cases_v1.jsonl` — 题目 + checks
- `scorer.py` — 自动打分
- `run_bench.py` — 入口

## 与幻觉实验

幻觉：`eval/cases_realtime_v1.jsonl` + `--tools none`。  
能力：本目录 + 固定 `--tools bench`（由 runner 写死，无需手填）。
