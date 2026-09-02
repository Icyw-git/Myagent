# 能力评测设计（τ-bench / SWE-bench 思路 → myagent）

## 两条轨道

| 轨道 | 问题 | 入口 | 主指标 |
|------|------|------|--------|
| 幻觉 Phase1 | 没工具会不会编造 | `pipelines/compare.py` + `eval/score_suspected.py` | `suspected_hallucination` |
| **能力 Bench** | 有环境能不能办对 | `eval/bench/run_bench.py` | **`pass` / pass_rate`** |

二者共用 `pipelines.factory`，**case / 打分 / 结果文件互不混用**。

## 从公开 benchmark 借什么

- **τ-bench**：自然语言目标 + 可调用工具的可变环境；成功多为二元；可含 policy。
- **SWE-bench**：成功由自动检查器判定，不靠 LLM 主观分。
- **本仓库缩小版**：`TicketDesk` 内存工单 + `calculator`；检查器看答案 / 工具次数 / DB 终态。

## 评分原则

- **终态优先**：`db_equals` / 答案匹配为主（贴近 τ-bench 的 DB reward）
- **工具次数**：`min_tool_calls` 使用 `TicketDesk.tool_call_count`（含 calculator），**不用** stdout 粗估
- **拒答**：`must_refuse` 覆盖「无法 / 不存在 / 查不到」等说法；policy 题另要求答案含「关闭」

## 本轮不做


- SWE 风格「改代码 + pytest」
- LLM-as-judge
- 改动 `compare.py` 主流程（避免两套目标挤在一张文本表）

## 扩展点

- 新域：仿 `env_ticket.py` 再挂一个 env，在 `tools=bench` 或新 `tools=` 枚举注册
- SWE-lite：另开 `eval/bench_swe/`，与本目录并列即可
