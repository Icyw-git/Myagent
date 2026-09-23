# MyAgent：从零学习构建 LLM 智能体

这是一个以 Python 实现智能体核心机制的学习型项目。项目不依赖单一的高层编排框架，而是围绕 LLM 调用、提示词约束、工具调用、上下文管理、记忆系统与评测体系，逐步实现并比较多种经典智能体范式。

> 适合希望理解 Agent 内部运行机制，并将原型逐步工程化的学习者。

## 学习内容

| 主题 | 对应实现 | 关注点 |
| --- | --- | --- |
| LLM 客户端与配置 | `llm_client.py`、`Config.py` | OpenAI 兼容接口、流式输出、环境变量配置 |
| 基础 Agent | `my_simple_agent.py`、`Agent.py` | 对话历史、基础工具调用 |
| ReAct | `my_react_agent.py`、`ReAct.py` | Thought → Action → Observation 循环、工具结果反馈 |
| Plan-and-Solve | `my_planandsolve_agent.py`、`PlanAndSolve.py` | 先规划、后分步执行 |
| Reflection | `my_reflection_agent.py` | 生成、反思、评分与改进 |
| Tree of Thought | `Tree_of_Thought.py` | 多分支候选、质量评分、择优输出 |
| Hybrid Agent | `HybridAgent.py` | 规划与工具执行结合 |
| 上下文工程 | `ContextBuilder.py`、`contextbase.py` | 历史裁剪、相关性、上下文包与压缩 |
| 记忆系统 | `memory_src/` | 工作、情景、语义、感知记忆及统一调度 |
| 项目工具 | `NoteTool.py`、`TerminalTool.py`、`async_tool_executor.py` | 项目笔记、受限终端、异步工具执行 |
| 评测与对比 | `pipelines/`、`eval/` | 范式对比、幻觉测试、确定性能力基准 |

配套学习笔记见 [agent知识.md](agent知识.md) 与 [agent框架.md](agent框架.md)。

## 项目结构

```text
.
├── my_simple_agent.py              # 基础 Agent
├── my_react_agent.py               # ReAct Agent
├── my_planandsolve_agent.py        # Plan-and-Solve Agent
├── my_reflection_agent.py          # Reflection Agent
├── Tree_of_Thought.py              # Tree of Thought Agent
├── HybridAgent.py                  # 规划 + ReAct 混合 Agent
├── ContextBuilder.py               # 上下文构建与预算管理
├── NoteTool.py / TerminalTool.py   # 项目笔记与终端工具
├── memory_src/                     # 可扩展记忆子系统
├── pipelines/                      # 三维 Pipeline 工厂与对比 CLI
├── eval/                           # 幻觉与能力评测材料
├── eval_results/                   # Pipeline 对比结果
└── tests/                          # 单元测试、离线冒烟与联网示例
```

## 环境准备

建议使用 Python 3.10 或更高版本，并在项目根目录创建虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`hello_agents` 已包含在 `requirements.txt` 中。若要使用本地语义相关性或完整向量记忆检索，还可按需安装：

```bash
python -m pip install qdrant-client sentence-transformers
```

其中，工作记忆和离线测试不要求 Qdrant；未安装 `sentence-transformers` 时，上下文选择会使用轻量关键词相关性。

## 配置

在根目录新建 `.env`，不要提交真实密钥。最小配置如下：

```dotenv
LLM_API_KEY="your-api-key"
LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
LLM_MODEL_ID="your-model-name"
TIMEOUT=60
```

使用实时搜索时，额外配置：

```dotenv
SERPAPI_API_KEY="your-serpapi-key"
```

注意：请使用 `google-search-results` 提供的 `serpapi` 接口；部分同名 `serpapi` 包不包含 `GoogleSearch`，会导致导入错误。

可选的记忆与向量检索配置：`EMBED_API_KEY`、`EMBED_BASE_URL`、`EMBED_MODEL_NAME`、`QDRANT_URL`、`QDRANT_API_KEY`，以及 Neo4j 相关环境变量。完整字段以本地 `.env` 和 `memory_src/base.py` 为准。

## 快速开始

以下命令均在项目根目录运行。

### 运行单个 Agent 示例

```bash
python tests/test_simple_agent.py
python tests/test_planandsolve_agent.py
python tests/test_reflection_agent.py
```

这些示例会调用配置的 LLM 服务。ReAct、混合 Agent 与工具调用示例可参考 `tests/` 下的对应脚本。

### 运行离线功能检查

```bash
python tests/test_pipelines.py
python tests/test_eval_bench.py
python tests/test_context_aware_agent.py
python tests/test_withtools_agent.py
```

也可以通过 pytest 执行：

```bash
python -m pytest -m unit -v
```

## Pipeline 对比

`pipelines/` 将 Agent 组合抽象为三个维度：

| 维度 | 可选值 | 说明 |
| --- | --- | --- |
| 范式 | `simple`、`react`、`hybrid`、`plan`、`tot`、`reflection` | 六种 Agent 实现 |
| 工具 | `none`、`search`、`calc`、`search+calc`、`bench` | 搜索、计算或可复现评测工具 |
| 记忆 | `off`、`working`、`episodic`、`rag` | `working` 当前仅接入 `react`；其他范式会明确拒绝该组合 |

运行默认 A/B 预设：

```bash
python pipelines/compare.py --case qa
```

对比所有范式的纯推理表现：

```bash
python pipelines/compare.py --grid paradigm --tools none --memory off --case qa
```

以时间戳保存结果：

```bash
python pipelines/compare.py --case qa --stamp
```

结果默认写入 `eval_results/compare_latest.txt`；使用 `--stamp` 时同时生成带时间戳的副本。

## 评测

项目目前包含两条相互独立的评测轨道：

1. 幻觉 Phase 1：在禁用工具的前提下，对实时信息题评估回答是否违反事实约束。
2. 能力 Bench：使用 TicketDesk 与计算器的确定性环境，主指标为 `pass_rate`。

示例命令：

```bash
# 幻觉规则演示
python eval/score_suspected.py --demo

# 对实时题进行无工具范式对比
python pipelines/compare.py --cases-file eval/cases_realtime_v1.jsonl --case rt_weather_wuhan --grid paradigm --tools none --stamp

# TicketDesk 能力评测
python eval/bench/run_bench.py --case tk_get --paradigm react
```

更多说明见 [eval/README.md](eval/README.md)、[eval/bench/README.md](eval/bench/README.md) 和 [eval/bench/DESIGN.md](eval/bench/DESIGN.md)。

## 当前实现状态

- `simple`、`react`、`hybrid` 会使用 Pipeline 的工具轴；`plan`、`tot`、`reflection` 当前不调用外部工具。
- `memory_src/` 已提供多类记忆及 `MemoryTool`；Pipeline 的 `working` memory 当前仅接入 ReAct，`episodic`、`semantic`、`rag` 仍会明确报出未实现。
- 搜索与真实 Agent 示例需要有效的 LLM 配置；离线测试用于验证工厂、评测环境、上下文和工具组件，不消耗模型调用。
- `TerminalTool` 默认只提供 `echo`、`dir` 和工作区内的 `cd`，不通过 shell 运行管道或重定向。需要执行其他程序时，调用方必须通过 `allowed_commands` 显式授权；工作区路径约束不等于完整的操作系统沙箱。
- Pipeline 对支持统一结果接口的 Agent 使用真实步骤数和工具调用数；旧 Agent 暂时回退到 stdout 估算。

默认离线测试：

```bash
python -m pytest -m unit -q
```

需要额外运行集成工具测试时使用 `python -m pytest -m integration -q`；需要 LLM 或外部服务的测试标记为 `online`。

## 后续学习方向

- 将 `MemoryTool` 正式接入 Pipeline 记忆轴，比较不同记忆策略对多轮任务的影响。
- 为工具调用增加结构化参数校验、重试、超时与审计日志。
- 扩展评测集，结合 `pass_rate`、工具调用次数、耗时和人工质量评审进行分析。
- 将当前原生实现与 LangGraph、AutoGen 等框架方案进行同题对照。

## 参考

项目中的理论笔记和基础接口参考了 Datawhale 的 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 学习材料，并在此基础上完成了本项目的实验性实现与扩展。
