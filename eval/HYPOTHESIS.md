# 假设（幻觉实验 v1）

## H1（主假设，可证伪）

在「需要外部信息但工具不可用」的场景下，多步推理型 paradigm（plan / tot / reflection）
编造未经验证事实的比例，显著高于单步 / 工具驱动型 paradigm（simple / react）。

（hybrid 本实验默认也关工具，归入「多步」一侧做探索性对比，不进主对比的核心对子时需在文中说明。）

## H1a（机制假设）

多步 paradigm 每一步都要求模型「继续往下推理」，这种结构性压力会诱使模型在缺乏依据时
倾向于「编一个合理答案填补空白」，而不是承认不知道。

## 本阶段范围（Phase 1）

- 固定：同一 LLM、同一批 realtime case、`tools=none`
- 因变量粗筛：`suspected_hallucination`（规则）
- 因变量精标：`is_fabricated`（见 `labels/judge_rubric.md`，本阶段先定标准，大规模精标可后做）
