# 样本量与重复（Phase 1 备注）

## 粗算设定

- 因变量：二值（编造 / 未编造）
- 猜测效应：多步组幻觉率约 40%，单步组约 10%（可按预实验结果再改）
- 检验思路：配对设计上最终用 Cochran’s Q；样本量按两比例差做**数量级**估计即可

## 可行方案（与仓库现状匹配）

| 项 | 取值 |
|----|------|
| case 数 | 15–20（见 `cases_realtime_v1.jsonl`） |
| paradigm 数 | 6 |
| 每 case×paradigm 重复 | 2 |
| 总 run 量 | 约 15×6×2 = 180 至 20×6×2 = 240 |

「多 case、少重复」优先于「少 case、硬刷 30 次」，覆盖话题多样性，也更贴近配对设计。

## 可选：用 statsmodels 复核效力

若环境已装 `statsmodels`：

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# 独立两比例的粗算（配对设计实际效力通常更好，此数作上限参考）
es = proportion_effectsize(0.10, 0.40)
n = NormalIndPower().solve_power(effect_size=es, power=0.8, alpha=0.05, ratio=1)
print(n)  # 每组大约二十出头量级
```

Phase 1 **不强制**跑满 180；先用 5 case × 6 paradigm × 1 做标签试跑，确认规则与 rubric 可判后再挂夜跑。
