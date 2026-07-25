# 评测结果目录（Pipeline Compare）

本目录只存放 `pipelines/compare.py` 的评测输出，与 `memory_data/`（SQLite / Qdrant 本地库）分离。

- `compare_latest.txt`：最近一次评测（UTF-8 BOM）
- `compare_YYYYMMDD_HHMMSS.txt`：加 `--stamp` 时按时间戳另存

不要用 PowerShell `Tee-Object` 往这里重定向；由 compare 的 `--out` 写文件即可。
