"""加载能力评测 case（保留 setup / checks，不丢字段）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_bench_cases(
    path: str | Path,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj["id"])
            if case_id and cid != case_id:
                continue
            if "question" not in obj or "checks" not in obj:
                raise ValueError(f"case {cid} 需要 question 与 checks")
            rows.append(obj)
    return rows
