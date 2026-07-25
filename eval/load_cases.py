"""
加载 eval case 文件（jsonl）——字段兼容 pipelines 的 id/question/note。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_cases_jsonl(path: str | Path, case_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    每行一个 json。至少要有 id、question。
    why/topic 会拼进 note，方便 compare 打印。
    """
    path = Path(path)
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj["id"])
            if case_id and cid != case_id:
                continue
            note = obj.get("note") or obj.get("why") or ""
            topic = obj.get("topic")
            if topic:
                note = f"[{topic}] {note}".strip()
            rows.append(
                {
                    "id": cid,
                    "question": str(obj["question"]),
                    "note": note,
                }
            )
    return rows
