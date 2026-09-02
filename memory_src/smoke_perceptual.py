"""
PerceptualMemory（文本）冒烟

    conda run -n aitest01_py310 python memory_src/smoke_perceptual.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
sys.path.insert(0, str(_THIS))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from base import MemoryConfig, MemoryItem
from perceptual import PerceptualMemory


def main() -> None:
    config = MemoryConfig(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )
    pm = PerceptualMemory(config)
    samples = [
        MemoryItem(
            id="10000000-0000-4000-8000-000000000001",
            content="红色的猫坐在窗台上晒太阳",
            memory_type="perceptual",
            importance=0.7,
            user_id="smoke_user",
            metadata={"modality": "text"},
        ),
        MemoryItem(
            id="10000000-0000-4000-8000-000000000002",
            content="蓝色的狗在草地上奔跑",
            memory_type="perceptual",
            importance=0.6,
            user_id="smoke_user",
            metadata={"modality": "text"},
        ),
    ]
    for s in samples:
        print("[add]", pm.add(s)[:8], s.content[:20])

    hits = pm.retrieve("猫在哪里", limit=2, user_id="smoke_user", target_modality="text")
    print("[retrieve]", [h.content for h in hits])
    assert hits and "猫" in hits[0].content
    print("[ok] perceptual text smoke passed")


if __name__ == "__main__":
    main()
