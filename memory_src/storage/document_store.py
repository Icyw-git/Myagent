"""
SQLite 文档存储 —— 情景记忆的「结构化」那一半

职责（给 EpisodicMemory 用的最小接口）：
- save_episode: 写入一条情景
- get_episode: 按 id 读取
- filter_ids: 按时间/重要性/session/user 预过滤，返回 id 集合
- delete_episode: 双写失败时回滚用
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Optional, Set


class SQLiteDocumentStore:
    """情景记忆的结构化存储（SQLite）"""

    def __init__(self, database_path: str = "./memory_data/memory.db"):
        self.database_path = database_path
        # 确保目录存在
        parent = os.path.dirname(os.path.abspath(database_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """建表：episodes 存结构化字段，context_json 存原始 metadata"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_id TEXT,
                    timestamp TEXT,
                    content TEXT,
                    importance REAL DEFAULT 0.5,
                    context_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp)"
            )
            conn.commit()

    def save_episode(
        self,
        episode_id: str,
        content: str,
        timestamp: str,
        session_id: str = "default",
        user_id: str = "default_user",
        importance: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写入 / 覆盖一条情景"""
        context = context or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                (episode_id, session_id, user_id, timestamp, content, importance, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    session_id,
                    user_id,
                    timestamp,
                    content,
                    importance,
                    json.dumps(context, ensure_ascii=False),
                ),
            )
            conn.commit()

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """按 id 取回一条，找不到返回 None"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def delete_episode(self, episode_id: str) -> None:
        """删除一条（双写回滚用）"""
        with self._connect() as conn:
            conn.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
            conn.commit()

    def filter_ids(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        min_importance: Optional[float] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Set[str]:
        """结构化预过滤，返回候选 episode_id 集合"""
        sql = "SELECT episode_id FROM episodes WHERE 1=1"
        params: list = []

        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if min_importance is not None:
            sql += " AND importance >= ?"
            params.append(min_importance)
        if start_time is not None:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            sql += " AND timestamp <= ?"
            params.append(end_time)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {row["episode_id"] for row in rows}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        context = {}
        raw = row["context_json"]
        if raw:
            try:
                context = json.loads(raw)
            except json.JSONDecodeError:
                context = {}
        return {
            "episode_id": row["episode_id"],
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "timestamp": row["timestamp"],
            "content": row["content"],
            "importance": row["importance"],
            "context": context,
        }
