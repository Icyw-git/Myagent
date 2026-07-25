"""
TicketDesk —— 能力评测用的极简可变环境（τ-bench 思路缩小版）。

- 内存工单库，可每题 reset
- 工具：get_ticket / update_ticket / cancel_ticket
- policy：status=closed 的工单不可 update / cancel
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional


# 默认种子数据（cases 的 setup 可覆盖）
_DEFAULT_TICKETS: Dict[str, Dict[str, Any]] = {
    "T100": {
        "id": "T100",
        "title": "无法登录",
        "assignee": "韩梅梅",
        "status": "open",
        "priority": "high",
    },
    "T200": {
        "id": "T200",
        "title": "发票重开",
        "assignee": "李雷",
        "status": "open",
        "priority": "low",
    },
    "T300": {
        "id": "T300",
        "title": "已解决的网络故障",
        "assignee": "王芳",
        "status": "closed",
        "priority": "medium",
    },
}


class TicketDesk:
    """可变环境：工具读写同一份 tickets。"""

    def __init__(self) -> None:
        self.tickets: Dict[str, Dict[str, Any]] = {}
        self.reset()

    def reset(self, setup: Optional[Dict[str, Any]] = None) -> None:
        """每题开始前调用。setup 可含 tickets 覆盖表（深拷贝）。"""
        setup = setup or {}
        base = setup.get("tickets")
        if base is None:
            self.tickets = copy.deepcopy(_DEFAULT_TICKETS)
        else:
            self.tickets = copy.deepcopy(base)
            # 保证每条有 id 字段
            for tid, row in self.tickets.items():
                row.setdefault("id", tid)
        # 真实工具调用次数（评分用，不依赖 stdout 粗估）
        self.tool_call_count = 0

    def note_tool_call(self) -> None:
        self.tool_call_count = int(getattr(self, "tool_call_count", 0)) + 1

    def snapshot(self) -> Dict[str, Any]:
        return {"tickets": copy.deepcopy(self.tickets)}

    def get_path(self, path: str) -> Any:
        """点分路径，如 tickets.T100.assignee"""
        cur: Any = self.snapshot()
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    # ---- 工具实现（入参均为字符串，适配 register_function）----

    def tool_get_ticket(self, ticket_id: str) -> str:
        self.note_tool_call()
        tid = (ticket_id or "").strip()
        row = self.tickets.get(tid)
        if not row:
            return json.dumps({"error": f"工单不存在: {tid}"}, ensure_ascii=False)
        return json.dumps(row, ensure_ascii=False)

    def tool_update_ticket(self, raw: str) -> str:
        """
        入参 JSON，例如:
          {"id":"T100","assignee":"李雷"}
          {"id":"T100","priority":"low"}
        也可写 id=T100;assignee=李雷
        """
        self.note_tool_call()
        try:
            data = _parse_update(raw)
        except ValueError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        tid = data.get("id") or data.get("ticket_id")
        if not tid:
            return json.dumps({"error": "缺少 id"}, ensure_ascii=False)
        row = self.tickets.get(tid)
        if not row:
            return json.dumps({"error": f"工单不存在: {tid}"}, ensure_ascii=False)
        if row.get("status") == "closed":
            return json.dumps(
                {"error": "policy: 已关闭工单不可修改", "id": tid},
                ensure_ascii=False,
            )

        allowed = ("assignee", "priority", "title", "status")
        changed = {}
        for k in allowed:
            if k in data and k != "id":
                row[k] = data[k]
                changed[k] = data[k]
        if not changed:
            return json.dumps({"error": "未提供可更新字段"}, ensure_ascii=False)
        return json.dumps({"ok": True, "id": tid, "changed": changed, "ticket": row}, ensure_ascii=False)

    def tool_cancel_ticket(self, ticket_id: str) -> str:
        self.note_tool_call()
        tid = (ticket_id or "").strip()
        row = self.tickets.get(tid)
        if not row:
            return json.dumps({"error": f"工单不存在: {tid}"}, ensure_ascii=False)
        if row.get("status") == "closed":
            return json.dumps(
                {"error": "policy: 已关闭工单不可取消", "id": tid},
                ensure_ascii=False,
            )
        row["status"] = "cancelled"
        return json.dumps({"ok": True, "id": tid, "ticket": row}, ensure_ascii=False)


def _parse_update(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("空参数")
    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("JSON 必须是对象")
        return obj
    # id=T100;assignee=李雷 或 T100|assignee=李雷
    data: Dict[str, Any] = {}
    if "|" in text and "=" not in text.split("|", 1)[0]:
        tid, rest = text.split("|", 1)
        data["id"] = tid.strip()
        text = rest
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            if "id" not in data:
                data["id"] = part
            continue
        k, v = part.split("=", 1)
        data[k.strip()] = v.strip()
    return data


# 进程内单例：run_bench 每题 reset；tools 注册时绑定此实例
_ACTIVE: Optional[TicketDesk] = None


def get_active_desk() -> TicketDesk:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = TicketDesk()
    return _ACTIVE


def reset_active_desk(setup: Optional[Dict[str, Any]] = None) -> TicketDesk:
    desk = get_active_desk()
    desk.reset(setup)
    return desk


def list_ticket_tool_fns(desk: Optional[TicketDesk] = None) -> List[tuple]:
    """返回 (name, description, func) 列表，供 pipelines 工具轴挂载。"""
    d = desk or get_active_desk()
    return [
        (
            "get_ticket",
            "查询工单。输入工单 id（如 T100），返回 JSON 字段：id/title/assignee/status/priority。",
            d.tool_get_ticket,
        ),
        (
            "update_ticket",
            '更新工单。输入 JSON，例如 {"id":"T100","assignee":"李雷"}；'
            "也可 id=T100;assignee=李雷。已关闭(closed)工单不可改。",
            d.tool_update_ticket,
        ),
        (
            "cancel_ticket",
            "取消工单。输入工单 id；将 status 设为 cancelled。已关闭(closed)工单不可取消。",
            d.tool_cancel_ticket,
        ),
    ]
