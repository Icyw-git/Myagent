# test_note_tool.py
# NoteTool 离线冒烟：create / read / update / search / list / summary / delete + 索引持久化
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_note_tool.py
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from NoteTool import NoteTool


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_note_tool_full_flow() -> None:
    with tempfile.TemporaryDirectory(prefix="note_tool_test_") as tmp:
        workspace = os.path.join(tmp, "notes")
        tool = NoteTool(workspace=workspace)

        print("=== 1) 初始化：目录 + 空索引 ===")
        _assert(os.path.isdir(workspace), "workspace 目录应已创建")
        _assert(tool.index == {}, "新工作区索引应为空")
        print(f"workspace={workspace}")
        print("通过\n")

        print("=== 2) create：写 .md + 更新 index.yaml ===")
        nid1 = tool._create_note(
            title="项目进展",
            content="已完成需求分析，下一步：设计方案",
            note_type="task_state",
            tags=["milestone", "phase1"],
        )
        nid2 = tool._create_note(
            title="阻塞项",
            content="等待 API Key 审批",
            note_type="blocker",
            tags=["urgent"],
        )
        nid3 = tool._create_note(
            title="参考资料",
            content="武汉东湖风景区介绍",
            note_type="reference",
            tags=["travel"],
        )
        print(f"created: {nid1}, {nid2}, {nid3}")
        _assert(nid1 in tool.index and nid2 in tool.index, "索引应含新建 note_id")
        md_path = tool.index[nid1]["file_path"]
        _assert(os.path.isfile(md_path), f".md 文件应存在: {md_path}")
        _assert(os.path.isfile(tool.index_file), "index.yaml 应已写出")
        print("通过\n")

        print("=== 3) read：YAML 头 + 正文分离 ===")
        note = tool._read_note(nid1)
        print(f"title={note['metadata'].get('title')}, content={note['content'][:40]}...")
        _assert(note["metadata"].get("title") == "项目进展", "元数据标题应匹配")
        _assert("需求分析" in note["content"], "正文应保留原文")
        try:
            tool._read_note("note_not_exist")
            _assert(False, "不存在的 id 应抛 ValueError")
        except ValueError:
            pass
        print("通过\n")

        print("=== 4) update：改标题/正文/标签 ===")
        msg = tool._update_note(
            nid1,
            title="项目进展-更新",
            content="设计方案已完成一半",
            tags=["milestone", "phase2"],
        )
        print(msg)
        updated = tool._read_note(nid1)
        _assert(updated["metadata"]["title"] == "项目进展-更新", "标题应已更新")
        _assert("设计方案已完成一半" in updated["content"], "正文应已更新")
        _assert("phase2" in updated["metadata"].get("tags", []), "标签应已更新")
        print("通过\n")

        print("=== 5) search：关键词 + type/tags 过滤 ===")
        hits = tool._search_notes("设计方案")
        print(f"search '设计方案' -> {len(hits)} 条: {[h['title'] for h in hits]}")
        _assert(len(hits) >= 1 and any(h["note_id"] == nid1 for h in hits), "应命中更新后的进展笔记")

        hits_type = tool._search_notes("等待", note_type="blocker")
        _assert(len(hits_type) == 1 and hits_type[0]["note_id"] == nid2, "按 type=blocker 应只命中阻塞项")

        hits_tag = tool._search_notes("东湖", tags=["travel"])
        _assert(len(hits_tag) == 1 and hits_tag[0]["note_id"] == nid3, "按 tags=travel 应命中参考资料")

        hits_miss = tool._search_notes("不存在的关键词xyz")
        _assert(hits_miss == [], "无关关键词应无结果")
        print("通过\n")

        print("=== 6) list + summary ===")
        listed = tool._list_notes(limit=10)
        print(f"list 共 {len(listed)} 条，最新={listed[0].get('title')}")
        _assert(len(listed) == 3, "应列出 3 条")
        # 更新过的 nid1 应排在前面（updated_at 最新）
        _assert(listed[0]["id"] == nid1, "list 应按 updated_at 倒序，更新过的在前")

        blockers = tool._list_notes(note_type="blocker")
        _assert(len(blockers) == 1, "list 按 type 过滤")

        summary = tool._summary()
        print(f"summary: total={summary['total_count']}, types={summary['type_distribution']}")
        _assert(summary["total_count"] == 3, "总数应为 3")
        _assert(summary["type_distribution"].get("blocker") == 1, "类型分布应含 blocker=1")
        _assert(len(summary["recent_notes"]) <= 5, "最近笔记最多 5 条")
        print("通过\n")

        print("=== 7) delete + 持久化重载 ===")
        del_msg = tool._delete_note(nid2)
        print(del_msg)
        _assert(nid2 not in tool.index, "删除后索引不应再含该 id")
        _assert(not os.path.exists(md_path) or nid2 not in tool.index, "索引已删")
        # nid2 的文件应不存在
        gone = tool.index.get(nid2) is None
        _assert(gone, "nid2 应从索引移除")
        # 确认 .md 删掉（用原先路径）
        old_path = os.path.join(workspace, f"{nid2}.md")
        _assert(not os.path.exists(old_path), f".md 应已删除: {old_path}")

        # 新实例从 index.yaml 重载，应只剩 2 条
        tool2 = NoteTool(workspace=workspace)
        _assert(len(tool2.index) == 2, f"重载后应剩 2 条，实际={len(tool2.index)}")
        _assert(nid1 in tool2.index and nid3 in tool2.index, "重载应保留未删笔记")
        _assert(nid2 not in tool2.index, "重载不应再出现已删笔记")
        reloaded = tool2._read_note(nid1)
        _assert("设计方案已完成一半" in reloaded["content"], "重载后正文应仍可读")
        print("通过\n")

        print("全部 NoteTool 测试通过。")


if __name__ == "__main__":
    test_note_tool_full_flow()
