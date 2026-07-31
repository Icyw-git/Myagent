"""
my_react_agent_withtools.py —— ContextToolsAgent（ReAct 接入 上下文 + 记忆 + NoteTool + TerminalTool）

在 ContextAwareAgent（my_react_agent_context.py）基础上平行扩展，不重写 ReAct 循环：
- ContextBuilder 背景（被动 build）—— 复用父类，不动
- MemoryTool / memory 工具（主动记忆）—— 复用父类，不动
- note 工具（主动）：结构化长期项目记忆。NoteTool = YAML 索引 + Markdown 笔记，
  适合沉淀「项目知识/进展/阻塞/结论」，比向量记忆更偏结构化项目笔记。
- terminal 工具（主动）：即时上下文。TerminalTool 在 workspace 沙箱内执行 shell 命令，
  实时读环境/跑命令，给 LLM 当前系统即时信息。
- 被动注入笔记上下文（可选开关 inject_notes_context）：每轮用当前 task 检索相关笔记，
  转成 ContextPacket(type=note_context) 注入背景 [Context]；TerminalTool 有副作用，
  只做主动工具、不被动注入。

工具注册风格与父类 memory 一致：register_function(name, desc, func)，func 收 str 返 str。
运行（项目根目录）:
  D:\\Anaconda_envs\\envs\\aitest01_py310\\python.exe my_react_agent_withtools.py
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import List, Optional

from contextbase import ContextConfig, ContextPacket
from NoteTool import NoteTool
from TerminalTool import TerminalTool
from my_react_agent_context import ContextAwareAgent
from hello_agents import ToolRegistry, HelloAgentsLLM
from Config import Config

# note 工具描述：action 对照表 + JSON 示例，给 LLM 看
NOTE_TOOL_DESCRIPTION = (
    "结构化项目笔记工具（长期项目记忆，Markdown + YAML 索引，落盘持久化）。"
    "参数必须是 JSON 字符串，action 支持：\n"
    '- {"action":"create","title":"标题","content":"正文","note_type":"task_state/conclusion/blocker/action/reference/general","tags":["tag1"]}\n'
    '- {"action":"search","query":"关键词","limit":5,"note_type":"可选","tags":["可选"]}\n'
    '- {"action":"read","note_id":"note_xxx"}\n'
    '- {"action":"update","note_id":"note_xxx","title":"可选","content":"可选","note_type":"可选","tags":["可选"]}\n'
    '- {"action":"list","note_type":"可选","tags":["可选"],"limit":10}\n'
    '- {"action":"summary"}\n'
    '- {"action":"delete","note_id":"note_xxx"}\n'
    "项目进展、结论、阻塞项、参考资料等请用 note 沉淀，后续任务可通过 search 复用。"
)

# terminal 工具描述：给 LLM 写清边界
TERMINAL_TOOL_DESCRIPTION = (
    "终端命令执行工具（即时上下文）。参数直接是 shell 命令字符串（如 dir、echo hello、python --version）。"
    "在 workspace 沙箱内执行，带超时与输出截断；cd 只能在工作空间范围内。"
    "需要实时系统/环境信息（文件、路径、版本、命令输出）时使用。"
)


def _note_fn(note_tool: NoteTool):
    """note 工具适配：入参 JSON 字符串 → NoteTool 私有方法 → JSON 文本。

    回退规则：非 JSON 入参当作 search 查询（与 memory 工具 search 回退一致）。
    """
    def note_fn(raw: str) -> str:
        raw = (raw or "").strip()
        try:
            params = json.loads(raw) if raw.startswith("{") else {"action": "search", "query": raw}
            action = params.get("action", "search")

            if action == "create":
                note_id = note_tool._create_note(
                    title=params.get("title", ""),
                    content=params.get("content", ""),
                    note_type=params.get("note_type", "general"),
                    tags=params.get("tags") or [],
                )
                return json.dumps({"ok": True, "note_id": note_id}, ensure_ascii=False)

            if action == "read":
                note = note_tool._read_note(params["note_id"])
                return json.dumps(note, ensure_ascii=False)

            if action == "update":
                msg = note_tool._update_note(
                    note_id=params["note_id"],
                    title=params.get("title"),
                    content=params.get("content"),
                    note_type=params.get("note_type"),
                    tags=params.get("tags"),
                )
                return json.dumps({"ok": True, "message": msg}, ensure_ascii=False)

            if action == "search":
                hits = note_tool._search_notes(
                    query=params.get("query", ""),
                    limit=params.get("limit", 5),
                    note_type=params.get("note_type"),
                    tags=params.get("tags"),
                )
                return json.dumps({"count": len(hits), "results": hits}, ensure_ascii=False)

            if action == "list":
                notes = note_tool._list_notes(
                    note_type=params.get("note_type"),
                    tags=params.get("tags"),
                    limit=params.get("limit", 10),
                )
                return json.dumps({"count": len(notes), "results": notes}, ensure_ascii=False)

            if action == "summary":
                return json.dumps(note_tool._summary(), ensure_ascii=False)

            if action == "delete":
                msg = note_tool._delete_note(params["note_id"])
                return json.dumps({"ok": True, "message": msg}, ensure_ascii=False)

            return json.dumps({"error": f"未知 note action={action}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Note error: {e}"}, ensure_ascii=False)
    return note_fn


def _terminal_fn(terminal_tool: TerminalTool):
    """terminal 工具适配：入参直接是 shell 命令字符串。"""
    def terminal_fn(raw: str) -> str:
        raw = (raw or "").strip()
        if not raw:
            return "错误：命令为空"
        return terminal_tool._execute_command(raw)
    return terminal_fn


class ContextToolsAgent(ContextAwareAgent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        max_steps: int = 5,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompt: Optional[str] = None,
        context_config: Optional[ContextConfig] = None,
        custom_context: Optional[str] = None,
        memory_tool=None,
        register_memory_tool: bool = True,
        note_tool: Optional[NoteTool] = None,
        terminal_tool: Optional[TerminalTool] = None,
        inject_notes_context: bool = True,
    ):
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            max_steps=max_steps,
            system_prompt=system_prompt,
            config=config,
            custom_prompt=custom_prompt,
            context_config=context_config,
            custom_context=custom_context,
            memory_tool=memory_tool,
            register_memory_tool=register_memory_tool,
        )

        # 结构化长期项目记忆：NoteTool（Markdown + YAML 索引落盘）
        self.note_tool = note_tool or NoteTool(workspace="./notes")

        # 即时上下文：TerminalTool 沙箱，默认目录固定在项目根下 terminal_workspace
        if terminal_tool is None:
            terminal_workspace = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "terminal_workspace"
            )
            terminal_tool = TerminalTool(workspace=terminal_workspace)
        self.terminal_tool = terminal_tool

        # 被动注入笔记上下文开关（每轮用当前 task 检索相关笔记进背景）
        self.inject_notes_context = inject_notes_context
        # 最近一次用户输入，供 _build_custom_context 里检索笔记（不空转父类带 custom 参数）
        self._last_input_text = ""

        self._register_extra_tools()
        print(f"{name}初始化完成，启用 note 项目笔记 + terminal 即时上下文")

    def _register_extra_tools(self) -> None:
        """注册 note / terminal 工具；与父类 memory 注册一致做去重。"""
        names = set(getattr(self.tool_registry, "_tools", {}) or {}) | set(
            getattr(self.tool_registry, "_functions", {}) or {}
        )
        if "note" not in names:
            self.tool_registry.register_function(
                "note", NOTE_TOOL_DESCRIPTION, _note_fn(self.note_tool)
            )
        if "terminal" not in names:
            self.tool_registry.register_function(
                "terminal", TERMINAL_TOOL_DESCRIPTION, _terminal_fn(self.terminal_tool)
            )

    def _expand_search_queries(self, query: str) -> List[str]:
        """把中文自然语言查询扩展成可检索的候选词。

        NoteTool._search_notes 是子串匹配（query in content），整句中文往往不命中；
        这里拆出「每段头部 2~6 字窗口」的片段并行检索，再合并去重（不改 NoteTool）。
        """
        queries = [query]
        parts = [p.strip() for p in re.split(r"[，。？?！!、\s,;；]+", query) if p.strip()]
        for part in parts:
            # 头部窗口：2 到 min(6, len(part))，如「服务器部署在哪里」→ 服务/服务器/服务器部/服务器部署...
            for end in range(2, min(7, len(part) + 1)):
                queries.append(part[:end])
        seen = set()
        uniq: List[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen and len(q) >= 2:
                seen.add(q)
                uniq.append(q)
            if len(uniq) >= 10:
                break
        return uniq

    def _build_custom_context(self, custom_context: str) -> List[ContextPacket]:
        """在父类 custom 包基础上，追加相关项目笔记包（可选开关）。

        思路：ContextAwareAgent.run() 每轮都会调 _build_custom_context(self.custom_context)；
        这里覆写它做「缺口补全」，避免整段复制父类 ReAct 循环。TerminalTool 有副作用，
        不被动注入，只做主动工具。
        """
        packets = super()._build_custom_context(custom_context)

        if not self.inject_notes_context or self.note_tool is None:
            return packets

        query = self._last_input_text or custom_context or ""
        if not query:
            return packets

        try:
            # 查询词扩展：整句 + 头部短词片段，合并去重后注入（不改 NoteTool）
            hits: List[dict] = []
            seen_ids = set()
            for q in self._expand_search_queries(query):
                for hit in self.note_tool._search_notes(q, limit=5):
                    nid = hit.get("note_id")
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        hits.append(hit)
                if len(hits) >= 5:
                    break
            for hit in hits[:5]:
                content = f"笔记[{hit.get('title', '')}]: {hit.get('content', '')}"
                packets.append(
                    ContextPacket(
                        timestamp=datetime.now(),
                        content=content,
                        token_count=self.context_builder._count_tokens(content),
                        relevance_score=0.8,  # 非 0.5，避免触发 SentenceTransformer 重算
                        metadata={
                            "type": "note_context",
                            "source": "note_tool",
                            "note_id": hit.get("note_id"),
                        },
                    )
                )
            if hits:
                print(f"[NoteContext] 注入 {len(hits)} 条相关项目笔记")
        except Exception as e:
            print(f"[Warning] 注入笔记上下文失败: {e}")

        return packets

    def run(self, input_text: str, **kwargs) -> str:
        # 记录当前任务文本，供 _build_custom_context 检索笔记；其余完全复用父类 ReAct 循环
        self._last_input_text = input_text
        return super().run(input_text, **kwargs)