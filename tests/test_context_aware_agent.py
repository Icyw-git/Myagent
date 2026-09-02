# test_context_aware_agent.py
# ContextBuilder 背景模式 + ContextAwareAgent 接线冒烟
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_context_aware_agent.py
# search 全流程（需 .env 里 LLM + SERPAPI_API_KEY）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_context_aware_agent.py --search
# 记忆双轨全流程（需 .env 里 LLM；working 记忆，不依赖 Qdrant）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_context_aware_agent.py --memory
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime

from ContextBuilder import ContextBuilder
from contextbase import ContextConfig, ContextPacket
from Message import Message


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_builder(max_tokens: int = 2000) -> ContextBuilder:
    # 离线：不压 LLM；关掉 memory，避免 Qdrant/库依赖
    cfg = ContextConfig(
        max_tokens=max_tokens,
        enable_compression=False,
        reserve_ratio=0.0,  # 测结构时用满预算，避免和本测无关的裁剪
        min_relevance=0.1,
        relevance_weight=0.7,
        recency_weight=0.3,
        max_history=10,
    )
    builder = ContextBuilder(llm=None, config=cfg)
    builder.memory_tool = None
    # 避免 custom 包 relevance=0.5 时加载 SentenceTransformer
    builder._calculate_relevance = lambda content, query: 0.85
    return builder


def test_build_background_no_task_output() -> None:
    print("=== 1) build(include_task_sections=False)：只要背景，不要 [Task]/[Output] ===")
    builder = _make_builder()
    custom = [
        ContextPacket(
            content="用户偏好：回答用中文，居住武汉。",
            timestamp=datetime.now(),
            token_count=builder._count_tokens("用户偏好：回答用中文，居住武汉。"),
            relevance_score=0.8,  # 非 0.5，不触发相关性重算
            metadata={"type": "custom_context"},
        )
    ]
    out = builder.build(
        "我住在哪？",
        conversation_history=None,
        system_instructions="你是简洁助手。",
        custom_packets=custom,
        include_task_sections=False,
    )
    print(out[:500])
    _assert("[Role & Policies]" in out, "应含系统角色段")
    _assert("简洁助手" in out, "系统指令应进入 Role")
    _assert("武汉" in out, "custom_context 应进入背景")
    _assert("[Task]" not in out, "背景模式不应含 [Task]")
    _assert("[Output]" not in out, "背景模式不应含 [Output]")
    print("通过\n")


def test_build_qa_has_task_output() -> None:
    print("=== 2) build(默认)：独立问答应含 [Task]/[Output] ===")
    builder = _make_builder()
    out = builder.build(
        "今天天气如何？",
        system_instructions="你是助手。",
        include_task_sections=True,
    )
    print(out[:400])
    _assert("[Task]" in out and "今天天气如何？" in out, "应含 Task 与原问题")
    _assert("[Output]" in out, "问答模式应含 Output 指引")
    print("通过\n")


def test_system_type_and_history() -> None:
    print("=== 3) system_instructions 类型对齐 + max_history 不炸 ===")
    builder = _make_builder()
    history = [
        Message("上一轮问题A", "user"),
        Message("上一轮答案A", "assistant"),
        Message("上一轮问题B", "user"),
        Message("上一轮答案B", "assistant"),
    ]
    out = builder.build(
        "下一问",
        conversation_history=history,
        system_instructions="角色：测试员",
        include_task_sections=False,
    )
    print(out[:500])
    _assert("角色：测试员" in out, "system 包应被选中并进入 Role（非掉进杂烩后丢失）")
    _assert("上一轮" in out or "问题" in out, "对话历史应进入 Context")
    print("通过\n")


def test_agent_wires_context_config() -> None:
    print("=== 4) ContextAwareAgent 把 context_config 传进 builder ===")
    from my_react_agent_context import ContextAwareAgent
    from pipelines.registry.tools import build_hello_registry

    # 不调真 LLM：只检查构造与 background 组装；registry 用 hello_agents 风格
    class _DummyLLM:
        def invoke(self, messages, **kwargs):
            return "Thought: 已知答案\nAction: Finish[武汉]"

    cfg = ContextConfig(
        max_tokens=1500,
        enable_compression=False,
        reserve_ratio=0.0,
        relevance_weight=0.7,
        recency_weight=0.3,
    )
    agent = ContextAwareAgent(
        name="上下文助手",
        llm=_DummyLLM(),  # type: ignore[arg-type]
        tool_registry=build_hello_registry("search"),
        max_steps=3,
        system_prompt="你是带记忆的 ReAct 助手。",
        context_config=cfg,
        custom_context="用户住在武汉，喜欢简洁回答。",
    )
    _assert(agent.context_builder.config is agent.context_config, "builder 应共用同一 ContextConfig")
    agent.context_builder.memory_tool = None
    agent.context_builder._calculate_relevance = lambda c, q: 0.85

    answer = agent.run("我住在哪里？")
    print(f"Agent 回答: {answer}")
    _assert("武汉" in answer, "DummyLLM Finish 应带回武汉")
    _assert(len(agent.get_history()) >= 2, "结束后应写入 _history（user+assistant）")
    print("通过\n")


def test_online_search_full_flow() -> None:
    """全流程：ContextBuilder 背景 + ReAct 循环 + SerpAPI search。"""
    print("=== 5) 联网全流程：ContextAwareAgent + search ===")
    from dotenv import load_dotenv
    from hello_agents import HelloAgentsLLM
    from my_react_agent_context import ContextAwareAgent
    from pipelines.registry.tools import build_hello_registry

    load_dotenv()
    llm = HelloAgentsLLM()
    # 与 pipelines 同一 search（ReAct.search + SERPAPI_API_KEY）
    registry = build_hello_registry("search")
    cfg = ContextConfig(
        max_tokens=2500,
        enable_compression=False,
        reserve_ratio=0.0,
        relevance_weight=0.7,
        recency_weight=0.3,
    )
    agent = ContextAwareAgent(
        name="搜索上下文助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
        system_prompt=(
            "你是会使用工具的助手。需要实时/外部信息时必须调用 search 工具，"
            "格式为 Action: search[查询词]；信息足够后再 Finish[最终答案]。"
        ),
        context_config=cfg,
        custom_context="用户在武汉，关心出行天气，回答请用中文、简洁。",
    )
    agent.context_builder.memory_tool = None
    agent.context_builder._calculate_relevance = lambda c, q: 0.85

    question = "请搜索一下今天武汉的天气概况，并给出简短出行建议。"
    answer = agent.run(question)
    print(f"\n最终回答:\n{answer}\n")

    # 全流程验收：至少走过工具 Observation，且答案非空
    hist = "\n".join(agent.current_history)
    print(f"--- current_history ---\n{hist}\n-----------------------")
    _assert(bool(answer and answer.strip()), "最终答案不应为空")
    _assert(
        any(h.startswith("Observation:") for h in agent.current_history),
        "应至少调用过一次工具并写入 Observation（search 全流程）",
    )
    _assert(
        any("Action:" in h and "search" in h.lower() for h in agent.current_history)
        or "search" in hist.lower(),
        "轨迹中应出现 search 调用",
    )
    _assert(len(agent.get_history()) >= 2, "结束后应写入多轮 _history")
    # 背景里的武汉偏好：答案侧尽量沾边（天气/出行/武汉），不强制精确气象字段
    soft = any(k in answer for k in ("武汉", "天气", "出行", "温", "雨", "晴", "风"))
    _assert(soft, f"答案应与天气/出行相关，实际={answer[:200]}")
    print("通过\n")


def test_online_memory_full_flow() -> None:
    """联网双轨记忆：被动 build 召回 + 主动 memory 工具写入 + 下轮被动召回。"""
    print("=== 6) 联网全流程：被动召回 + 主动 memory ===")
    from dotenv import load_dotenv
    from hello_agents import HelloAgentsLLM, ToolRegistry
    from memory_src import MemoryTool, MemoryConfig
    from my_react_agent_context import ContextAwareAgent

    load_dotenv()
    llm = HelloAgentsLLM()

    # 隔离测试用户 + 仅 working（内存检索，不依赖 Qdrant/SerpAPI）
    mem_tool = MemoryTool(
        user_id="test_context_aware_memory",
        memory_config=MemoryConfig(),
        memory_types=["working"],
    )
    passive_fact = "用户住在武汉市武昌区紫荆花园7栋502室"
    add_ret = mem_tool.run(
        {
            "action": "add",
            "content": passive_fact,
            "memory_type": "working",
            "importance": 0.9,
        }
    )
    print(f"[预热写入] {add_ret}")

    hits = mem_tool.search_items("住在哪个小区", limit=5, min_importance=0.3)
    print(f"[search_items] 命中 {len(hits)} 条")
    _assert(hits, "写入后 search_items 应能命中")
    _assert(any("紫荆花园" in (h.content or "") for h in hits), "检索结果应含紫荆花园")

    cfg = ContextConfig(
        max_tokens=2500,
        enable_compression=False,
        reserve_ratio=0.0,
        relevance_weight=0.7,
        recency_weight=0.3,
    )
    agent = ContextAwareAgent(
        name="记忆上下文助手",
        llm=llm,
        tool_registry=ToolRegistry(),
        max_steps=6,
        system_prompt=(
            "你是带记忆的助手。背景 system 里可能已有相关记忆；"
            "用户要求「记住」时，必须用 memory 工具 add（JSON 参数）；"
            "不要编造未记录的事实。"
        ),
        context_config=cfg,
        memory_tool=mem_tool,
        register_memory_tool=True,
    )
    # 避免 custom 包 relevance=0.5 触发 SentenceTransformer 加载
    agent.context_builder._calculate_relevance = lambda c, q: 0.85

    # --- 被动：build 应带上预热记忆 ---
    bg = agent.context_builder.build(
        "我住在哪个小区？",
        agent._history,
        agent.system_prompt,
        include_task_sections=False,
    )
    print(f"[被动 build 预览]\n{bg[:600]}\n...")
    _assert("紫荆花园" in bg, "被动 build 的 system 背景应含紫荆花园")

    answer1 = agent.run("不要搜索外部。只根据背景记忆回答：我住在哪个小区？尽量说出完整地址。")
    print(f"\n[被动问答] {answer1}")
    _assert("紫荆花园" in answer1, f"答案应含紫荆花园，实际={answer1[:200]}")

    # --- 主动：本轮让模型 memory add，下轮被动应召回 ---
    agent.run(
        '请用 memory 工具记住这句话：用户的猫叫小豆。'
        '成功后 Finish[已记住]。'
        '调用示例：Action: memory[{"action":"add","content":"用户的猫叫小豆","memory_type":"working","importance":0.9}]'
    )
    hist1 = "\n".join(agent.current_history)
    print(f"--- run2 current_history ---\n{hist1}\n---")
    active_wrote = any(
        h.startswith("Observation:") and ("已添加" in h or "记忆已添加" in h or "小豆" in h)
        for h in agent.current_history
    ) or any("小豆" in (h.content or "") for h in mem_tool.search_items("猫", limit=3))
    _assert(active_wrote, "主动 memory add 后库中应能查到「小豆」或 Observation 显示添加成功")

    answer2 = agent.run("不要编造。根据记忆回答：我的猫叫什么名字？只答名字。")
    print(f"\n[下轮被动召回] {answer2}")
    _assert("小豆" in answer2, f"下轮应被动召回猫名小豆，实际={answer2[:200]}")
    _assert(len(agent.get_history()) >= 4, "多轮后 _history 应累积")
    print("通过\n")


def main() -> None:
    online = "--online" in sys.argv
    search_only = "--search" in sys.argv
    memory_only = "--memory" in sys.argv
    if search_only:
        # 只跑 search 全流程（需 LLM + SERPAPI）
        test_online_search_full_flow()
        print("search 全流程测试通过。")
        return
    if memory_only:
        test_online_memory_full_flow()
        print("memory 全流程测试通过。")
        return

    test_build_background_no_task_output()
    test_build_qa_has_task_output()
    test_system_type_and_history()
    test_agent_wires_context_config()
    if online:
        test_online_search_full_flow()
        test_online_memory_full_flow()
    else:
        print("（跳过联网；全流程请加 --search / --memory 或 --online）")
    print("全部离线测试通过。")


if __name__ == "__main__":
    main()
