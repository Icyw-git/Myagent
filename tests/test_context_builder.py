# test_context_builder.py
# 联网冒烟：ContextBuilder 的 token 统计 / LLM 截断 / _compress
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_context_builder.py
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from ContextBuilder import ContextBuilder
from contextbase import ContextConfig

load_dotenv()


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    llm = HelloAgentsLLM()
    builder = ContextBuilder(llm=llm, config=ContextConfig(max_tokens=3000))

    print("=== 1) _count_tokens ===")
    n_en = builder._count_tokens("hello world")
    n_zh = builder._count_tokens("今天去武汉东湖散步。")
    print(f"英文 tokens={n_en}, 中文 tokens={n_zh}")
    _assert(n_en > 0 and n_zh > 0, "token 计数应 > 0")

    print("\n=== 2) 短文本不调 LLM（直接返回）===")
    short = "短文本无需压缩"
    out_short = builder._truncate_text(short, max_tokens=200)
    print(f"输出: {out_short}")
    _assert(out_short == short, "短文本应原样返回")

    print("\n=== 3) 联网 LLM 截断 _truncate_text ===")
    long_text = (
        "用户档案：姓名 icy，居住武汉，职业 Python 开发者。"
        "近期在做 myagent 项目，模块包括 Memory、ReAct、ContextBuilder、评测双轨。"
        "偏好：回答简洁、尽量少改无关代码、注释用中文。"
        "今日任务：验证上下文超限时，能否用 LLM 把冗长段落压成保留关键事实的短摘要。"
    ) * 15
    before = builder._count_tokens(long_text)
    budget = 80
    print(f"原文 tokens≈{before}，预算={budget}")
    _assert(before > budget, "原文应明显超预算，否则测不到 LLM 路径")

    truncated = builder._truncate_text(long_text, max_tokens=budget)
    after = builder._count_tokens(truncated)
    print(f"截断后 tokens≈{after}")
    print(f"截断预览:\n{truncated[:400]}\n...")
    _assert(bool(truncated.strip()), "截断结果不应为空")
    _assert(after <= budget, f"截断后应 ≤{budget} tokens，实际={after}")
    # 粗检：摘要里应还沾一点原文关键信息（不强制精确，防 LLM 胡写空话）
    hit = any(k in truncated for k in ("武汉", "Python", "myagent", "ContextBuilder", "icy"))
    _assert(hit, "摘要应保留至少一处原文关键信息")

    print("\n=== 4) 联网 _compress（分段 + LLM 截断）===")
    ctx = "\n\n".join(
        [
            "[Role & Policies]\n你是一个简洁的中文助手。",
            "[Task]\n根据上下文概括用户是谁、在做什么。",
            "[Context]\n" + long_text,
        ]
    )
    before_c = builder._count_tokens(ctx)
    limit = 150
    print(f"压缩前 tokens≈{before_c}，上限={limit}")
    compressed = builder._compress(ctx, max_tokens=limit)
    after_c = builder._count_tokens(compressed)
    print(f"压缩后 tokens≈{after_c}")
    print(f"压缩预览:\n{compressed[:500]}\n...")
    _assert(after_c <= limit, f"压缩后应 ≤{limit} tokens，实际={after_c}")
    _assert("[Role & Policies]" in compressed or "助手" in compressed, "应尽量保留靠前段落")

    print("\n全部联网测试通过。")


if __name__ == "__main__":
    main()
