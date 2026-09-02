# test_terminal_tool.py
# TerminalTool 离线冒烟：init / execute / cd 沙箱 / 超时 / 输出截断
# 运行（项目根目录）:
#   D:\Anaconda_envs\envs\aitest01_py310\python.exe tests/test_terminal_tool.py
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from TerminalTool import TerminalTool


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_terminal_tool() -> None:
    with tempfile.TemporaryDirectory(prefix="terminal_tool_test_") as tmp:
        workspace = Path(tmp) / "ws"
        sub = workspace / "sub"
        nested = sub / "deep"
        nested.mkdir(parents=True)
        (workspace / "hello.txt").write_text("hello terminal\n", encoding="utf-8")
        (sub / "a.txt").write_text("in sub\n", encoding="utf-8")

        tool = TerminalTool(workspace=str(workspace), allow_cd=True, timeout=10, max_output_size=200)

        print("=== 1) 初始化 ===")
        _assert(tool.workspace == workspace.resolve(), "workspace 应为 resolve 后的 Path")
        _assert(tool.current_dir == tool.workspace, "起始目录应为 workspace")
        _assert(tool.workspace.is_dir(), "workspace 目录应存在")
        print(f"workspace={tool.workspace}")
        print("通过\n")

        print("=== 2) _execute_command：正常输出 ===")
        # Windows / Unix 通用：echo
        out = tool._execute_command("echo hello_world")
        print(f"echo -> {out!r}")
        _assert("hello_world" in out, f"echo 应含 hello_world，实际={out!r}")
        print("通过\n")

        print("=== 3) _execute_command：失败返回码 ===")
        # Windows: cmd /c exit 1；跨平台用 python 更稳
        fail = tool._execute_command(
            f'"{sys.executable}" -c "import sys; sys.exit(2)"'
        )
        print(f"exit2 -> {fail[:120]!r}...")
        _assert("返回码" in fail or "失败" in fail, f"非零退出应标记失败，实际={fail[:200]}")
        print("通过\n")

        print("=== 4) _execute_command：输出截断 ===")
        # 生成超过 max_output_size=200 的输出
        big = tool._execute_command(
            f'"{sys.executable}" -c "print(\'X\'*500)"'
        )
        print(f"截断后长度={len(big)}, 尾部={big[-40:]!r}")
        _assert("输出被截断" in big, "超长输出应提示截断")
        _assert(len(big) < 500 + 80, "截断后总长应明显小于原文")
        print("通过\n")

        print("=== 5) _execute_command：超时 ===")
        slow = TerminalTool(workspace=str(workspace), timeout=1, max_output_size=10000)
        t0 = time.time()
        timed = slow._execute_command(
            f'"{sys.executable}" -c "import time; time.sleep(5)"'
        )
        elapsed = time.time() - t0
        print(f"timeout 结果={timed!r}, elapsed≈{elapsed:.1f}s")
        _assert("超时" in timed, f"应返回超时信息，实际={timed}")
        # Windows + shell=True 时杀进程可能偏慢，只要求别无限挂着
        _assert(elapsed < 15, f"超时路径不应挂死，elapsed={elapsed}")
        print("通过\n")

        print("=== 6) _handle_cd：子目录 / . / .. / ~ ===")
        r1 = tool._handle_cd(["cd", "sub"])
        print(r1)
        _assert("已切换" in r1, "应能进入 sub")
        _assert(tool.current_dir == sub.resolve(), f"current_dir 应为 sub，实际={tool.current_dir}")

        r_dot = tool._handle_cd(["cd", "."])
        _assert(tool.current_dir == sub.resolve(), "cd . 应保持原目录")

        r_deep = tool._handle_cd(["cd", "deep"])
        _assert(tool.current_dir == nested.resolve(), "应进入 deep")

        r_up = tool._handle_cd(["cd", ".."])
        _assert(tool.current_dir == sub.resolve(), "cd .. 应回到 sub")

        r_home = tool._handle_cd(["cd", "~"])
        print(r_home)
        _assert(tool.current_dir == tool.workspace, "cd ~ 应回到 workspace 根")
        print("通过\n")

        print("=== 7) _handle_cd：沙箱越界 / 不存在 / 非目录 ===")
        # 越界：尝试跳到 workspace 之外
        outside = tool._handle_cd(["cd", ".."])  # 已在 workspace 根，再 .. 会出界
        # 若当前在 workspace，parent 可能仍 relative_to 失败
        tool.current_dir = tool.workspace
        outside = tool._handle_cd(["cd", ".."])
        print(f"越界 cd .. -> {outside}")
        _assert("超出工作空间" in outside or "错误" in outside, f"应拒绝越界，实际={outside}")
        _assert(tool.current_dir == tool.workspace, "越界后 current_dir 不应改变")

        miss = tool._handle_cd(["cd", "no_such_dir"])
        _assert("不存在" in miss, f"不存在目录应报错，实际={miss}")

        not_dir = tool._handle_cd(["cd", "hello.txt"])
        _assert("不是一个目录" in not_dir, f"对文件 cd 应报错，实际={not_dir}")

        bare = tool._handle_cd(["cd"])
        _assert("当前目录" in bare, f"无参数 cd 应打印当前目录，实际={bare}")
        print("通过\n")

        print("=== 8) allow_cd=False ===")
        locked = TerminalTool(workspace=str(workspace), allow_cd=False)
        denied = locked._handle_cd(["cd", "sub"])
        _assert("禁用" in denied, f"禁用 cd 时应拒绝，实际={denied}")
        _assert(locked.current_dir == locked.workspace, "禁用后目录不变")
        print("通过\n")

        print("=== 9) 切换目录后命令在新 cwd 执行 ===")
        tool2 = TerminalTool(workspace=str(workspace))
        tool2._handle_cd(["cd", "sub"])
        # 用 python 读当前目录下的 a.txt
        read_out = tool2._execute_command(
            f'"{sys.executable}" -c "print(open(\'a.txt\',encoding=\'utf-8\').read().strip())"'
        )
        print(f"在 sub 读 a.txt -> {read_out!r}")
        _assert("in sub" in read_out, f"应读到 sub/a.txt，实际={read_out}")
        print("通过\n")

        print("全部 TerminalTool 测试通过。")


if __name__ == "__main__":
    test_terminal_tool()
