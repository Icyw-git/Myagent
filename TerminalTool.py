import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

class TerminalTool:
    def __init__(
        self,
        workspace: str,
        allow_cd: bool = True,
        timeout: int = 30,
        max_output_size: int = 10000,
        allowed_commands: Optional[Iterable[str]] = None,
    ):
        # 错误记录：曾用 os.path.abspath 得到 str，但 _handle_cd 用 .parent / 运算符 / relative_to，
        # 需要 Path。统一成 Path，execute 时 cwd=str(...) 即可。
        self.workspace=Path(workspace).resolve()
        self.workspace.mkdir(parents=True,exist_ok=True)
        self.current_dir=self.workspace
        self.allow_cd=allow_cd
        self.timeout=timeout
        self.max_output_size=max_output_size
        self.allowed_commands = {
            command.lower()
            for command in (allowed_commands or ("python", "python.exe", "git", "git.exe", "pip", "pip.exe", "pytest", "pytest.exe", "echo", "dir"))
        }


    def _execute_command(self,command:str)->str:
        try:
            parts = self._parse_command(command)
            if not parts:
                return '错误：命令为空'

            name = Path(parts[0]).name.lower()
            if name == "cd":
                return self._handle_cd(parts)
            if name not in self.allowed_commands:
                return f'命令被拒绝：{name} 不在允许列表中'
            if name == "echo":
                return " ".join(parts[1:])
            if name == "dir":
                return "\n".join(entry.name for entry in self.current_dir.iterdir()) or '目录为空'

            result=subprocess.run(
                parts,
                shell=False,
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,

                env=os.environ.copy()
            )

            output=result.stdout
            if result.stderr:
                output+=f"\n[stderr]\n{result.stderr}"

            if len(output)>self.max_output_size:
                output=output[:self.max_output_size]
                output+=f'\n\n输出被截断（超过{self.max_output_size}字节）'

            if result.returncode !=0:
                output=f'命令执行失败，返回码：{result.returncode}\n\n{output}'

            return output if output else '命令执行成功（无输出）'

        except subprocess.TimeoutExpired:
            return f'命令执行超时（超过{self.timeout}秒）'
        except Exception as e:
            return f'命令执行异常：{str(e)}'

    @staticmethod
    def _parse_command(command: str) -> List[str]:
        parts = shlex.split(command, posix=False)
        return [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in parts]


    def _handle_cd(self,parts:List[str])->str:
        if not self.allow_cd:
            return 'cd命令被禁用'

        if len(parts)<2:
            return f'当前目录：{self.current_dir}'

        target_dir=parts[1]

        if target_dir=='..':
            new_dir=self.current_dir.parent
        elif target_dir=='.':
            new_dir=self.current_dir
        elif target_dir=='~':
            new_dir=self.workspace
        else:
            new_dir=(self.current_dir / target_dir).resolve()

        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f'错误：无法切换到{new_dir}，超出工作空间范围'

        if not new_dir.exists():
            return f'错误：目录{new_dir}不存在'
        if not new_dir.is_dir():
            return f'错误：{new_dir}不是一个目录'

        self.current_dir=new_dir

        return f'已切换到目录：{self.current_dir}'





