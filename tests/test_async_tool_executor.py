"""
测试 AsyncToolExecutor 的异步并行执行速度。
对比串行执行 vs 并行执行，直观展示加速效果。
"""
import asyncio
import time
import sys
import os

# 强制使用 UTF-8 编码输出，避免 Windows cmd GBK 编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from async_tool_executor import AsyncToolExecutor
from Tool import Tool, ToolParameter, ToolRegistry
from typing import Dict, Any, List


# ========== 1. 创建耗时工具（模拟 I/O 密集型操作） ==========

class SleepTool(Tool):
    """模拟耗时操作的工具，每次执行休眠指定秒数"""

    def __init__(self):
        super().__init__(
            name='sleep',
            description='休眠指定秒数，模拟耗时操作（如网络请求、文件I/O）'
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        seconds = float(parameters.get('seconds', 1))
        time.sleep(seconds)
        return f"休眠 {seconds}s 完成"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name='seconds',
                type='float',
                description='休眠秒数',
                required=False,
                default=1.0,
            )
        ]


# ========== 2. 准备测试环境 ==========

def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(SleepTool())
    return registry


# ========== 3. 串行执行测试 ==========

async def test_serial(executor: AsyncToolExecutor, task_count: int = 5, sleep_seconds: float = 1.0):
    """串行执行：逐个等待每个任务完成后再启动下一个"""
    print(f"\n{'='*60}")
    print(f"[串行] 逐个等待执行，共 {task_count} 个任务，每个耗时 {sleep_seconds}s")
    print(f"{'='*60}")

    start = time.perf_counter()

    for i in range(task_count):
        task_start = time.perf_counter()
        result = await executor.execute_tool_async('sleep', {'seconds': sleep_seconds})
        task_elapsed = time.perf_counter() - task_start
        print(f"  [{i+1}/{task_count}] {result} (耗时 {task_elapsed:.2f}s)")

    total = time.perf_counter() - start
    print(f"\n  [结果] 串行总耗时: {total:.2f}s")
    return total


# ========== 4. 并行执行测试 ==========

async def test_parallel(executor: AsyncToolExecutor, task_count: int = 5, sleep_seconds: float = 1.0):
    """并行执行：asyncio.gather 同时启动所有任务"""
    print(f"\n{'='*60}")
    print(f"[并行] asyncio.gather 同时启动，共 {task_count} 个任务，每个耗时 {sleep_seconds}s")
    print(f"{'='*60}")

    tasks_list = [
        {'tool_name': 'sleep', 'parameters': {'seconds': sleep_seconds}}
        for _ in range(task_count)
    ]

    start = time.perf_counter()
    results = await executor.execute_tools_parallel(tasks_list)
    total = time.perf_counter() - start

    for i, r in enumerate(results):
        print(f"  [{i+1}/{task_count}] {r}")

    print(f"\n  [结果] 并行总耗时: {total:.2f}s")
    return total


# ========== 5. 主测试入口 ==========

async def main():
    print("=" * 60)
    print("  AsyncToolExecutor 异步并行速度对比测试")
    print("=" * 60)

    TASK_COUNT = 6
    SLEEP_SECONDS = 1.0

    registry = create_registry()

    # 使用 6 个 worker 线程，这样 6 个任务可以完全并行
    executor = AsyncToolExecutor(registry, max_workers=TASK_COUNT)

    serial_time = await test_serial(executor, TASK_COUNT, SLEEP_SECONDS)
    parallel_time = await test_parallel(executor, TASK_COUNT, SLEEP_SECONDS)

    # ========== 6. 对比总结 ==========
    speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f">>> 对比总结 <<<")
    print(f"{'='*60}")
    print(f"  任务数量:     {TASK_COUNT}")
    print(f"  单任务耗时:   {SLEEP_SECONDS}s")
    print(f"  线程池大小:   {TASK_COUNT}")
    print(f"  ------------------------------")
    print(f"  串行总耗时:   {serial_time:.2f}s")
    print(f"  并行总耗时:   {parallel_time:.2f}s")
    print(f"  加速比:       {speedup:.1f}x")
    print(f"  节省时间:     {serial_time - parallel_time:.2f}s")
    print(f"{'='*60}")

    if speedup >= TASK_COUNT * 0.8:
        print(f"\n[结论] 并行执行接近理论最大值 ({TASK_COUNT}x)，异步效果显著!")
    elif speedup >= 2.0:
        print(f"\n[结论] 并行执行有明显加速 ({speedup:.1f}x)，异步方案有效。")
    else:
        print(f"\n[结论] 加速效果不明显，可能受线程池大小或 GIL 影响。")


if __name__ == '__main__':
    asyncio.run(main())