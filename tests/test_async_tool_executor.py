import asyncio

from async_tool_executor import AsyncToolExecutor


class EchoRegistry:
    def execute_tool(self, tool_name: str, input_data: str) -> str:
        return f"{tool_name}:{input_data}"


def test_execute_tool_async():
    async def run():
        with AsyncToolExecutor(EchoRegistry()) as executor:
            result = await executor.execute_tool_async("echo", "one")
        assert result == "echo:one"

    asyncio.run(run())


def test_execute_task_async_preserves_input_order():
    async def run():
        tasks = [
            {"tool_name": "echo", "input_data": value}
            for value in ("first", "second", "third")
        ]
        with AsyncToolExecutor(EchoRegistry(), max_workers=3) as executor:
            return await executor.execute_task_async(tasks)

    assert asyncio.run(run()) == ["echo:first", "echo:second", "echo:third"]
