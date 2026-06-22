from ReAct import ToolExecutor

def test_register_and_get_tool():
    executor=ToolExecutor()

    def add(a,b):
        return a+b

    executor.register_tool('add','加法',add)
    func=executor.getTool('add')

    assert func(1,2)==3


def test_get_available_tools_format():
    executor=ToolExecutor()

    def greet(name):
        return f'Hello {name}'
    executor.register_tool('greet','问候',greet)
    func=executor.getTool('greet')
    output=executor.getAvailableTools()

    assert 'greet' in output

    assert '问候' in output
    assert ':' in output

