from typing import Dict, List, Any

from Tool import ToolParameter,ToolRegistry,Tool


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(name='calculator', description='一个计算器工具，可以进行基本的数学运算。输入为数学表达式，输出为计算结果。')

    def run(self,parameters:Dict[str,Any]) ->str:
        expression=parameters.get('expression')
        if not expression:
            return '错误：缺少数学表达式参数'
        try:
            # 使用eval计算数学表达式，注意安全性问题
            result=eval(expression, {"__builtins__": None}, {})
            return str(result)
        except Exception as e:
            return f'计算错误：{str(e)}'



    def get_parameters(self) ->List[ToolParameter]:
        return [
            ToolParameter(
                name='expression',
                type='string',
                description='数学表达式，例如"2+2"或"sqrt(16)"',
                required=True
            )
        ]


def create_calculator_registry():
    registry=ToolRegistry()
    calculator=CalculatorTool()
    registry.register_tool(calculator)
    return registry