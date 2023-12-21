import abc

from langchain.tools import BaseTool


class Calculator(BaseTool, abc.ABC):
    name = "Calculator"
    description = "Useful for when you need to answer questions about math"

    def __init__(self):
        super().__init__()

    def _run(self, para: str) -> str:
        import math
        para = para.replace("^", "**")
        para = para.replace(",", "")
        if "sqrt" in para:
            para = para.replace("sqrt", "math.sqrt")
        elif "log" in para:
            para = para.replace("log", "math.log")
        return eval(para)


class Weather(BaseTool, abc.ABC):
    name = "Weather"
    description = "Get the Temperature of given day"

    def __init__(self):
        super().__init__()

    def _run(self, day: str) -> str:
        return "今天天气晴朗，气温0度到10度"


class CurrentDay(BaseTool, abc.ABC):
    name = "CurrentDay"
    description = "get the which day of the year it is"

    def __init__(self):
        super().__init__()

    def _run(self) -> str:
        import datetime
        import time
        return datetime.datetime.fromtimestamp(time.time()).strftime('%d')


if __name__ == "__main__":
    calculator_tool = Calculator()
    result = calculator_tool.run("sqrt(2) + 3")
    print(result)

    # day = CurrentDay().run({})
    # print(day)
    # temperature = Weather().run(day)
    # print(temperature)
