"""
Multi-role workforce script with Python Programmer and Info Collector workers.
"""

import asyncio

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import CodeExecutionToolkit, MathToolkit, MCPToolkit
from camel.types import ModelPlatformType, ModelType


def create_multi_role_workforce():
    workforce = Workforce("Multi-Role Workforce")

    model_platform = ModelPlatformType.DEFAULT
    model_type = ModelType.DEFAULT

    # Python Programmer worker with code execution tools
    code_exec_toolkit = CodeExecutionToolkit()
    python_programmer_agent = ChatAgent(
        model=ModelFactory.create(model_platform=model_platform, model_type=model_type),
        tools=code_exec_toolkit.get_tools(),
    )
    workforce.add_single_agent_worker("Python Programmer", python_programmer_agent)

    # Info collector worker with MCP toolkit and Math toolkit for math problems
    # Info collector worker with Math toolkit only (MCPToolkit omitted due to config requirements)
    math_toolkit = MathToolkit()
    math_toolkit = MathToolkit()
    math_toolkit = MathToolkit()
    info_collector_tools = math_toolkit.get_tools()
    info_collector_agent = ChatAgent(
        model=ModelFactory.create(model_platform=model_platform, model_type=model_type),
        tools=info_collector_tools,
    )
    workforce.add_single_agent_worker("Info Collector", info_collector_agent)
    math_toolkit = MathToolkit()
    # Other defined roles (example: general assistant)
    general_agent = ChatAgent(
        model=ModelFactory.create(model_platform=model_platform, model_type=model_type)
    )
    workforce.add_single_agent_worker("General Assistant", general_agent)

    return workforce


async def main():
    workforce = create_multi_role_workforce()

    task = Task(
        content="Solve math problems and write Python code as needed.",
        id="task_001",
    )
    workforce.process_task(task)

    await workforce.start()


if __name__ == "__main__":
    asyncio.run(main())
