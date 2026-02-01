from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.toolkits.search_toolkit import SearchToolkit
from camel.tasks.task import Task
from camel.types import ModelPlatformType, ModelType


def main():
    # Create Python Programmer worker with code execution tools
    code_exec_toolkit = CodeExecutionToolkit(verbose=True)
    code_exec_tools = code_exec_toolkit.get_tools()

    python_programmer_agent = ChatAgent(
        system_message="You are a Python programmer who can write and execute code to solve problems.",
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
        tools=code_exec_tools,
    )

    # Create Info Collector worker with LinkUp tools for math problems
    search_toolkit = SearchToolkit()
    linkup_tools = search_toolkit.get_tools()

    info_collector_agent_kwargs = {
        "system_message": "You are an info collector who uses LinkUp tools to find information and solve math problems.",
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
        "tools": linkup_tools,
    }

    # Create another example role - a simple role-playing worker
    guide_sysmsg = "You are a helpful guide who assists with general questions."
    planner_sysmsg = "You are a planner who organizes tasks efficiently."

    # Create workforce
    workforce = Workforce(description="Multi-role Workforce")

    # Add Python Programmer worker
    workforce.add_single_agent_worker(
        description="Python Programmer Worker",
        worker=python_programmer_agent,
    )

    # Add Info Collector worker
    workforce.add_single_agent_worker(
        description="Info Collector Worker",
        worker=ChatAgent(**info_collector_agent_kwargs),
    )

    # Add Role Playing worker
    workforce.add_role_playing_worker(
        description="Role Playing Worker",
        assistant_role_name="guide",
        user_role_name="planner",
        assistant_agent_kwargs={
            "system_message": guide_sysmsg,
            "model": ModelFactory.create(
                model_platform=ModelPlatformType.DEFAULT,
                model_type=ModelType.DEFAULT,
            ),
        },
        user_agent_kwargs={
            "system_message": planner_sysmsg,
            "model": ModelFactory.create(
                model_platform=ModelPlatformType.DEFAULT,
                model_type=ModelType.DEFAULT,
            ),
        },
        chat_turn_limit=1,
    )

    # Create a sample task
    task = Task(content="Solve a math problem and plan the solution.", id="task-1")

    # Process the task
    workforce.process_task(task)

    # Print workforce logs
    print("Workforce log tree:")
    print(workforce.get_workforce_log_tree())


if __name__ == "__main__":
    main()
