"""
Multi-role Workforce example:
- Python Programmer worker with code execution tools
- Info Collector worker with LinkedIn toolkit for math problems
- Other defined roles
"""

from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.toolkits import CodeExecutionToolkit, LinkedInToolkit, MathToolkit


def main():
    workforce = Workforce(description="Multi-role Workforce for programming and info collection")

    # Python Programmer worker with code execution tools
    python_programmer_agent = ChatAgent(
        system_message="You are a Python programmer skilled in code execution.",
        tools=CodeExecutionToolkit().get_tools(),
    )
    workforce.add_single_agent_worker(
        description="Python Programmer with code execution tools",
        worker=python_programmer_agent,
    )

    # Info Collector worker with LinkedIn toolkit for math problems
    info_collector_agent = ChatAgent(
        system_message="You are an info collector skilled in using LinkedIn tools for math problems.",
        tools=LinkedInToolkit().get_tools(),
    )
    workforce.add_single_agent_worker(
        description="Info Collector with LinkedIn tools for math problems",
        worker=info_collector_agent,
    )

    # Additional worker: Math expert with math toolkit
    math_expert_agent = ChatAgent(
        system_message="You are a math expert skilled in solving math problems.",
        tools=MathToolkit().get_tools(),
    )
    workforce.add_single_agent_worker(
        description="Math Expert with math toolkit",
        worker=math_expert_agent,
    )

    print("Multi-role Workforce created with Python Programmer, Info Collector, and Math Expert workers.")


if __name__ == "__main__":
    main()
