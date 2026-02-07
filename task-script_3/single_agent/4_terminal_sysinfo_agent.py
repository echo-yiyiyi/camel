from camel.agents import ChatAgent
from camel.toolkits.terminal_toolkit.terminal_toolkit import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelType


def main():
    # Create terminal and code execution toolkits
    terminal_toolkit = TerminalToolkit()
    code_execution_toolkit = CodeExecutionToolkit()

    # Create model config and model
    model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()
    model = ModelFactory.create(
        model_type=ModelType.DEFAULT,
        model_config_dict=model_config_dict,
    )

    # Create the agent with the two toolkits
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        tools=terminal_toolkit.get_tools() + code_execution_toolkit.get_tools(),
    )

    # Prompt to retrieve system info and print it in Python interpreter
    prompt = (
        "Retrieve system information using terminal commands and then print it using a Python interpreter. "
        "Use the terminal toolkit to get system info (e.g., uname -a, lscpu, free -h) and then use the code execution toolkit to print it. "
        "Show the output in the Python interpreter."
    )

    # Run the agent with the prompt
    response = agent.step(prompt)

    # Print the agent's response
    print(response.message)


if __name__ == "__main__":
    main()
