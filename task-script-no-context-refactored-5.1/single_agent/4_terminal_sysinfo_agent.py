import os
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.terminal_toolkit.terminal_toolkit import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.configs import ChatGPTConfig
from camel.types import ModelType


def main():
    # Define working directory for terminal toolkit
    workspace_dir = os.path.abspath("./workspace")

    # System message for the agent
    sys_msg = (
        "You are an assistant with terminal and code execution tools. "
        "Retrieve system information using terminal tools and print it using the Python interpreter."
    )

    # Create terminal toolkit
    terminal_toolkit = TerminalToolkit(working_directory=workspace_dir)
    terminal_tools = terminal_toolkit.get_tools()

    # Create code execution toolkit with Jupyter interpreter
    code_exec_toolkit = CodeExecutionToolkit(sandbox="jupyter", verbose=True)
    code_exec_tools = code_exec_toolkit.get_tools()

    # Combine tools
    tools = terminal_tools + code_exec_tools

    # Create model config
    model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()

    # Create model
    model = ModelFactory.create(model_type=ModelType.DEFAULT, model_config_dict=model_config_dict)

    # Create agent with system message, model, and tools
    agent = ChatAgent(system_message=sys_msg, model=model, tools=tools)
    agent.reset()

    # User message to retrieve system info and print it in Python
    usr_msg = (
        "Retrieve system information such as OS name, kernel version, and CPU info using terminal commands. "
        "Then, use the code execution tool to print the retrieved system information in Python. "
        "Format the output clearly."
    )

    # Get response
    response = agent.step(usr_msg)

    # Print the tool calls info
    print("Tool calls info:", response.info.get('tool_calls', []))

    # Print the agent's reply
    print("Agent reply:")
    print(response.msg.content)


if __name__ == '__main__':
    main()
