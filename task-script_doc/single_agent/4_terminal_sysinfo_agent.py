from camel.agents import ChatAgent
from camel.toolkits import TerminalToolkit, CodeExecutionToolkit


def main():
    # Initialize terminal and code execution toolkits
    terminal_toolkit = TerminalToolkit(working_directory="./workspace")
    code_exec_toolkit = CodeExecutionToolkit()

    # Combine tools from both toolkits
    tools = terminal_toolkit.get_tools() + code_exec_toolkit.get_tools()

    # System message to describe the agent's role
    system_message = "You are an assistant with access to terminal and code execution tools. " \
                     "Your task is to retrieve system information and print it using a Python interpreter."

    # Create the agent with the combined tools
    agent = ChatAgent(system_message=system_message, tools=tools)
    agent.reset()

    # Step 1: Retrieve system information using terminal tool
    sys_info_cmd = "uname -a"
    response = agent.step(f"Execute the terminal command to get system information: {sys_info_cmd}")
    print("Terminal command output:", response.msg.content)

    # Step 2: Use code execution tool to print the system information in Python interpreter
    python_code = f"print('System Information:')\nprint('''{response.msg.content}''')"
    response = agent.step(f"Execute the following Python code:\n{python_code}")
    print("Python interpreter output:", response.msg.content)


if __name__ == '__main__':
    main()
