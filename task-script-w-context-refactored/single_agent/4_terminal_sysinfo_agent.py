import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.types import ModelType

# Define workspace directory
base_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "workspace")

# Create toolkits
terminal_toolkit = TerminalToolkit(working_directory=workspace_dir)
code_exec_toolkit = CodeExecutionToolkit(sandbox="internal_python")

# Combine tools from both toolkits
tools = terminal_toolkit.get_tools() + code_exec_toolkit.get_tools()

# Create model
model = ModelFactory.create(model_type=ModelType.DEFAULT)

# Define system message
system_message = (
    "You are an assistant with access to terminal tools and code execution tools. "
    "Retrieve system information using terminal commands, then print it in a Python interpreter."
)

# Create agent
agent = ChatAgent(system_message=system_message, model=model, tools=tools)
agent.reset()

# Step 1: Retrieve system information
response = agent.step("Run 'uname -a' to get system information and save the output.")
print("Step 1 response:", response.msg.content)

# Step 2: Print the retrieved system information in Python interpreter
response = agent.step(
    "Now use the code execution tool to print the system information you retrieved. "
    "Use Python code to print the output from the previous step."
)
print("Step 2 response:", response.msg.content)


if __name__ == "__main__":
    pass
