import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, CodeExecutionToolkit
from camel.types import ModelType

# Define system message
system_message = (
    "You are an assistant with access to terminal and code execution tools. "
    "Retrieve system information using terminal commands and print it in a Python interpreter."
)

# Setup working directory
working_directory = os.getcwd()

# Create toolkits
terminal_toolkit = TerminalToolkit(working_directory=working_directory)
code_exec_toolkit = CodeExecutionToolkit(sandbox="internal_python")

# Get tools from toolkits
terminal_tools = terminal_toolkit.get_tools()
code_exec_tools = code_exec_toolkit.get_tools()

# Combine tools
tools = terminal_tools + code_exec_tools

# Create model
model = ModelFactory.create(
    model_type=ModelType.DEFAULT,
)

# Create agent
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=tools,
)

agent.reset()

# User prompt to retrieve system info and print it in Python interpreter
user_prompt = (
    "Retrieve system information such as OS details, CPU info, and memory usage "
    "using terminal commands. Then print the retrieved information in a Python interpreter."
)

# Step the agent
response = agent.step(user_prompt)

# Print the final response content
print(response.msg.content)
