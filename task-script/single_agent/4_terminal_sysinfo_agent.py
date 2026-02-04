import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, CodeExecutionToolkit
from camel.types import ModelPlatformType, ModelType

# Define working directory for terminal toolkit
working_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")

# Create terminal toolkit and get tools
terminal_toolkit = TerminalToolkit(working_directory=working_dir)
terminal_tools = terminal_toolkit.get_tools()

# Create code execution toolkit and get tools
code_exec_toolkit = CodeExecutionToolkit()
code_exec_tools = code_exec_toolkit.get_tools()

# Combine tools
tools = terminal_tools + code_exec_tools

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# System message for the agent
system_message = (
    "You are a system information assistant. "
    "You have access to terminal tools to retrieve system information, "
    "and code execution tools to process and print the information in Python."
)

# Create agent
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=tools,
)
agent.reset()

# User prompt to retrieve system info and print it in Python
user_prompt = (
    "Use terminal tools to get detailed system information (e.g., uname -a, cat /etc/os-release). "
    "Then use code execution tools to print the retrieved information in a Python interpreter. "
    "Show the output clearly."
)

# Get response
response = agent.step(user_prompt)

# Print all agent messages
for msg in response.msgs:
    print(msg.content)
