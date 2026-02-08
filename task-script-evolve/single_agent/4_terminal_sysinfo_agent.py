import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.terminal_toolkit.terminal_toolkit import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.configs import ChatGPTConfig
from camel.types import ModelType

# Setup working directory for terminal toolkit
working_directory = os.getcwd()

# Create terminal tools
terminal_tools = TerminalToolkit(working_directory=working_directory).get_tools()

# Create code execution tools
code_exec_tools = CodeExecutionToolkit().get_tools()

# Combine tools
all_tools = terminal_tools + code_exec_tools

# Create model
model_config = ChatGPTConfig(temperature=0.0).as_dict()
model = ModelFactory.create(
    model_type=ModelType.DEFAULT,
    model_config_dict=model_config,
)

# System message
system_message = (
    "You are an assistant with access to terminal and code execution tools. "
    "Retrieve system information using terminal tools and print it in the Python interpreter."
)

# Create agent
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=all_tools,
)
agent.reset()

# Prompt to retrieve system info and print it in Python interpreter
user_prompt = (
    "Retrieve system information such as OS name, kernel version, CPU info, and memory info using terminal commands. "
    "Then print the collected information in the Python interpreter."
)

response = agent.step(user_prompt)

# Print agent response
print("Agent response:")
for msg in response.msgs:
    print(msg.content)
