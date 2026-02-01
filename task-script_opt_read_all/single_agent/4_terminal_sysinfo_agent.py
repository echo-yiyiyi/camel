import os

from camel.agents import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.toolkits.terminal_toolkit.terminal_toolkit import TerminalToolkit
from camel.types import ModelPlatformType, ModelType

# Define workspace directory
workspace_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")

# Initialize terminal toolkit
terminal_toolkit = TerminalToolkit(working_directory=workspace_dir)
terminal_tools = terminal_toolkit.get_tools()

# Initialize code execution toolkit with internal python interpreter
code_exec_toolkit = CodeExecutionToolkit(sandbox="internal_python")
code_exec_tools = code_exec_toolkit.get_tools()

# Combine tools
tools = terminal_tools + code_exec_tools

# System message for the agent
system_message = (
    "You are an assistant with access to terminal tools and code execution tools. "
    "Retrieve system information using terminal tools and then print it using the Python interpreter."
)

# Model config
model_config = ChatGPTConfig(temperature=0.0).as_dict()

# Create model with DEFAULT (fallback)
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
    model_config_dict=model_config,
)

# Create agent
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=tools,
)

agent.reset()

# User message to retrieve system info and print it in python interpreter
user_message = (
    "Retrieve system information using terminal commands like 'uname -a' or 'cat /etc/os-release'. "
    "Then print the retrieved system information in the Python interpreter."
)

response = agent.step(user_message)

print("Agent response:")
print(response.msg.content)

# Print tool calls for debugging
print("Tool calls:")
for call in response.info.get('tool_calls', []):
    print(call)
