import os
from camel.agents.chat_agent import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits.terminal_toolkit.terminal_toolkit import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit

# Setup working directory
working_dir = os.path.abspath("./workspace")

# Create terminal toolkit and get tools
terminal_toolkit = TerminalToolkit(working_directory=working_dir)
terminal_tools = terminal_toolkit.get_tools()

# Create code execution toolkit and get tools
code_exec_toolkit = CodeExecutionToolkit(sandbox="jupyter", verbose=True)
code_exec_tools = code_exec_toolkit.get_tools()

# Combine tools
tools = terminal_tools + code_exec_tools

# Setup model config
model_config = ChatGPTConfig(temperature=0.0).as_dict()
model = ModelFactory.create(model_type="default", model_config_dict=model_config)

# System message
system_message = (
    "You are an assistant with access to terminal and code execution tools. "
    "Retrieve system information using terminal commands and print it using the Python interpreter."
)

# Create agent
agent = ChatAgent(system_message=system_message, model=model, tools=tools)
agent.reset()

# User message to retrieve system info and print it in Python
user_message = (
    "Retrieve system information such as OS details and kernel version using terminal commands. "
    "Then print the retrieved information in a Python interpreter."
)

# Step the agent
response = agent.step(user_message)

# Print the tool calls info
print(response.info['tool_calls'])

# Print the agent's final response content
print(response.msg.content)
