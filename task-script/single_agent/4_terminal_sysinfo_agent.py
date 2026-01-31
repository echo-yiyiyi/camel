from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.types import ModelPlatformType, ModelType

# Create toolkits
terminal_toolkit = TerminalToolkit()
code_exec_toolkit = CodeExecutionToolkit()

# Get tools from both toolkits
terminal_tools = terminal_toolkit.get_tools()
code_exec_tools = code_exec_toolkit.get_tools()

# Combine tools
all_tools = terminal_tools + code_exec_tools

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# Create agent with combined tools
agent = ChatAgent(
    system_message="You are an assistant with terminal and code execution tools.",
    model=model,
    tools=all_tools,
)

# Reset agent
agent.reset()

# Prompt to retrieve system info and print it in Python interpreter
prompt = (
    "Retrieve system information using terminal commands and "
    "print the information in a Python interpreter."
)

# Run agent
response = agent.step(prompt)

# Print response
print(response.msg.content)
