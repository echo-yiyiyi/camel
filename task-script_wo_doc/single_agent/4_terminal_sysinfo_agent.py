from camel.agents import ChatAgent
from camel.toolkits import TerminalToolkit, CodeExecutionToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Initialize toolkits
terminal_toolkit = TerminalToolkit()
code_exec_toolkit = CodeExecutionToolkit()

# Get tools from toolkits
tools = terminal_toolkit.get_tools() + code_exec_toolkit.get_tools()

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# System message for the agent
sys_msg = (
    "You are an assistant with access to terminal tools and code execution tools. "
    "Your task is to retrieve system information using terminal commands and print it using Python code execution."
)

# Create agent
agent = ChatAgent(sys_msg, model=model, tools=tools)

# Reset agent
agent.reset()

# Define prompt to retrieve system information and print it
prompt = "Retrieve system information using terminal tools and print it in Python interpreter."

# Step the agent with the prompt
response = agent.step(prompt)

# Print the agent's response
print(response.msg.content)

if __name__ == '__main__':
    print(response.msg.content)
