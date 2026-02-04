import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.configs import ChatGPTConfig
from camel.types import ModelPlatformType, ModelType

# Define workspace directory for the toolkit
base_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "workspace")

# Create terminal and code execution toolkits
terminal_tools = TerminalToolkit(working_directory=workspace_dir).get_tools()
code_exec_tools = CodeExecutionToolkit().get_tools()

# Combine tools
all_tools = terminal_tools + code_exec_tools

# Create model config
model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
    model_config_dict=model_config_dict,
)

# System message
sys_msg = (
    "You are a helpful assistant with access to terminal and code execution tools. "
    "Use terminal tools to retrieve system information and then use the code execution tool "
    "to print the system information in a Python interpreter."
)

# Create agent
agent = ChatAgent(system_message=sys_msg, model=model, tools=all_tools)
agent.reset()

# Prompt to retrieve system information and print it
user_prompt = (
    "Retrieve system information such as OS details, CPU info, and memory usage using terminal commands. "
    "Then print the retrieved information using Python code execution."
)

response = agent.step(user_prompt)
print(response.msg.content)
